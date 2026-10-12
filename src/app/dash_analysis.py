"""Framework-independent orchestration of the existing HydroTune analysis."""

from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.app.error_correction import ErrorCorrector, select_best_model
from src.app.routing import simulate_imported_model
from src.bma_ensemble import apply_bma_ensemble, calc_bma_weights
from src.hydro_calc import (apply_configured_routing, calibrate_model_fast, calc_kge,
                            calc_mae, calc_nse, calc_pbias, calc_rmse)
from src.llm_api import call_minimax, llm_settings
from src.llm_preanalyzer import SemanticPreAnalyzer
from src.models.registry import ModelRegistry


MODELS = ('HBV Model', 'XAJ Model', 'Tank Model')
COLORS = {'HBV Model': '#08A6AD', 'HBV Simple Model': '#209477', 'XAJ Model': '#5178B8', 'Tank Model': '#D97706', 'BMA集成': '#845EC2'}
UPLOAD_MODES = ('单文件（一场洪水）', '单文件（连续序列）', '多文件（每文件一场洪水）')


def read_hydro_files(files, mapping, routing=False):
    """Parse browser uploads into numeric event records. Validation lives here."""
    events = []
    for name, content in files:
        stream = BytesIO(content)
        frame = pd.read_csv(stream) if name.lower().endswith('.csv') else pd.read_excel(stream)
        frame = frame.rename(columns={source: target for target, source in mapping.items() if source in frame})
        required = {'precip', 'flow'} | ({'upstream'} if routing else set())
        if not required.issubset(frame.columns):
            raise ValueError(f'{name} 缺少列：{", ".join(sorted(required - set(frame.columns)))}')
        if frame.empty:
            raise ValueError(f'{name} 没有数据行。')
        missing = {c: int(pd.to_numeric(frame[c], errors='coerce').isna().sum()) for c in ('precip', 'flow', 'evap') if c in frame}
        for column in ('precip', 'flow', 'evap', 'upstream'):
            if column not in frame and column == 'evap':
                frame[column] = 0.0
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors='coerce')
                if column == 'upstream' and routing and not np.isfinite(frame[column]).all():
                    raise ValueError(f'{name} 上游出库流量须全部为有效数字。')
                if np.isinf(frame[column]).any():
                    raise ValueError(f'{name} / {column} 包含无穷数值。')
                frame[column] = frame[column].fillna(0.0)
        dates = (frame['date'] if 'date' in frame else
                 pd.Series([None] * len(frame), name='date'))
        events.append({'name': name, 'missing': missing, 'dates': dates.reset_index(drop=True),
                       'precip': frame['precip'].to_numpy(float),
                       'evap': frame['evap'].to_numpy(float),
                       'flow': frame['flow'].to_numpy(float),
                       'upstream': frame['upstream'].to_numpy(float) if routing else None})
    if not events:
        raise ValueError('请先上传水文数据。')
    return events


def read_parameter_files(files):
    result = {}
    for model_name, content in files.items():
        if not content:
            continue
        frame = pd.read_csv(BytesIO(content))
        if frame.empty:
            raise ValueError(f'{model_name} 参数文件没有数据。')
        result[model_name] = {name: float(frame.iloc[0][name]) for name in frame
                              if name != '模型'}
        if not all(np.isfinite(value) for value in result[model_name].values()):
            raise ValueError(f'{model_name} 参数必须是有限数值。')
    return result


def _metric_row(name, file_name, category, observed, simulated, warmup):
    if warmup and len(observed) > warmup:
        observed, simulated = observed[warmup:], simulated[warmup:]
    return {'文件': file_name, '类型': category, '模型': name,
            'NSE': float(calc_nse(observed, simulated)),
            'KGE': float(calc_kge(observed, simulated)),
            'RMSE': float(calc_rmse(observed, simulated)),
            'MAE': float(calc_mae(observed, simulated)),
            'PBIAS': float(calc_pbias(observed, simulated))}


def _chart(event, simulations, title, warmup):
    start = warmup if 0 < warmup < len(event['flow']) else 0
    dates = pd.to_datetime(event['dates'], errors='coerce')
    axis = event['dates'] if dates.notna().all() else pd.Series(range(len(event['flow'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=axis[start:], y=event['flow'][start:], name='实测流量', line={'color': '#102a43', 'width': 3}))
    for model_name, values in simulations.items():
        fig.add_trace(go.Scatter(x=axis[start:], y=np.asarray(values)[start:], name=model_name, line={'color': COLORS.get(model_name, '#64748B')}))
    fig.add_trace(go.Bar(x=axis[start:], y=event['precip'][start:], name='降水', yaxis='y2', opacity=.3,
                         marker_color='#38bdf8'))
    fig.update_layout(title=title, template='plotly_white', hovermode='x unified',
                      xaxis_title='时间' if dates.notna().all() else '记录序号',
                      yaxis={'title': '流量 (m³/s)'},
                      yaxis2={'title': '降水 (mm/步)', 'overlaying': 'y', 'side': 'right',
                              'autorange': 'reversed'}, legend={'orientation': 'h'}, autosize=True, margin={'l':55,'r':55,'t':55,'b':45}, font={'family':'Arial, Microsoft YaHei','color':'#172B4D'})
    return fig.to_plotly_json()


def preprocess(files, config):
    """Parse confirmed user inputs and compute statistics without calling an LLM."""
    mode = config['mode']
    if mode not in UPLOAD_MODES:
        raise ValueError('上传模式无效。')
    if mode != UPLOAD_MODES[2] and len(files) != 1:
        raise ValueError('单文件模式须上传一个水文文件。')
    area = float(config['area'])
    if not 0.1 <= area <= 100000:
        raise ValueError('流域面积须在 0.1 至 100000 km² 之间。')
    timestep = config['timestep']
    routing = bool(config.get('routing'))
    events = read_hydro_files(files, config['mapping'], routing)
    all_flow = np.concatenate([event['flow'] for event in events])
    all_precip = np.concatenate([event['precip'] for event in events])
    raw_dates = pd.concat([event['dates'] for event in events], ignore_index=True)
    detected = timestep
    parsed_dates = pd.to_datetime(raw_dates, errors='coerce')
    if parsed_dates.isna().any() or len(parsed_dates) < len(all_flow):
        start = parsed_dates.iloc[0] if len(parsed_dates) and pd.notna(parsed_dates.iloc[0]) else pd.Timestamp('2020-01-01')
        parsed_dates = pd.Series(pd.date_range(start, periods=len(all_flow),
                                                freq='h' if timestep == 'hourly' else 'D'))
    boundaries = None
    if mode == UPLOAD_MODES[2]:
        boundaries = []
        offset = 0
        for event in events:
            boundaries.append({'start_idx': offset, 'end_idx': offset + len(event['flow']) - 1,
                               'peak_idx': offset + int(np.argmax(event['flow'])),
                               'peak_flow': float(np.max(event['flow'])), 'file_name': event['name']})
            offset += len(event['flow'])
    analyzer = SemanticPreAnalyzer(area=area, timestep=timestep)
    if mode == UPLOAD_MODES[1]:
        boundaries = analyzer.detect_flood_events(all_flow, pd.Series(parsed_dates), all_precip)
    preanalysis = analyzer.analyze(
        all_flow, all_precip, pd.Series(parsed_dates), None,
        multi_file_mode=mode == UPLOAD_MODES[2], event_boundaries=boundaries)
    codes = preanalysis['semantic_codes']
    quality = preanalysis['quality']
    raw_count = sum(len(e['flow']) * 2 for e in events)
    raw_missing = sum(e['missing']['flow'] + e['missing']['precip'] for e in events)
    quality.completeness = 100 * (raw_count - raw_missing) / raw_count
    quality_data = {'完整率': float(quality.completeness), '连续性': float(quality.continuity),
                    '降水径流相关': float(quality.correlation), '质量等级': quality.quality_level,
                    '识别场次': len(codes), '检测时间尺度': detected}


    stats = []
    for event in events:
        dates = pd.to_datetime(event['dates'], errors='coerce')
        stats.append({'文件': event['name'], '记录数': len(event['flow']),
                      '开始': str(dates.min()) if dates.notna().all() else '时间列缺失或不可解析',
                      '结束': str(dates.max()) if dates.notna().all() else '按记录顺序显示',
                      '流量缺测': event['missing']['flow'], '降水缺测': event['missing']['precip'],
                      '最大流量 (m³/s)': float(event['flow'].max()),
                      '最小流量 (m³/s)': float(event['flow'].min()),
                      '平均流量 (m³/s)': float(event['flow'].mean())})
    return {'events': events, 'boundaries': boundaries, 'quality': quality_data,
            'stats': stats, 'codes': codes, 'quality_object': quality,
            'semantic_events': [{'场次': c.event_id, '峰值 (m³/s)': float(c.peak_flow),
                                  '洪量 (万m³)': float(c.flood_volume), '峰型': c.peak_shape} for c in codes],
            'preanalysis_report': '', 'charts': [{'title': e['name'], 'figure': _chart(e, {}, e['name'], 0)} for e in events],
            'warnings': ['缺失数值沿用原计算流程填零；流量统计基于处理后序列。原始缺测数量见数据表。'
                         + (' 时间缺失时语义编码沿用原分析器的等步长参考时间，不代表真实日期。' if any(pd.to_datetime(e['dates'], errors='coerce').isna().any() for e in events) else '')]}


def simulate(prepared, config, parameters=None, progress=None):
    progress = progress or (lambda phase, fraction: None)
    mode, area, timestep = config['mode'], float(config['area']), config['timestep']
    warmup_hours = int(config.get('warmup_hours') or 0)
    warmup = warmup_hours if timestep == 'hourly' else warmup_hours // 24
    routing = bool(config.get('routing'))
    fixed_routing = (config.get('routing_parameters') if routing and
                     (config.get('routing_fixed') or config.get('parameter_mode') == 'import') else None)
    events, boundaries = prepared['events'], prepared['boundaries']
    all_flow = np.concatenate([e['flow'] for e in events])
    imported = read_parameter_files(parameters or {}) if config.get('parameter_mode') == 'import' else {}
    if config.get('parameter_mode') == 'import' and not imported:
        raise ValueError('导入参数模式须上传至少一个模型参数文件。')
    if mode == UPLOAD_MODES[2]:
        order = np.random.RandomState(42).permutation(len(events))
        n_calib = max(1, len(events) * 3 // 4)
        calib_names = {events[i]['name'] for i in order[:n_calib]}
        calibration_events = [event for event in events if event['name'] in calib_names]
    elif mode == UPLOAD_MODES[1] and boundaries:
        calibration_events = []
        source = events[0]
        for boundary in boundaries[:5]:
            start = max(0, boundary['start_idx'])
            end = min(len(all_flow), boundary['end_idx'] + 1)
            calibration_events.append({key: (value[start:end] if isinstance(value, np.ndarray) else value)
                                       for key, value in source.items()})
    else:
        calibration_events = events
    if imported:
        calibration_events = events if mode != UPLOAD_MODES[2] else calibration_events

    results = {}
    warnings = []
    for index, model_name in enumerate(config.get('models', MODELS)):
        if imported and model_name not in imported:
            warnings.append(f'{model_name} 未上传参数文件，已跳过。')
            continue
        first = calibration_events[0]
        progress(f'{model_name} · {config.get("algorithm", "导入参数")}', None)
        kwargs = {'spatial_data': {'area': area}, 'timestep': timestep,
                  'upstream_flow': first['upstream'], 'enable_routing': routing,
                  'manual_routing_params': fixed_routing, 'warmup_steps': warmup}
        try:
            if imported:
                params, nse, simulation = simulate_imported_model(
                    model_name, first['precip'], first['evap'], first['flow'],
                    params=imported[model_name], calib_events=calibration_events, **kwargs)
            else:
                params, nse, simulation = calibrate_model_fast(
                    model_name, first['precip'], first['evap'], first['flow'],
                    max_iter=int(config['max_iter']), algorithm=config['algorithm'],
                    algo_params=config.get('algo_params') or {},
                    calib_events=calibration_events if mode != UPLOAD_MODES[0] else None,
                    progress_callback=(lambda fraction: progress(f'{model_name} · 优化器进度 {fraction:.0%}', fraction)) if config['algorithm'] in ('PSO', '遗传算法(GA)', 'SCE-UA') else None, **kwargs)
            results[model_name] = {'params': params, 'nse': float(nse),
                                   'simulated': np.asarray(simulation)}
        except Exception as exc:
            warnings.append(f'{model_name} 运行失败：{exc}')
        progress(f'{model_name} 已完成', None)
    if not results:
        raise ValueError('所有模型均未生成结果。' + ' '.join(warnings))

    rows, charts, file_simulation_results = [], [], {name: {} for name in results}
    for event in events:
        simulated = {}
        category = '率定' if mode != UPLOAD_MODES[2] or event['name'] in calib_names else '验证'
        for model_name, result in results.items():
            try:
                model_params = {key: value for key, value in result['params'].items()
                                if key not in ('k_routing', 'x_routing')}
                values = ModelRegistry.get_model(model_name).run(
                    event['precip'], event['evap'], model_params,
                    {'area': area, 'timestep': timestep}, warmup_steps=warmup)
                if routing:
                    routing_params = fixed_routing or result['params']
                    values = apply_configured_routing(values, event['upstream'], routing_params)
                values = np.asarray(values, dtype=float)
                simulated[model_name] = values
                row = _metric_row(model_name, event['name'], category, event['flow'], values, warmup)
                rows.append(row)
                file_simulation_results[model_name][event['name']] = {
                    'observed': event['flow'], 'simulated': values,
                    'precip': event['precip'], 'nse': row['NSE'], 'is_calib': category == '率定'}
            except Exception as exc:
                warnings.append(f'{event["name"]}/{model_name} 模拟失败：{exc}')
        charts.append({'title': event['name'], 'figure': _chart(event, simulated, event['name'], warmup)})


    return {'results': results, 'metrics': rows, 'charts': charts, 'warnings': warnings,
            'file_results': file_simulation_results, 'calib_names': calib_names if mode == UPLOAD_MODES[2] else {e['name'] for e in events},
            'configuration': {k:v for k,v in config.items() if k != 'llm_settings'},
            'parameters': {name: result['params'] for name, result in results.items()}}


def postprocess(prepared, simulation, config, kind):
    from copy import deepcopy
    mode = config['mode']
    warmup_hours = int(config.get('warmup_hours') or 0)
    warmup = warmup_hours if config['timestep'] == 'hourly' else warmup_hours // 24
    events, results = prepared['events'], simulation['results']
    file_simulation_results, calib_names = simulation['file_results'], simulation['calib_names']
    rows, charts, warnings = [], deepcopy(simulation['charts']), []
    weight_rows = []
    if kind == 'bma':
        for event, chart in zip(events, charts):
            available = {name: files[event['name']] for name, files in file_simulation_results.items() if event['name'] in files}
            if len(available) < 2:
                warnings.append(f"{event['name']} 至少需要两个有效模型才能运行 BMA。")
                continue
            weights = calc_bma_weights([r['nse'] for r in available.values()])
            ensemble = apply_bma_ensemble([r['simulated'] for r in available.values()], weights)
            category = '率定' if event['name'] in calib_names else '验证'
            rows.append(_metric_row('BMA集成', event['name'], category, event['flow'], ensemble, warmup))
            for name, weight in zip(available, weights):
                weight_rows.append({'文件': event['name'], '模型': name, '权重': float(weight)})
            start = warmup if 0 < warmup < len(ensemble) else 0
            chart['figure']['data'].append(go.Scatter(x=chart['figure']['data'][0]['x'], y=ensemble[start:], name='BMA集成', line={'color':COLORS['BMA集成']}).to_plotly_json())
    corrected_flow = {}
    if kind == 'xgb':
        best_name, _ = select_best_model(results)
        if best_name:
            training = [event for event in events if mode != UPLOAD_MODES[2] or event['name'] in calib_names]
            test_events = events
            try:
                if mode == UPLOAD_MODES[2] and len(training) < 3:
                    raise ValueError('率定场次少于 3 场，无法训练 XGBoost。')
                xgb_params = {'objective': 'reg:squarederror', 'max_depth': int(config.get('xgb_depth', 10)),
                              'learning_rate': float(config.get('xgb_rate', .05)),
                              'n_estimators': int(config.get('xgb_estimators', 300)),
                              'random_state': int(config.get('xgb_seed', 2025))}
                quantile = float(config.get('xgb_quantile', .9))
                corrector = ErrorCorrector(xgb_params=xgb_params,
                                           enable_error_clipping=bool(config.get('xgb_clip', True)),
                                           error_clip_quantiles=((1-quantile)/2, (1+quantile)/2),
                                           max_correction_ratio=float(config.get('xgb_ratio', .45)))
                train_sim = [file_simulation_results[best_name][event['name']]['simulated'] for event in training]
                if mode == UPLOAD_MODES[2]:
                    start = warmup if warmup and all(len(e['flow']) > warmup for e in training) else 0
                    corrector.train_by_events([e['precip'][start:] for e in training],
                                              [e['flow'][start:] for e in training],
                                              [s[start:] for s in train_sim], test_event_ratio=0)
                else:
                    corrector.train(training[0]['precip'], training[0]['flow'], train_sim[0])
                for event in test_events:
                    source = file_simulation_results[best_name][event['name']]['simulated']
                    corrected = (corrector.correct_with_true_error_lags(event['precip'], event['flow'], source)
                                 if mode == UPLOAD_MODES[2] else corrector.correct(event['precip'], source))
                    category = ('率定' if event['name'] in calib_names else '验证') if mode == UPLOAD_MODES[2] else '率定'
                    rows.append(_metric_row(best_name + '-XGB', event['name'], category,
                                            event['flow'], corrected, warmup))
                    corrected_flow[event['name']] = {'observed': event['flow'].tolist(),
                                                      'simulated': source.tolist(),
                                                      'corrected': corrected.tolist()}
                    for chart in charts:
                        if chart['title'] == event['name']:
                            start = warmup if 0 < warmup < len(corrected) else 0
                            chart['figure']['data'].append(go.Scatter(x=chart['figure']['data'][0]['x'], y=corrected[start:], name=best_name + '-XGB', line={'color':'#C25D8D'}).to_plotly_json())
            except Exception as exc:
                warnings.append(f'XGBoost 误差校正未完成：{exc}')


    return {'metrics': rows, 'charts': charts, 'warnings': warnings, 'weights': weight_rows,
            'corrected_flow': corrected_flow}


def summary(rows):
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    names = ['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS']
    for name in names:
        frame[name] = frame[name].where(frame[name] > -900)
    return frame.groupby(['类型','模型'], dropna=False)[names].mean().reset_index().to_dict('records')


def generate_report(prepared, simulation=None, posts=None, settings=None, kind='综合', basin=None):
    import json
    from plotly.utils import PlotlyJSONEncoder
    evidence = {'数据概况': prepared['stats'], '洪水统计': prepared['semantic_events'],
                '说明': prepared['warnings'], '阶段': kind}
    evidence['流域概况'] = {k:v for k,v in basin.items() if k!='figure'} if basin else '未提供空间数据'
    if simulation and kind != '预分析':
        rows = list(simulation['metrics'])
        for post in (posts or {}).values():
            rows += post['metrics']
        evidence.update({'模型指标': summary(rows), '率定配置': simulation['configuration'],
                         '参数': simulation['parameters']})
    evidence['后处理'] = {name: {'权重': p['weights'], '指标': summary(p['metrics']), '说明': p['warnings']} for name,p in (posts or {}).items()}
    evidence['评估条件'] = 'BMA 按本场观测 NSE 计算权重；多场 XGB 校正使用真实误差滞后，属于回顾性评估。未提供收敛轨迹或参数敏感性分析。'
    text = json.dumps(evidence, ensure_ascii=False, cls=PlotlyJSONEncoder)
    with llm_settings(settings):
        report = call_minimax('请根据以下真实计算证据生成'+kind+' Markdown 分析报告。标明未运行步骤和缺少证据的项目；不得编造指标、排名、收敛轨迹或敏感性结论。\n'+text)
    if report.startswith('[ERROR]'):
        return '### 智能分析暂不可用\n\n'+report+'\n\n确定性统计结果仍可在工作台查看。'
    return report


def analyze(files, config, parameters=None, progress=None):
    """Compatibility entrypoint for the original complete analysis workflow."""
    prepared = preprocess(files, config)
    simulation = simulate(prepared, config, parameters, progress)
    posts = {'bma': postprocess(prepared, simulation, config, 'bma')}
    if config.get('xgb'):
        posts['xgb'] = postprocess(prepared, simulation, config, 'xgb')
    rows = simulation['metrics'] + [r for p in posts.values() for r in p['metrics']]
    charts = combine_charts(simulation, posts)
    report = generate_report(prepared, simulation, posts, config.get('llm_settings'))
    return {'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'configuration': simulation['configuration'], 'quality': prepared['quality'],
            'semantic_events': prepared['semantic_events'], 'preanalysis_report': prepared['preanalysis_report'],
            'parameters': simulation['parameters'], 'metrics': rows, 'summary': summary(rows),
            'charts': charts, 'corrected_flow': posts.get('xgb', {}).get('corrected_flow', {}),
            'report': report, 'warnings': prepared['warnings'] + simulation['warnings'] + [w for p in posts.values() for w in p['warnings']]}


def combine_charts(simulation, posts):
    from copy import deepcopy
    charts = deepcopy(simulation['charts'])
    for post in posts.values():
        for chart, extra in zip(charts, post['charts']):
            original_names = {t['name'] for t in chart['figure']['data']}
            chart['figure']['data'].extend(t for t in extra['figure']['data'] if t['name'] not in original_names)
    return charts

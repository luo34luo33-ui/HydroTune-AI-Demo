"""Deterministic five-model comparison shared by UI and LLM reports."""
import numpy as np

METRICS = ('nse', 'kge', 'rmse', 'pbias')
MODELS = ('XAJ', 'HBV', 'Tank', 'BMA', 'XGB')


def model_family(name):
    text = str(name).lower()
    if 'xgb' in text:
        return 'XGB'
    for family in MODELS[:-1]:
        if family.lower() in text:
            return family
    return None


def normalize_comparison(comparison_results, file_simulation_results):
    """Accept canonical records or existing Chinese/English chart summary rows."""
    if comparison_results is None:
        source = [dict(result, model_name=model, file_name=file)
                  for model, files in file_simulation_results.items()
                  for file, result in files.items()]
    else:
        source = comparison_results
    records = {}
    for row in source:
        name = row.get('model_name', row.get('模型', row.get('Model', '')))
        family = model_family(name)
        if family is None:
            continue
        file = row.get('file_name', row.get('文件', row.get('File')))
        if file is None:
            continue
        kind = row.get('类型', row.get('Type'))
        if 'is_calib' in row:
            is_calib = bool(row['is_calib'])
        elif kind in ('率定', 'Calibration'):
            is_calib = True
        elif kind in ('验证', 'Validation'):
            is_calib = False
        else:
            # Unknown grouping must not silently become validation evidence.
            continue
        values = {}
        for metric in METRICS:
            try:
                value = float(row.get(metric, row.get(metric.upper(), np.nan)))
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value) or (metric == 'nse' and value == -9999):
                value = np.nan
            values[metric] = value
        base = row.get('base_model') or (str(name).removesuffix('-XGB') if family == 'XGB' else None)
        record = dict(values, model=family, model_name=str(name), file_name=str(file),
                      is_calib=is_calib, base_model=base)
        record['valid'] = all(np.isfinite(values[m]) for m in METRICS)
        records[(family, str(file), is_calib)] = record
    return list(records.values())


def summarize(rows):
    """Equal event weights; undefined metrics never become zero."""
    if not rows:
        return None
    return dict(n=len(rows), nse=np.mean([r['nse'] for r in rows]),
                std=np.std([r['nse'] for r in rows]),
                kge=np.mean([r['kge'] for r in rows]),
                rmse=np.mean([r['rmse'] for r in rows]),
                pbias=np.mean([r['pbias'] for r in rows]),
                abs_pbias=np.mean([abs(r['pbias']) for r in rows]))


def build_comparison_report(file_data_list, calibration_results, file_simulation_results,
                            call_llm, warmup_hours, comparison_results, language):
    en = language == 'en'
    def tr(zh, eng):
        return eng if en else zh
    rows = normalize_comparison(comparison_results, file_simulation_results)
    valid = [r for r in rows if r['valid']]
    labels = {m: m for m in MODELS}
    corrected = next((r for r in rows if r['model'] == 'XGB'), None)
    labels['XGB'] = corrected['model_name'] if corrected else tr('最优模型＋XGB', 'Best model + XGB')
    missing = tr('未生成／数据不足', 'Not generated / insufficient data')
    def fmt(value):
        return f'{value:.4f}' if np.isfinite(value) else 'N/A'
    def name(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = [tr('# 水文模型综合比较报告', '# Hydrological Model Comparison Report'), '',
             tr(f'分析场次：{len(file_data_list)}；预热期：{warmup_hours} 小时（沿用图表处理）。',
                f'Events: {len(file_data_list)}; warm-up: {warmup_hours} hours (same processing as charts).'), '',
             tr('## 1. 五类模型指标', '## 1. Five-model metrics'), '',
             tr('场次等权汇总；有效场次须同时具有有限的 NSE、KGE、RMSE、PBIAS。缺失指标不填零。',
                'Events receive equal weights. Valid events require finite NSE, KGE, RMSE and PBIAS. Missing metrics are not replaced by zero.'), '',
             tr('| 模型 | 分组 | 有效场次数 | 平均 NSE | NSE 标准差 | 平均 KGE | 平均 RMSE | 平均 PBIAS (%) |',
                '| Model | Group | Valid events | Mean NSE | NSE std | Mean KGE | Mean RMSE | Mean PBIAS (%) |'),
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        for calib in (True, False):
            subset = [r for r in valid if r['model'] == model and r['is_calib'] == calib]
            stats = summarize(subset)
            group = tr('率定', 'Calibration') if calib else tr('验证', 'Validation')
            if stats:
                values = ' | '.join(fmt(stats[k]) for k in ('nse', 'std', 'kge', 'rmse', 'pbias'))
                lines.append(f"| {name(labels[model])} | {group} | {stats['n']} | {values} |")
            else:
                lines.append(f'| {name(labels[model])} | {group} | {missing} | N/A | N/A | N/A | N/A | N/A |')
    lines += ['', tr('## 2. 共同验证场次比较', '## 2. Common validation comparison'), '']
    sets = [{r['file_name'] for r in valid if r['model'] == m and not r['is_calib']} for m in MODELS]
    common = set.intersection(*sets)
    if common:
        lines += [tr('五类模型共同有效的验证场次：', 'Valid validation events shared by all five models: ') +
                  ', '.join(name(f) for f in sorted(common)), '',
                  tr('| 模型 | 平均 NSE | NSE 标准差 | 平均 KGE | 平均 RMSE | 平均绝对 PBIAS (%) |',
                     '| Model | Mean NSE | NSE std | Mean KGE | Mean RMSE | Mean absolute PBIAS (%) |'),
                  '|---|---:|---:|---:|---:|---:|']
        scores = {m: summarize([r for r in valid if r['model'] == m and not r['is_calib'] and r['file_name'] in common]) for m in MODELS}
        for m in sorted(MODELS, key=lambda m: -scores[m]['nse']):
            s = scores[m]
            lines.append(f"| {name(labels[m])} | " + ' | '.join(fmt(s[k]) for k in ('nse', 'std', 'kge', 'rmse', 'abs_pbias')) + ' |')
        top = max(s['nse'] for s in scores.values())
        winners = [name(labels[m]) for m in MODELS if abs(scores[m]['nse'] - top) < 1e-12]
        lines += ['', tr('共同验证场次平均 NSE 最高：', 'Highest mean NSE on common validation events: ') + ', '.join(winners) +
                  tr('。需结合其他指标及下述评估限制解读。', '. Interpret alongside other metrics and the evaluation limitations below.')]
    elif not any(not r['is_calib'] for r in rows):
        lines.append(tr('没有验证数据，仅评价率定表现；不生成验证排名或泛化能力结论。',
                        'No validation data. Assess calibration only; do not produce validation rankings or generalization conclusions.'))
    else:
        lines.append(tr('五类模型没有共同有效验证场次，不生成统一排名。各模型的独立汇总不可直接用于统一排名。',
                        'No valid validation events shared by all five models. No unified ranking; separate model summaries must not be used as a unified ranking.'))
    lines += ['', tr('## 3. 误差校正相对基础模型的变化', '## 3. Error correction versus its base model'), '']
    index = {(r['model'], r['file_name'], r['is_calib']): r for r in valid}
    for calib in (True, False):
        pairs = [(r, index.get((model_family(r['base_model']), r['file_name'], calib)))
                 for r in valid if r['model'] == 'XGB' and r['is_calib'] == calib]
        pairs = [(r, base) for r, base in pairs if base is not None]
        group = tr('率定', 'Calibration') if calib else tr('验证', 'Validation')
        if not pairs:
            lines.append(f'- {group}: {missing}')
            continue
        delta = {k: np.mean([r[k] - base[k] for r, base in pairs]) for k in METRICS[:-1]}
        delta['abs_pbias'] = np.mean([abs(r['pbias']) - abs(base['pbias']) for r, base in pairs])
        direction = tr('提升', 'improved') if delta['nse'] > 0 else (tr('退化', 'degraded') if delta['nse'] < 0 else tr('持平', 'unchanged'))
        lines.append(f"- {group}: n={len(pairs)}; ΔNSE={delta['nse']:+.4f} ({direction}); ΔKGE={delta['kge']:+.4f}; ΔRMSE={delta['rmse']:+.4f}; Δ|PBIAS|={delta['abs_pbias']:+.4f}")
    lines += ['', tr('Δ 为校正后减校正前；NSE、KGE 增加为改善，RMSE、|PBIAS| 减少为改善。',
                     'Deltas are corrected minus original; higher NSE/KGE and lower RMSE/absolute PBIAS indicate improvement.'), '',
              tr('## 4. 各场次结果', '## 4. Per-event results'), '',
              tr('| 场次 | 分组 | 模型 | NSE | KGE | RMSE | PBIAS (%) |',
                 '| Event | Group | Model | NSE | KGE | RMSE | PBIAS (%) |'),
              '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        group = tr('率定', 'Calibration') if r['is_calib'] else tr('验证', 'Validation')
        lines.append(f"| {name(r['file_name'])} | {group} | {name(r['model_name'])} | " +
                     ' | '.join(fmt(r[m]) for m in METRICS) + ' |')
    lines += ['', tr('## 5. 评估限制', '## 5. Evaluation limitations'), '',
              tr('BMA 权重由本场观测指标确定；XGB 校正使用真实误差滞后信息。以上为现有条件下的回顾性比较，不能直接代表独立预报能力。NSE 标准差用于描述场次波动，不使用可能因均值接近零而失真的变异系数。',
                 'BMA weights use observed metrics from the evaluated event; XGB correction uses true error lags. This is a retrospective comparison under existing conditions, not independent forecasting validation. NSE standard deviation describes event variability; coefficient of variation is not used near zero means.'), '',
              tr('## 6. 智能综合分析', '## 6. Intelligent comparative analysis'), '']
    prompt = tr('请根据下列数据用中文分析 XAJ、HBV、Tank、BMA、最优模型＋XGB。比较单模型差异、BMA效果、校正效果、稳定性，并给出有依据的建议。只使用给定数据；缺失项不得虚构；遵守共同验证场次排名规则和评估限制。无验证数据时不得推断泛化能力。控制在800字以内。\n\n',
                'Analyze XAJ, HBV, Tank, BMA and best model + XGB in English. Compare individual models, ensemble and correction effects, stability, and evidence-based recommendations. Use only supplied data; do not invent missing results. Respect common-validation ranking rules and evaluation limitations. Do not infer generalization without validation. Limit to 1200 words.\n\n') + '\n'.join(lines[:-2])
    parameter_lines = []
    for model, result in calibration_results.items():
        params = result.get('params', {})
        parameter_lines.append(f'{name(model)}: ' + ', '.join(
            f'{name(k)}={v}' for k, v in params.items() if np.isscalar(v)))
    if parameter_lines:
        prompt += '\n\n' + tr('率定参数（用于辅助解释，不替代验证指标）：',
                               'Calibration parameters (context only, not validation evidence):') + '\n' + '\n'.join(parameter_lines)
    try:
        result = call_llm(prompt) if call_llm else None
    except Exception:
        # API failures must not discard the deterministic comparison report.
        result = None
    if isinstance(result, str) and result.strip() and not result.startswith('[ERROR]'):
        lines.append(result)
    else:
        lines.append(tr('LLM 暂时不可用。以上指标表、共同验证比较及校正差值为实际数据生成的小结。',
                        'LLM unavailable. The tables, common-validation comparison and correction deltas above are a summary generated from actual results.'))
    lines += ['', '---', tr('*本报告由 HydroTune 系统自动生成*', '*Report generated by HydroTune*')]
    return '\n'.join(lines)

"""Behavior checks for stage separation, task ownership and Dash callbacks."""
from copy import deepcopy
from io import BytesIO
from unittest.mock import patch
import threading

import numpy as np
import pandas as pd
import pytest

from src.app import dash_analysis as analysis
from src.dash_ui.application import create_app
from src.dash_ui import tasks
from src.dash_ui.configuration import ALGORITHMS, CONFIG_IDS, parse_config
from src.dash_ui.layouts import layout
from src.models.registry import ModelRegistry


@pytest.fixture
def config():
    return dict(mode=analysis.UPLOAD_MODES[0],area=150.7944,timestep='daily',warmup_hours=0,
                parameter_mode='import',models=['Tank Model'],algorithm='PSO',max_iter=5,algo_params={'n_particles':10},
                mapping={key:key for key in ('date','precip','evap','flow','upstream')},routing=False,
                routing_fixed=False,routing_parameters={'k_routing':2.5,'x_routing':.25},
                xgb_depth=3,xgb_rate=.05,xgb_estimators=50,xgb_seed=2025,xgb_quantile=.9,xgb_ratio=.45,xgb_clip=True)


@pytest.fixture
def files():
    frame=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=32),'precip':np.tile([0,5,0,0],8),
                        'evap':.2,'flow':np.linspace(2,15,32)})
    return [('event.csv',frame.to_csv(index=False).encode())]


def imported_parameters():
    return {name:pd.DataFrame([model.default_params]).to_csv(index=False).encode()
            for name,model in ModelRegistry.get_all_models().items()}


def test_confirmation_only_computes_preanalysis(files,config):
    with patch.object(analysis,'call_minimax',side_effect=AssertionError('unexpected LLM call')),patch.object(
        analysis,'calibrate_model_fast',side_effect=AssertionError('unexpected calibration')):
        prepared=analysis.preprocess(files,config)
    assert prepared['stats'][0]['记录数']==32
    assert len(prepared['charts'])==1
    assert prepared['preanalysis_report']==''


def test_preanalysis_excludes_each_file_warmup_and_keeps_full_chart(config):
    from src.dash_ui.semantics import portrait
    config.update(mode=analysis.UPLOAD_MODES[2],timestep='hourly',warmup_hours=3)
    frame=pd.DataFrame(dict(date=pd.date_range('2026-01-01',periods=8,freq='h'),
                            flow=[999,800,700,5,3,7,20,6],precip=[100,100,100,1,2,3,4,5]))
    raw=frame.to_csv(index=False).encode()
    prepared=analysis.preprocess([('a.csv',raw),('b.csv',raw)],config)
    assert [b['start_idx'] for b in prepared['semantic_boundaries']]==[3,11]
    assert [c.peak_flow for c in prepared['codes']]==[20,20]
    assert [c.duration for c in prepared['codes']]==[5,5]
    assert [c.total_precip for c in prepared['codes']]==[15,15]
    assert len(prepared['charts'][0]['figure']['data'][0]['x'])==8
    assert prepared['charts'][0]['figure']['layout']['shapes'][0]['fillcolor']=='#E5E7EB'
    view=portrait(prepared,1)
    assert '2026-01-01 03:00' in view['period']
    markers=view['figure']['data'][-1]
    assert list(markers['text'])==['起涨点','主峰点','终点']
    assert pd.Timestamp(markers['x'][0])==pd.Timestamp('2026-01-01 04:00')
    assert pd.Timestamp(markers['x'][1])==pd.Timestamp('2026-01-01 06:00')
    assert any(s['fillcolor']=='#E5E7EB' for s in view['figure']['layout']['shapes'])
    config['warmup_hours']=24
    prepared=analysis.preprocess([('a.csv',raw)],config)
    assert prepared['codes']==[] and prepared['semantic_boundaries']==[]
    assert prepared['stats'][0]['记录数']==8


def test_supplementary_basin_image_is_validated_and_exported(config):
    import base64
    from PIL import Image
    from src.dash_ui.snapshot import build_snapshot,capture_state
    from test_dash_snapshot import payload
    app=create_app()
    buffer=BytesIO()
    Image.new('RGB',(20,20),'#08A6AD').save(buffer,format='PNG')
    image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    upload=app.callback_map['basin-display-message.children']['callback'].__wrapped__
    with app.server.test_request_context('/'):
        state=tasks.current()
        state['config']=config
        with patch('src.dash_ui.callbacks.ctx') as context:
            context.triggered_id='basin-display-upload'
            assert '已展示' in upload(image,None,'basin.png')
            assert state['prepared'] is None and state['job_id']==0
            data=payload(build_snapshot(capture_state(state)))
            assert image in str(data['views']['data'])
            assert '补充展示图' in str(data)
            assert '失败' in upload('data:image/png;base64,YmFk',None,'bad.png')
            context.triggered_id='remove-basin-display'
            upload(None,1,None)
            assert state['basin_display'] is None


def test_semantic_portraits_link_duplicate_event_dates_to_correct_files(files,config):
    from src.dash_ui.semantics import options, portrait, code_rows
    frame=pd.read_csv(BytesIO(files[0][1]))
    frame['date']=pd.date_range('2020-01-01',periods=32,freq='h')
    config.update(mode=analysis.UPLOAD_MODES[2],timestep='hourly')
    first=frame.to_csv(index=False).encode()
    frame['flow']*=3
    second=frame.to_csv(index=False).encode()
    with patch.object(analysis.SemanticPreAnalyzer,'generate_base64_plot',side_effect=AssertionError('unexpected LLM preparation')):
        prepared=analysis.preprocess([('first.csv',first),('second.csv',second)],config)
    assert prepared['codes'][0].event_id==prepared['codes'][1].event_id
    assert [item['value'] for item in options(prepared)]==[0,1]
    first_view,second_view=portrait(prepared,0),portrait(prepared,1)
    assert first_view['source']=='first.csv' and second_view['source']=='second.csv'
    assert second_view['metrics'][0][1]==f'{frame.flow.max():,.2f}'
    assert first_view['metrics'][2][2]=='小时'
    assert '2020-01-02 07:00' in first_view['period']
    assert pd.Timestamp(second_view['figure']['data'][0]['x'][0])==pd.Timestamp('2020-01-01')
    assert len(code_rows(prepared)[0])==1+len(prepared['codes'][0].to_dict())


def test_semantic_portrait_preserves_detected_boundaries_and_missing_dates(files,config):
    from src.dash_ui.semantics import portrait
    boundaries=[dict(start_idx=3,end_idx=12,peak_idx=12,peak_flow=7)]
    with patch.object(analysis.SemanticPreAnalyzer,'detect_flood_events',return_value=boundaries):
        prepared=analysis.preprocess(files,config)
    view=portrait(prepared,0)
    assert len(view['figure']['data'][0]['x'])==10
    assert pd.Timestamp(view['figure']['data'][0]['x'][0])==pd.Timestamp('2020-01-04')
    assert view['metrics'][2][1:] == ('10','天')
    prepared['events'][0]['dates']=pd.Series([None]*32)
    view=portrait(prepared,0)
    assert view['figure']['layout']['xaxis']['type']=='linear'
    assert '不代表真实日期' in view['period']


def test_semantic_selection_is_read_only_and_exported(files,config):
    from src.dash_ui.snapshot import build_snapshot
    config['mode']=analysis.UPLOAD_MODES[2]
    prepared=analysis.preprocess(files+[('second.csv',files[0][1])],config)
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        state.update(prepared=prepared,confirmed_config=config)
        before=(state['revision'],state['input_version'],state['job_id'])
        choose=app.callback_map['semantic-detail.children']['callback'].__wrapped__
        with patch.object(tasks,'submit',side_effect=AssertionError('selection submitted a job')):
            assert choose(1)
            assert choose(99) is __import__('dash').no_update
        assert (state['revision'],state['input_version'],state['job_id'])==before
        assert state['semantic_selection']==1
        exported=build_snapshot(state)
    from test_dash_snapshot import payload
    data=payload(exported)
    assert data['semantic_selection']==1
    assert len(data['portraits'])==2
    assert '完整语义编码表' in str(data) and '代表性规则评分' in str(data)
    assert 'second.csv' in str(data) and '非洪水等级或重现期' in str(data)


def test_process_chart_uses_real_dates_and_preserves_missing_time_fallback(files,config):
    prepared=analysis.preprocess(files,config)
    figure=prepared['charts'][0]['figure']
    assert figure['layout']['xaxis']['type']=='date'
    assert figure['layout']['xaxis']['nticks']==7
    assert pd.Timestamp(figure['data'][0]['x'][1])-pd.Timestamp(figure['data'][0]['x'][0])==pd.Timedelta(days=1)
    event=prepared['events'][0]
    hourly={**event,'dates':pd.Series(pd.date_range('2020-01-01',periods=32,freq='h'))}
    chart=analysis._chart(hourly,{},'hourly',0)
    assert pd.Timestamp(chart['data'][0]['x'][1])-pd.Timestamp(chart['data'][0]['x'][0])==pd.Timedelta(hours=1)
    fallback=analysis._chart({**event,'dates':pd.Series([None]*32)}, {},'missing',0)
    assert fallback['layout']['xaxis']['type']=='linear'
    assert fallback['layout']['xaxis']['title']['text']=='记录序号'


def test_raw_missingness_survives_existing_zero_fill(files,config):
    frame=pd.read_csv(BytesIO(files[0][1]))
    frame.loc[2,'flow']=np.nan
    result=analysis.preprocess([('missing.csv',frame.to_csv(index=False).encode())],config)
    assert result['stats'][0]['流量缺测']==1
    assert result['events'][0]['flow'][2]==0


@pytest.mark.parametrize('algorithm',list(ALGORITHMS))
def test_algorithm_and_parameters_reach_calibration(files,config,algorithm):
    config.update(parameter_mode='calibrate',algorithm=algorithm,
                  algo_params={key:default for key,_,default,*rest in ALGORITHMS[algorithm]})
    prepared=analysis.preprocess(files,config)
    with patch.object(analysis,'calibrate_model_fast',return_value=(ModelRegistry.get_model('Tank Model').default_params,.5,np.ones(32))) as calculate:
        result=analysis.simulate(prepared,config)
    assert calculate.call_args.kwargs['algorithm']==algorithm
    assert calculate.call_args.kwargs['algo_params']==config['algo_params']
    assert set(result['parameters'])=={'Tank Model'}


def test_real_simulation_and_bma_weights(files,config):
    config['models']=ModelRegistry.list_models()
    prepared=analysis.preprocess(files,config)
    with patch.object(analysis,'call_minimax',side_effect=AssertionError('unexpected LLM')):
        simulation=analysis.simulate(prepared,config,imported_parameters())
        post=analysis.postprocess(prepared,simulation,config,'bma')
    assert len(simulation['parameters'])==len(config['models'])
    assert sum(row['权重'] for row in post['weights'])==pytest.approx(1)
    assert post['metrics'][0]['模型']=='BMA集成'
    assert all(row['模型']!='BMA集成' for row in simulation['metrics'])


def test_real_pso_progress_and_no_llm(files,config):
    config['parameter_mode']='calibrate'
    prepared=analysis.preprocess(files,config)
    progress=[]
    result=analysis.simulate(prepared,config,progress=lambda phase,fraction:progress.append(fraction))
    assert result['metrics']
    assert any(value is not None and value>0 for value in progress)


@pytest.mark.parametrize('missing_result',[False,True])
def test_bma_uses_global_mean_nse_weights_for_every_event(files,config,missing_result):
    config.update(mode=analysis.UPLOAD_MODES[2],models=['Tank Model','HBV Model'])
    prepared=analysis.preprocess(files+[('second.csv',files[0][1])],config)
    simulation=analysis.simulate(prepared,config,imported_parameters())
    names=[event['name'] for event in prepared['events']]
    for model,scores,value in [('Tank Model',[.9,.1],10.),('HBV Model',[.1,.9],20.)]:
        for name,score in zip(names,scores):
            simulation['file_results'][model][name].update(nse=score,simulated=np.full(32,value))
    if missing_result:
        del simulation['file_results']['HBV Model'][names[1]]
    post=analysis.postprocess(prepared,simulation,config,'bma')
    expected=analysis.calc_bma_weights([.5,.1 if missing_result else .5])
    assert [row['权重'] for row in post['weights']]==pytest.approx(expected)
    assert [row['场次平均 NSE'] for row in post['weights']]==pytest.approx([.5,.1 if missing_result else .5])
    assert all('文件' not in row for row in post['weights'])
    assert len(post['metrics'])==(1 if missing_result else 2)
    for chart in post['charts']:
        traces=[trace for trace in chart['figure']['data'] if trace['name']=='BMA集成']
        if traces:
            assert np.asarray(traces[0]['y'])==pytest.approx(np.full(32,10*expected[0]+20*expected[1]))
    if missing_result:
        assert '统一模型权重' in post['warnings'][0]


def test_only_one_model_has_explanatory_bma_state(files,config):
    prepared=analysis.preprocess(files,config)
    simulation=analysis.simulate(prepared,config,imported_parameters())
    post=analysis.postprocess(prepared,simulation,config,'bma')
    assert not post['weights']
    assert '两个' in post['warnings'][0]


def test_llm_failure_keeps_results_and_no_key_in_report(files,config):
    prepared=analysis.preprocess(files,config)
    with patch.object(analysis,'call_minimax',return_value='[ERROR] offline'):
        text=analysis.generate_report(prepared,settings={'provider':'minimax','api_key':'PRIVATE-KEY'})
    assert '暂不可用' in text
    assert 'PRIVATE-KEY' not in text
    assert prepared['stats']


def test_stage_report_requests_analysis_first_and_summary_last(files,config):
    prepared=analysis.preprocess(files,config)
    simulation=analysis.simulate(prepared,config,imported_parameters())
    with patch.object(analysis,'call_minimax',return_value='## 模型表现\n真实结果\n\n## 小结\n继续验证。') as call:
        report=analysis.generate_report(prepared,simulation,kind='阶段')
    prompt=call.call_args.args[0]
    assert '直接从模型模拟表现开始' in prompt
    assert '不要设置“报告范围与证据边界”' in prompt
    assert '不逐行复述文件清单' in prompt
    assert call.call_args.kwargs==dict(max_output_tokens=8192,continuations=2)
    assert report.endswith('继续验证。')


def test_preanalysis_report_has_only_flood_evidence_even_after_simulation(files,config):
    import json
    prepared=analysis.preprocess(files,config)
    with patch.object(analysis,'call_minimax',return_value='## 小结\n洪水特征。') as call:
        analysis.generate_report(prepared,kind='预分析')
        before=call.call_args.args[0]
        # Later-stage inputs are deliberately unusable: preanalysis must never read them.
        analysis.generate_report(prepared,simulation=object(),posts=object(),kind='预分析')
        after=call.call_args.args[0]
    assert before==after
    evidence=json.loads(after.rsplit('\n',1)[1])
    assert not any(word in json.dumps(evidence,ensure_ascii=False) for word in
                   ('模型','模拟状态','率定','验证','BMA','XGB','后处理','评估条件'))
    assert evidence['洪水语义特征'][0]['peak_flow']==prepared['codes'][0].peak_flow
    assert evidence['数据质量']==prepared['quality']
    assert '不涉及任何水文模型' in after and '重点说清楚数据呈现了什么特征' in after
    assert call.call_args.kwargs==dict(max_output_tokens=8192,continuations=2)


def test_report_counts_distinguish_48_inputs_from_42_successful_evaluations(files,config):
    import json
    prepared=analysis.preprocess(files,config)
    source=prepared['events'][0]
    prepared['events']=[{**source,'name':f'event{i}.csv'} for i in range(48)]
    prepared['codes']=[prepared['codes'][0]]*48  # Same date IDs still represent 48 events.
    config['mode']=analysis.UPLOAD_MODES[2]
    names=[event['name'] for event in prepared['events']]
    successful=names[:32]+names[38:]
    rows=[{'模型':'Tank Model','文件':name,'类型':'率定' if name in names[:36] else '验证',
           **dict.fromkeys(('NSE','KGE','RMSE','MAE','PBIAS'),1.)} for name in successful]
    simulation=dict(configuration=config,calib_names=set(names[:36]),metrics=rows,parameters={},
                    warnings=['六场模型模拟失败。'])
    counts=analysis.report_counts(prepared,simulation)
    assert counts['识别洪水场次']==counts['输入文件数']==48
    assert counts['记录总数']==48*32
    assert (counts['率定文件数'],counts['验证文件数'])==(36,12)
    assert counts['各模型实际评价数量']==[{'模型':'Tank Model','成功生成指标的文件数':42,'率定文件数':32,'验证文件数':10}]
    with patch.object(analysis,'call_minimax',return_value='## 小结\n完成。') as call:
        analysis.generate_report(prepared,simulation)
    prompt=call.call_args.args[0]
    evidence=json.loads(prompt.rsplit('\n',1)[1])
    assert evidence['程序统计数量']==counts
    assert evidence['模型运行说明']==simulation['warnings']
    assert '不得把某模型仅成功评价的数量写成全部实测洪水数量' in prompt
    with patch.object(analysis,'call_minimax',return_value='## 小结\n完成。') as call:
        analysis.generate_report(prepared,simulation,kind='预分析')
    evidence=json.loads(call.call_args.args[0].rsplit('\n',1)[1])
    assert '各模型实际评价数量' not in evidence['程序统计数量']
    assert evidence['程序统计数量']['识别洪水场次']==48


def test_continuous_report_counts_do_not_call_one_file_one_flood(files,config):
    prepared=analysis.preprocess(files,config)
    prepared['codes']*=3
    simulation=analysis.simulate(prepared,config,imported_parameters())
    counts=analysis.report_counts(prepared,simulation)
    assert counts['输入文件数']==1 and counts['识别洪水场次']==len(prepared['codes'])
    assert '不等于识别洪水场次' in counts['评价计数单位']
    assert counts['验证文件数']==0
    assert counts['各模型实际评价数量'][0]['成功生成指标的文件数']==1


def test_sessions_isolated_and_stale_task_cannot_publish(files,config):
    app=create_app()
    with app.server.test_request_context('/'):
        first=tasks.current()
        tasks.configure(first,config)
    with app.server.test_request_context('/'):
        second=tasks.current()
    assert first is not second
    first['pending']={'file':dict(kind='hydro',name=files[0][0],content=files[0][1])}
    started,finish=threading.Event(),threading.Event()
    original=analysis.preprocess
    def delayed(*args):
        started.set()
        assert finish.wait(10)
        return original(*args)
    with patch.object(analysis,'preprocess',side_effect=delayed):
        tasks.submit(first,'confirm')
        assert started.wait(10)
        with pytest.raises(ValueError,match='尚未完成'):
            tasks.submit(first,'confirm')
        changed=deepcopy(config)
        changed['area']=200
        tasks.configure(first,changed)
        finish.set()
        tasks.workers.submit(lambda:None).result(timeout=15)
    assert first['prepared'] is None
    assert not first['busy']
    assert not second['pending']


def test_pending_upload_does_not_invalidate_confirmed_results(config):
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        tasks.configure(state,config)
        state['prepared']={'stats':[]}
        state['pending']['new']={'name':'new.csv'}
        assert not state['stale_data']
        changed={**config,'area':300}
        tasks.configure(state,changed)
        assert state['stale_data']


def test_preanalysis_report_survives_module_roundtrip(files,config):
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        tasks.configure(state,config)
        state.update(prepared=analysis.preprocess(files,config),confirmed_config=config,
                     reports={'pre-report':'已生成的洪水预分析报告'})
        show=next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if 'module-view.children' in k)
        show(state['revision'],'data',files[0][0],None,'all','NSE',None)
        show(state['revision'],'simulation',None,None,'all','NSE',None)
        view,_,_=show(state['revision'],'data',files[0][0],None,'all','NSE',None)
        assert '已生成的洪水预分析报告' in str(view)
        assert state['reports']['pre-report']=='已生成的洪水预分析报告'


def test_reports_invalidate_only_when_their_dependencies_change(files,config):
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        tasks.configure(state,config)
        state.update(prepared=analysis.preprocess(files,config),confirmed_config=config,
                     reports={'pre-report':'洪水报告','stage-report':'旧阶段报告','report':'旧综合报告'})
        with patch.object(analysis,'simulate',return_value={'completed':True}):
            tasks.submit(state,'simulate')
            tasks.workers.submit(lambda:None).result(timeout=15)
        assert state['reports']=={'pre-report':'洪水报告'}
        for action in ('bma','xgb'):
            state['reports'].update({'stage-report':'阶段报告','report':'综合报告'})
            with patch.object(analysis,'postprocess',return_value={'completed':True}):
                tasks.submit(state,action)
                tasks.workers.submit(lambda:None).result(timeout=15)
            assert state['reports']=={'pre-report':'洪水报告'}
        state['reports']['report']='综合报告'
        changed={**config,'xgb_depth':config['xgb_depth']+1}
        tasks.configure(state,changed)
        assert state['reports']=={'pre-report':'洪水报告'}
        tasks.configure(state,{**changed,'max_iter':config['max_iter']+1})
        assert state['reports']=={'pre-report':'洪水报告'}
        tasks.configure(state,{**changed,'area':config['area']+1})
        assert state['reports']=={}


@pytest.mark.parametrize('module',['data','simulation','post','report'])
def test_dash_routes_and_callback_navigation_do_not_start_jobs(module):
    app=create_app()
    client=app.server.test_client()
    assert client.get('/').status_code==200
    assert client.get('/workspace?module='+module).status_code==200
    assert client.get('/_dash-layout').status_code==200
    key=next(k for k in app.callback_map if 'landing.style' in k)
    outputs=[{'id':id_,'property':prop} for id_,prop in [('landing','style'),('workspace','style'),('module-tabs','value'),('url','search')]]
    with patch.object(tasks,'submit',side_effect=AssertionError('navigation submitted work')):
        response=client.post('/_dash-update-component',json={'output':key,'outputs':outputs,
            'inputs':[{'id':'url','property':'pathname','value':'/workspace'},
                      {'id':'url','property':'search','value':'?module='+module},
                      {'id':'module-tabs','property':'value','value':'data'}],
            'state':[],'changedPropIds':['url.search']})
    assert response.status_code==200
    assert response.json['response']['module-tabs']['value']==module


def test_result_polling_only_publishes_changed_results(files,config):
    from dash import no_update
    app=create_app()
    config['mode']=analysis.UPLOAD_MODES[2]
    prepared=analysis.preprocess([(f'event{i}.csv',files[0][1]) for i in range(48)],config)
    with app.server.test_request_context('/'):
        state=tasks.current()
        publish=app.callback_map['result-revision.data']['callback'].__wrapped__
        show=next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if 'module-view.children' in k)
        choose=next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if 'chart-choice.options' in k)
        assert publish(0,None)==0
        assert publish(1,0) is no_update
        state.update(prepared=prepared,confirmed_config=config,revision=1)
        assert publish(2,0)==1
        output,key,_=show(1,'data','event0.csv',None,'all','NSE',None)
        from plotly.io.json import to_json_plotly
        import json
        serialized=json.dumps(json.loads(to_json_plotly(output)),ensure_ascii=False)
        assert 'event47.csv' in serialized and '洪水语义画像' in serialized
        assert choose(1,'data','event0.csv')[1] is no_update
        # Status polling during/after slow result delivery never republishes it.
        state.update(busy=True,phase='运行中',progress=.5)
        assert publish(3,1) is no_update
        state.update(busy=False,phase='完成',progress=1)
        assert publish(4,1) is no_update
        assert show(1,'data','event0.csv',None,'all','NSE',key)==(no_update,no_update,no_update)
    for entry in app.callback_map.values():
        if any(out.component_id in ('module-view','chart-choice') for out in (entry['output'] if isinstance(entry['output'],list) else [entry['output']])):
            assert {'id':'poll','property':'n_intervals'} not in entry['inputs']


def test_form_selects_active_algorithm_and_rejects_bad_numbers():
    values={}
    def visit(component):
        if not hasattr(component, 'to_plotly_json'):
            return
        props=component.to_plotly_json()['props']
        if isinstance(props.get('id'),str) and 'value' in props:
            values[props['id']]=props['value']
        children=props.get('children',[])
        for child in children if isinstance(children,list) else [children]:
            visit(child)
    visit(layout())
    for algorithm,fields in ALGORITHMS.items():
        values['algorithm']=algorithm
        parsed=parse_config([values[id_] for id_ in CONFIG_IDS])
        assert parsed['algo_params']=={key:default for key,_,default,*rest in fields}
    values['max_iter']=2.5
    with pytest.raises(ValueError,match='max_iter'):
        parse_config([values[id_] for id_ in CONFIG_IDS])
    values.update(max_iter=5,timestep='hourly',models=['HBV Model'])
    assert parse_config([values[id_] for id_ in CONFIG_IDS])['models']==['HBV Model']


def test_boundary_only_map_and_unsafe_zip():
    from test_dash_features import _basin_files
    from src.app.basin_overview import build_basin_overview
    from zipfile import ZipFile
    boundary,_=_basin_files()
    assert build_basin_overview(boundary)['area_km2']>0
    unsafe=BytesIO()
    with ZipFile(unsafe,'w') as archive:
        archive.writestr('../outside.shp',b'not a shapefile')
    with pytest.raises(ValueError,match='不安全'):
        build_basin_overview(unsafe.getvalue())


def test_bad_spatial_zip_does_not_discard_hydro_analysis(files,config):
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        tasks.configure(state,config)
        state['pending']={'hydro':dict(name=files[0][0],content=files[0][1],kind='hydro'),
                          'space':dict(name='broken.zip',content=b'not a zip',kind='boundary')}
        tasks.submit(state,'confirm')
        tasks.workers.submit(lambda:None).result(timeout=15)
    assert state['prepared']['stats'][0]['记录数']==32
    assert state['space_error']
    assert state['confirmed_files']==['hydro','space']


def test_spatial_only_confirmation_with_integer_dem(config):
    from test_dash_features import _basin_files
    from src.dash_ui.presentation import render, charts_for
    boundary,dem=_basin_files('int16')
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        tasks.configure(state,config)
        state['pending']={'boundary':dict(name='boundary.zip',content=boundary,kind='boundary'),
                          'dem':dict(name='dem.tif',content=dem,kind='dem')}
        with patch.object(analysis,'preprocess',side_effect=AssertionError('no hydrology provided')),patch.object(
                analysis,'call_minimax',side_effect=AssertionError('no LLM requested')):
            tasks.submit(state,'confirm')
            tasks.workers.submit(lambda:None).result(timeout=15)
    assert state['prepared'] is None
    assert state['basin']['min_m']==22
    assert state['basin']['max_m']==77
    assert not state['space_error']
    assert charts_for(state,'data')[0]['title']=='流域概况'
    assert all('name' in trace for trace in charts_for(state,'data')[0]['figure']['data'])
    assert '流域概况' in str(render(state,'data',None,[], 'all','NSE'))
    with pytest.raises(ValueError,match='确认并分析'):
        tasks.submit(state,'simulate')
    tasks.configure(state,{**config,'area':200})
    assert state['stale_data']


def test_dem_only_confirmation_explains_missing_boundary(config):
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        tasks.configure(state,config)
        state['pending']={'dem':dict(name='dem.tif',content=b'dem',kind='dem')}
        tasks.submit(state,'confirm')
        tasks.workers.submit(lambda:None).result(timeout=15)
    assert '补充 SHP ZIP' in state['space_error']
    assert state['prepared'] is None


def test_deepseek_key_is_scoped_to_current_request():
    from src.llm_api import call_minimax, llm_settings, get_llm_config
    with patch('src.llm_api._get_secret',return_value=''),patch('src.llm_api._request_chat',return_value='OK') as request:
        with llm_settings({'provider':'deepseek','api_key':'temporary-key'}):
            assert call_minimax('connection test')=='OK'
        assert request.call_args.args[2]=='temporary-key'
        assert get_llm_config()['api_key']==''


def test_reset_during_task_clears_own_session_without_old_result(files,config):
    app=create_app()
    with app.server.test_request_context('/'):
        other=tasks.current()
        other['reports']['report']='another visitor'
    started,finish=threading.Event(),threading.Event()
    original=analysis.preprocess
    def delayed(*args):
        started.set()
        assert finish.wait(10)
        return original(*args)
    with app.server.test_request_context('/'):
        old=tasks.current()
        tasks.configure(old,config)
        old['llm']={'api_key':'reset-this-key'}
        old['pending']={'file':dict(kind='hydro',name=files[0][0],content=files[0][1])}
        with patch.object(analysis,'preprocess',side_effect=delayed):
            tasks.submit(old,'confirm')
            assert started.wait(10)
            fresh=tasks.reset()
            assert fresh is tasks.current()
            assert old not in list(tasks.sessions.values())
            assert not fresh['pending'] and not fresh['llm'] and fresh['config'] is None
            finish.set()
            tasks.workers.submit(lambda:None).result(timeout=15)
        assert fresh['prepared'] is None and fresh['phase']=='等待确认数据'
        assert not fresh['busy'] and not fresh['reports'] and not fresh['posts']
        assert other['reports']['report']=='another visitor'


def test_offline_exhibition_export_uses_real_results_without_credentials(config):
    from test_dash_features import _analysis_inputs
    from src.dash_ui.snapshot import build_snapshot
    files,_,_= _analysis_inputs()
    config['mode']=analysis.UPLOAD_MODES[2]
    config['models']=ModelRegistry.list_models()
    prepared=analysis.preprocess(files,config)
    simulation=analysis.simulate(prepared,config,imported_parameters())
    posts={kind:analysis.postprocess(prepared,simulation,config,kind) for kind in ('bma','xgb')}
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        state.update(prepared=prepared,simulation=simulation,posts=posts,confirmed_config=config,
                     llm={'api_key':'NEVER-EXPORT'},pending={'raw':b'PRIVATE-UPLOAD'},
                     reports={'report':'# 展会诊断\n\n<script>alert(1)</script>\n\n| 项目 | 结果 |\n|---|---|\n|模型|真实|'},
                     stale_simulation=True)
        exported=build_snapshot(state)
    assert 'NEVER-EXPORT' not in exported and 'PRIVATE-UPLOAD' not in exported
    from test_dash_snapshot import payload
    decoded=str(payload(exported))
    assert '<h1>展会诊断</h1>' in decoded and '<td>真实</td>' in decoded
    assert '<script>alert(1)</script>' not in exported
    assert '配置已变化' in decoded and '尚未生成' in decoded
    assert 'Plotly.newPlot' in exported and '<script src=' not in exported
    assert 'BMA集成' in exported and posts['xgb']['metrics']
    assert all(row['模型'] in exported for row in posts['xgb']['metrics'])
    assert all(name in exported for name in simulation['parameters'])

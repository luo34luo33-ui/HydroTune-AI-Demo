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
    assert '流域概况' in str(render(state,'data',None,[], 'all',None,None,'NSE'))
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

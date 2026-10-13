"""Single-process sessions and bounded background execution for the Dash UI."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import RLock, BoundedSemaphore
from time import time
from uuid import uuid4
from zipfile import BadZipFile

from flask import session

from src.app import dash_analysis as analysis
from src.app.basin_overview import build_basin_overview
from src.llm_api import call_minimax, llm_settings
from .configuration import DATA_KEYS, SIMULATION_KEYS, POST_KEYS

lock = RLock()
sessions = {}
# Existing optimizers use NumPy's process-global RNG; serialize calculations.
workers = ThreadPoolExecutor(max_workers=1)
slots = BoundedSemaphore(4)


def current():
    if 'hydrotune' not in session:
        session['hydrotune'] = uuid4().hex
    sid = session['hydrotune']
    with lock:
        now = time()
        for key in list(sessions):
            state = sessions[key]
            if now - state['touched'] > 12 * 3600 and not state['busy']:
                del sessions[key]
        state = sessions.setdefault(sid, {
            'touched': now, 'pending': {}, 'confirmed_files': [], 'confirmed_file_metadata': [], 'parameters': {}, 'confirmed_parameters': {},
            'config': None, 'config_error': '', 'confirmed_config': None, 'llm': {},
            'prepared': None, 'simulation': None, 'posts': {}, 'reports': {}, 'basin': None, 'basin_display': None, 'semantic_selection': 0,
            'revision': 0, 'input_version': 0, 'job_id': 0, 'busy': False,
            'phase': '等待确认数据', 'progress': None, 'stale_data': False, 'stale_simulation': False,
            'space_error': '', 'llm_status': '使用服务器配置'})
        state['touched'] = now
        return state


def reset():
    """Detach the old session, including any job still finishing in the worker."""
    with lock:
        sessions.pop(session.get('hydrotune'), None)
        session['hydrotune'] = uuid4().hex
        return current()


def configure(state, config):
    with lock:
        previous = state['config']
        state['config_error'] = ''
        if previous == config:
            return
        state['config'] = config
        state['input_version'] += 1
        state['revision'] += 1
        if previous is None:
            return
        if any(previous[k] != config[k] for k in DATA_KEYS):
            state['stale_data'] = state['prepared'] is not None or state['basin'] is not None
            state['stale_simulation'] = state['simulation'] is not None
            state['reports'] = {}
        elif any(previous[k] != config[k] for k in SIMULATION_KEYS):
            state['stale_simulation'] = state['simulation'] is not None
            state['reports'] = {key:text for key,text in state['reports'].items() if key=='pre-report'}
        elif any(previous[k] != config[k] for k in POST_KEYS):
            state['posts'].pop('xgb', None)
            state['reports'] = {key:text for key,text in state['reports'].items() if key=='pre-report'}


def submit(state, action):
    with lock:
        if state['busy']:
            raise ValueError('当前任务尚未完成。')
        if action not in ('test-llm',) and state['config_error']:
            raise ValueError(state['config_error'])
        if action == 'confirm' and not any(f['kind'] in ('hydro','boundary','dem') for f in state['pending'].values()):
            raise ValueError('请先选择水文数据或流域空间文件。')
        if action not in ('confirm','test-llm') and (state['prepared'] is None or state['stale_data']):
            raise ValueError('请先确认并分析当前数据。')
        if action in ('bma','xgb','stage-report','report') and state['stale_simulation']:
            raise ValueError('模拟配置已变化，请重新运行模型。')
        if action in ('bma','xgb','stage-report') and state['simulation'] is None:
            raise ValueError('请先运行水文模型。')
        if not slots.acquire(blocking=False):
            raise ValueError('任务队列已满，请稍后重试。')
        state['busy'] = True
        state['job_id'] += 1
        job_id, version = state['job_id'], state['input_version']
        snapshot = deepcopy({k:v for k,v in state.items() if k != 'touched'})
        state['phase'], state['progress'] = '排队中', None
    workers.submit(run, state, snapshot, action, job_id, version)


def run(state, snapshot, action, job_id, version):
    def progress(phase, fraction=None):
        with lock:
            state['phase'], state['progress'] = phase, fraction
    try:
        progress({'confirm':'解析文件与预分析','simulate':'模型模拟','bma':'BMA 融合',
                  'xgb':'XGB 误差校正','test-llm':'测试 LLM 连接'}.get(action, '生成智能报告'))
        config = snapshot['config']
        if action == 'confirm':
            files = list(snapshot['pending'].values())
            hydro = [(f['name'],f['content']) for f in files if f['kind']=='hydro']
            prepared = analysis.preprocess(hydro, config) if hydro else None
            spatial = {f['kind']:f['content'] for f in files if f['kind']!='hydro'}
            basin, space_error = None, ''
            if 'boundary' in spatial:
                try:
                    basin = build_basin_overview(spatial['boundary'], spatial.get('dem'))
                except (ValueError, OSError, RuntimeError, BadZipFile) as exc:
                    space_error = str(exc)
            elif 'dem' in spatial:
                space_error = '已选择 DEM，请补充 SHP ZIP 流域边界。'
            update = dict(prepared=prepared, simulation=None, posts={}, reports={}, basin=basin, semantic_selection=0,
                          space_error=space_error, stale_data=False, stale_simulation=False,
                          confirmed_config=config, confirmed_files=list(snapshot['pending']),confirmed_parameters=snapshot['parameters'],
                          confirmed_file_metadata=[{'key':key,'name':f['name'],'kind':f['kind'],'size':len(f['content'])}
                                                   for key,f in snapshot['pending'].items()])
        elif action == 'simulate':
            result = analysis.simulate(snapshot['prepared'], config, snapshot['confirmed_parameters'], progress)
            update = dict(simulation=result, posts={},
                          reports={key:text for key,text in snapshot['reports'].items() if key=='pre-report'},
                          stale_simulation=False)
        elif action in ('bma','xgb'):
            result = analysis.postprocess(snapshot['prepared'], snapshot['simulation'], config, action)
            update = dict(posts={**snapshot['posts'], action:result},
                          reports={key:text for key,text in snapshot['reports'].items() if key=='pre-report'})
        elif action == 'test-llm':
            with llm_settings(snapshot['llm']):
                response = call_minimax('请只回复：连接成功。')
            update = dict(llm_status=response if response.startswith('[ERROR]') else '连接成功')
        else:
            kind = {'pre-report':'预分析','stage-report':'阶段','report':'综合'}[action]
            simulation = None if snapshot['stale_simulation'] else snapshot['simulation']
            text = analysis.generate_report(snapshot['prepared'], simulation, snapshot['posts'] if simulation else {}, snapshot['llm'], kind, snapshot['basin'])
            update = dict(reports={**snapshot['reports'], action:text})
        with lock:
            if job_id == state['job_id'] and version == state['input_version']:
                state.update(update)
                state['phase'], state['progress'] = '完成', 1
                state['revision'] += 1
            else:
                state['phase'], state['progress'] = '配置已变化，请重新运行当前步骤', None
    except Exception as exc:
        # Background task boundary: retain completed results and expose a readable failure.
        with lock:
            state['phase'], state['progress'] = f'任务失败：{exc}', None
            state['revision'] += 1
    finally:
        with lock:
            state['busy'] = False
        slots.release()

"""User inputs and navigation; calculations live in services and background tasks."""
import base64
from copy import deepcopy
from urllib.parse import parse_qs, urlencode
from uuid import uuid4
import re
import pandas as pd
from dash import ALL, Input, Output, State, ctx, html, no_update
from . import tasks
from .configuration import ALGORITHMS, CONFIG_IDS, parse_config
from .layouts import MODULES
from .snapshot import build_snapshot
from .presentation import charts_for, filtered_figure, render, semantic_detail

ACTIONS=['confirm','simulate','bma','xgb','pre-report','stage-report','report','test-llm']


def file_list(state):
    return [html.Div([html.Strong(f['name']),html.Small(f"{f['kind']} · {len(f['content'])/1024:.1f} KB · "+('已确认' if key in state['confirmed_files'] else '待确认')),
                      html.Button('移除',id={'type':'remove-file','key':key},className='remove',n_clicks=0)],className='file-item')
            for key,f in state['pending'].items()]


def register(app):
    @app.callback(Output('semantic-detail','children'),Input('semantic-event','value'))
    def select_semantic_event(index):
        with tasks.lock:
            state=tasks.current()
            prepared=state['prepared']
            if prepared is None or not isinstance(index,int) or not 0 <= index < len(prepared['codes']):
                return no_update
            state['semantic_selection']=index
        return semantic_detail(prepared,index)

    @app.callback(Output('reset-token','data'),Input('reset-session','n_clicks'),prevent_initial_call=True)
    def reset_session(_):
        tasks.reset()
        return uuid4().hex

    app.clientside_callback("""
        function(token) {
            if (!token) return window.dash_clientside.no_update;
            Object.keys(sessionStorage).forEach(function(key) {
                if (key.startsWith('_dash_persistence.') || key === 'had-confirmed' || key === 'had-confirmed-timestamp')
                    sessionStorage.removeItem(key);
            });
            window.location.replace('/workspace?module=data');
            return token;
        }
    """,Output('reset-complete','data'),Input('reset-token','data'),prevent_initial_call=True)

    @app.callback(Output('upstream-field','style'),Input('routing','value'))
    def upstream_mapping(routing):
        return {} if 'on' in routing else {'display':'none'}

    @app.callback(Output('download-results','data'),Output('snapshot-message','children'),
                  Input('save-results','n_clicks'),prevent_initial_call=True)
    def save_results(_):
        with tasks.lock:
            state=tasks.current()
            if state['busy'] or not (state['prepared'] or state['basin']):
                return no_update,'请先完成数据分析，再保存运行结果。'
            snapshot=deepcopy({key:state[key] for key in (
                'prepared','simulation','posts','reports','basin','confirmed_config',
                'stale_data','stale_simulation','space_error','semantic_selection')})
        return {'content':build_snapshot(snapshot),'filename':'HydroTune-results.html','type':'text/html'},'已保存完整展示页面，可离线打开。'

    @app.callback(Output('landing','style'),Output('workspace','style'),Output('module-tabs','value'),Output('url','search'),
                  Input('url','pathname'),Input('url','search'),Input('module-tabs','value'))
    def navigate(path,search,tab):
        module=parse_qs((search or '').lstrip('?')).get('module',['data'])[0]
        module=module if module in MODULES else 'data'
        if ctx.triggered_id=='module-tabs':
            module=tab
        workspace=path=='/workspace'
        return ({'display':'none'} if workspace else {},{} if workspace else {'display':'none'},module,
                '?'+urlencode({'module':module}) if workspace else '')

    @app.callback(*[Output(f'algo-form-{i}','style') for i in range(len(ALGORITHMS))],Output('sce-note','style'),
                  Output('parameter-uploads','style'),Input('algorithm','value'),Input('parameter_mode','value'))
    def algorithm_forms(algorithm,mode):
        return [({} if key==algorithm and mode=='calibrate' else {'display':'none'}) for key in ALGORITHMS]+[
            {} if algorithm=='SCE-UA' and mode=='calibrate' else {'display':'none'},{} if mode=='import' else {'display':'none'}]

    @app.callback(Output('config-note','children'),*[Input(id_,'value') for id_ in CONFIG_IDS])
    def configure(*values):
        state=tasks.current()
        try:
            tasks.configure(state,parse_config(values))
            return '配置已保存到当前会话。'
        except (ValueError,TypeError) as exc:
            with tasks.lock:
                state['config_error']=str(exc)
            return str(exc)

    @app.callback(Output('upload-message','children'),
                  Input('hydro-upload','contents'),Input('boundary-upload','contents'),Input('dem-upload','contents'),
                  Input({'type':'parameter-upload','model':ALL},'contents'),Input({'type':'remove-file','key':ALL},'n_clicks'),
                  State('hydro-upload','filename'),State('boundary-upload','filename'),State('dem-upload','filename'),
                  State({'type':'parameter-upload','model':ALL},'filename'),State({'type':'parameter-upload','model':ALL},'id'),prevent_initial_call=True)
    def receive(hydro,boundary,dem,parameters,removes,hydro_names,boundary_name,dem_name,parameter_names,parameter_ids):
        state=tasks.current()
        trigger=ctx.triggered_id
        if isinstance(trigger,dict) and trigger['type']=='remove-file':
            if not any(removes):
                return no_update
            with tasks.lock:
                item=state['pending'].pop(trigger['key'],None)
                if item and item['kind']=='parameter':
                    state['parameters'].pop(item['model'],None)
                state['revision']+=1
            return '已移除待确认文件。已确认结果保持不变。'
        if trigger=='hydro-upload':
            kind,contents,names='hydro',hydro,hydro_names
        elif trigger=='boundary-upload':
            kind,contents,names='boundary',boundary,boundary_name
        elif trigger=='dem-upload':
            kind,contents,names='dem',dem,dem_name
        elif isinstance(trigger,dict):
            index=parameter_ids.index(trigger)
            kind,contents,names='parameter',parameters[index],parameter_names[index]
        else:
            return no_update
        if not contents:
            return no_update
        contents=contents if isinstance(contents,list) else [contents]
        names=names if isinstance(names,list) else [names]
        extensions={'hydro':('.csv','.xlsx','.xls'),'boundary':('.zip',),'dem':('.tif','.tiff'),'parameter':('.csv',)}
        try:
            accepted=[]
            for name,content in zip(names,contents):
                if not name.lower().endswith(extensions[kind]):
                    raise ValueError('文件格式不支持：'+name)
                raw=base64.b64decode(content.split(',',1)[1],validate=True)
                if not raw or len(raw)>100*1024*1024:
                    raise ValueError('每个文件须非空且不超过 100 MB。')
                accepted.append(dict(name=name,content=raw,kind=kind,**({'model':trigger['model']} if kind=='parameter' else {})))
            with tasks.lock:
                if sum(len(f['content']) for f in state['pending'].values())+sum(len(f['content']) for f in accepted)>250*1024*1024:
                    raise ValueError('暂存文件总量不能超过 250 MB。请先移除不需要的文件。')
                for item in accepted:
                    for key,old in list(state['pending'].items()):
                        if old['kind']==kind and (kind in ('boundary','dem') or old['name']==item['name'] or kind=='parameter' and old['model']==item['model']):
                            del state['pending'][key]
                    state['pending'][uuid4().hex]=item
                    if kind=='parameter':
                        state['parameters'][item['model']]=item['content']
                state['revision']+=1
            return '文件已暂存。点击“确认并分析”后生效。'
        except (ValueError,IndexError) as exc:
            return str(exc)

    @app.callback(Output('llm-message','children'),Output('llm-key','value'),Input('save-llm','n_clicks'),
                  State('llm-provider','value'),State('llm-region','value'),State('llm-model','value'),State('llm-key','value'),prevent_initial_call=True)
    def save_llm(_,provider,region,model,key):
        with tasks.lock:
            state=tasks.current()
            state['llm']={'provider':provider,'region':region,'model':model,'api_key':key or ''}
            state['llm_status']='配置已保存；尚未测试连接'
            state['revision']+=1
        return '已保存。留空密钥时使用服务器配置。',''

    @app.callback(Output('task-message','children'),*[Input(action,'n_clicks') for action in ACTIONS],
                  *[State(id_,'value') for id_ in CONFIG_IDS],prevent_initial_call=True)
    def start(*values):
        state=tasks.current()
        try:
            if ctx.triggered_id!='test-llm':
                tasks.configure(state,parse_config(values[len(ACTIONS):]))
            tasks.submit(state,ctx.triggered_id)
            return '任务已提交，可继续浏览其他模块。'
        except (ValueError,TypeError) as exc:
            return str(exc)

    @app.callback(Output('job-status','children'),Output('job-progress','value'),Output('top-status','children'),
                  Output('llm-status','children'),Output('job-progress','style'),Output('session-note','children'),Output('had-confirmed','data'),
                  Output('report-download','disabled'),Output('corrected-download','disabled'),Output('save-results','disabled'),
                  *[Output(action,'disabled') for action in ACTIONS],Input('poll','n_intervals'),State('had-confirmed','data'))
    def poll(_,had_confirmed):
        state=tasks.current()
        with tasks.lock:
            value=state['progress']
            return [state['phase'],round(value*100) if value is not None else None,'运行中' if state['busy'] else '就绪',state['llm_status'],
                    {} if state['busy'] or value is not None else {'display':'none'},
                    '服务端会话已失效（可能因服务重启或长时间未使用），请重新选择并确认数据。' if had_confirmed and state['confirmed_config'] is None else '',
                    bool(had_confirmed or state['confirmed_config'] is not None),
                    not bool(state['reports'].get('report')),not bool(state['posts'].get('xgb',{}).get('corrected_flow')),state['busy'] or not bool(state['prepared'] or state['basin'])]+[state['busy'] for action in ACTIONS]

    @app.callback(Output('file-list','children'),Output('file-list-key','data'),Input('poll','n_intervals'),State('file-list-key','data'))
    def pending_files(_,previous):
        state=tasks.current()
        key=[list(state['pending']),state['confirmed_files']]
        return (file_list(state),key) if key!=previous else (no_update,no_update)

    @app.callback(*[Output('actions-'+module,'style') for module in MODULES],Output('chart-tools','style'),Input('module-tabs','value'))
    def actions(module):
        return [{} if key==module else {'display':'none'} for key in MODULES]+[{'display':'none'} if module=='report' else {}]

    @app.callback(Output('chart-choice','options'),Output('chart-choice','value'),Output('series-choice','options'),
                  Input('poll','n_intervals'),Input('module-tabs','value'),State('chart-choice','value'))
    def choices(_,module,choice):
        state=tasks.current()
        with tasks.lock:
            charts=charts_for(state,module)
            options=[c['title'] for c in charts]
            names=list(dict.fromkeys(trace['name'] for c in charts if c['title']!='模型性能比较' for trace in c['figure']['data']))
        return options,choice if choice in options else (options[0] if options else None),names

    @app.callback(Output('module-view','children'),Output('render-key','data'),Output('report-copy','content'),
                  Input('poll','n_intervals'),Input('module-tabs','value'),Input('chart-choice','value'),Input('series-choice','value'),
                  Input('phase-choice','value'),Input('metric-choice','value'),State('render-key','data'))
    def show(_,module,choice,series,phase,metric,last_key):
        state=tasks.current()
        with tasks.lock:
            key=[state['revision'],module,choice,series,phase,metric]
            if key==last_key:
                return no_update,no_update,no_update
            snapshot=deepcopy(state)
        return render(snapshot,module,choice,series,phase,metric),key,snapshot['reports'].get('report','')

    @app.callback(Output('module-tabs','value',allow_duplicate=True),Input('confirm','n_clicks'),prevent_initial_call=True)
    def after_confirm(_):
        return 'data'

    @app.callback(Output('download-report','data'),Input('report-download','n_clicks'),prevent_initial_call=True)
    def report_download(_):
        text=tasks.current()['reports'].get('report')
        return {'content':text,'filename':'HydroTune-report.md','type':'text/markdown'} if text else no_update

    @app.callback(Output('download-corrected','data'),Input('corrected-download','n_clicks'),prevent_initial_call=True)
    def corrected_download(_):
        result=tasks.current()['posts'].get('xgb')
        if result is None:
            return no_update
        records=[]
        for file,flows in result['corrected_flow'].items():
            for index,(obs,sim,corrected) in enumerate(zip(flows['observed'],flows['simulated'],flows['corrected'])):
                records.append({'文件':file,'记录':index,'实测流量_m3s':obs,'模拟流量_m3s':sim,'校正流量_m3s':corrected})
        return {'content':pd.DataFrame(records).to_csv(index=False),'filename':'HydroTune-XGB.csv','type':'text/csv'}

    @app.callback(Output('download-chart','data'),Output('export-message','children'),Input('chart-download','n_clicks'),
                  State('module-tabs','value'),State('chart-choice','value'),State('series-choice','value'),State('phase-choice','value'),
                  State('chart-format','value'),State('metric-choice','value'),prevent_initial_call=True)
    def chart_download(_,module,choice,series,phase,format_,metric):
        figure=filtered_figure(tasks.current(),module,choice,series,phase,metric)
        if figure is None:
            return no_update,'请先生成并选择一张过程图。'
        name='HydroTune-'+module+'-'+re.sub(r'[^\w.-]','_',choice or 'chart')+'.'+format_
        if format_=='html':
            return {'content':figure.to_html(include_plotlyjs=True),'filename':name,'type':'text/html'},'交互图表已导出。'
        try:
            data=figure.to_image(format=format_,width=1400,height=700,scale=1)
            return {'content':base64.b64encode(data).decode(),'filename':name,'base64':True},'静态图表已导出。'
        except (ValueError,RuntimeError,OSError,ImportError):
            return no_update,'静态导出不可用，请检查 Kaleido 与 Chrome 安装。可以改用 HTML 导出。'

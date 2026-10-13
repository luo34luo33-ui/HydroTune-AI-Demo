"""Self-contained, read-only workbench snapshots of a coherent session state."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import dash
from markdown_it import MarkdownIt
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder
from dash import html
from .configuration import CONFIG_IDS, FIELDS, ALGORITHMS
from .layouts import layout, MODULES, group
from .presentation import charts_for, render, semantic_detail

ASSETS=Path(__file__).parent/'assets'
FORM_IDS=CONFIG_IDS+['llm-provider','llm-region','llm-model']


def scientific_config(config):
    return {k:v for k,v in (config or {}).items() if k in set(FIELDS)|{'mapping','routing_parameters','algo_params'}}


def capture_state(state):
    """Called under the session lock; exclude upload bytes and credentials."""
    result=deepcopy({k:state[k] for k in ('prepared','simulation','posts','reports','basin',
                    'stale_data','stale_simulation','space_error','semantic_selection')})
    result.update({k:state.get(k) for k in ('phase','progress','busy','config_error')})
    result['confirmed_config']=scientific_config(state.get('confirmed_config'))
    result['config']=scientific_config(state.get('config'))
    result['llm_public']={k:state.get('llm',{}).get(k) for k in ('provider','region','model')}
    result['file_metadata']=[dict(name=f['name'],kind=f['kind'],size=len(f['content']),
                             status='已确认' if key in state['confirmed_files'] else '待确认')
                            for key,f in state.get('pending',{}).items() if isinstance(f,dict)]
    result['file_metadata'] += [dict(name=f['name'],kind=f['kind'],size=f['size'],status='结果来源（已从暂存区移除）')
                               for f in state.get('confirmed_file_metadata',[]) if f['key'] not in state.get('pending',{})]
    existing={f['name'] for f in result['file_metadata']}
    if state['prepared']:
        result['file_metadata'] += [dict(name=e['name'],kind='hydro',size=None,status='结果来源')
                                   for e in state['prepared']['events'] if e['name'] not in existing]
    # These values are only used to remove credential echoes; never serialize them.
    result['redactions']=[state.get('llm',{}).get('api_key','')]
    return result


class SafeMarkdown(HTMLParser):
    """Preserve generated Markdown markup without external image loads."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='img':
            self.parts.append(escape(dict(attrs).get('alt','图片未嵌入快照')))
            return
        clean=[]
        for key,value in attrs:
            if key in ('class','title','align') or (tag=='a' and key=='href' and value.startswith(('https://','http://','mailto:','#'))):
                clean.append(f' {key}="{escape(value,quote=True)}"')
        self.parts.append('<'+tag+''.join(clean)+'>')
    def handle_endtag(self,tag):
        if tag!='img': self.parts.append('</'+tag+'>')
    def handle_data(self,data): self.parts.append(data)
    def handle_entityref(self,name): self.parts.append('&'+name+';')
    def handle_charref(self,name): self.parts.append('&#'+name+';')


def markdown_html(text):
    def block_math(state,start,end,silent):
        pos=state.bMarks[start]+state.tShift[start]
        opener=next((o for o in ('$$','\\[') if state.src.startswith(o,pos)),None)
        if opener is None: return False
        closer='\\]' if opener=='\\[' else '$$'
        finish=state.src.find(closer,pos+len(opener))
        if finish<0 or finish>state.eMarks[end-1]: return False
        last=start
        while state.eMarks[last]<finish+len(closer): last+=1
        if state.src[finish+len(closer):state.eMarks[last]].strip(): return False
        if not silent:
            token=state.push('snapshot_math_block','',0)
            token.block=True
            token.content=state.src[pos:finish+len(closer)]
            token.map=[start,last+1]
            state.line=last+1
        return True
    def inline_math(state,silent):
        start=state.pos
        opener=next((o for o in ('$$','\\(','\\[','$') if state.src.startswith(o,start)),None)
        if opener is None: return False
        closer={'\\(':'\\)','\\[':'\\]'}.get(opener,opener)
        end=state.src.find(closer,start+len(opener))
        if end<0: return False
        if not silent:
            token=state.push('snapshot_math','',0)
            token.content=state.src[start:end+len(closer)]
        state.pos=end+len(closer)
        return True
    engine=MarkdownIt('commonmark',{'html':False}).enable('table')
    engine.block.ruler.before('paragraph','snapshot_math_block',block_math)
    engine.inline.ruler.before('escape','snapshot_math',inline_math)
    engine.renderer.rules['snapshot_math']=lambda tokens,index,*args: '<span class="snapshot-math">'+escape(tokens[index].content)+'</span>'
    engine.renderer.rules['snapshot_math_block']=lambda tokens,index,*args: '<div class="snapshot-math">'+escape(tokens[index].content)+'</div>'
    parser=SafeMarkdown()
    parser.feed(engine.render(text))
    return ''.join(parser.parts)


def component_tree(component):
    if component is None or isinstance(component,(str,int,float,bool)): return component
    if isinstance(component,(list,tuple)): return [component_tree(child) for child in component]
    node=component.to_plotly_json()
    props=dict(node['props'])
    if node['type']=='Markdown':
        return {'type':'SnapshotMarkdown','props':{'className':'markdown','html':markdown_html(props.get('children') or
                '尚未生成智能报告。使用上方按钮根据当前计算证据生成。')}}
    for key in ('children','label'):
        if key in props: props[key]=component_tree(props[key])
    if 'options' in props:
        props['options']=[{**o,'label':component_tree(o['label'])} if isinstance(o,dict) else o for o in props['options']]
    if isinstance(props.get('id'),dict): props.pop('id')
    return {'type':node['type'],'props':props}


def walk(node):
    if isinstance(node,list):
        for item in node: yield from walk(item)
    elif isinstance(node,dict) and 'props' in node:
        yield node
        yield from walk(node['props'].get('children'))


def configure_shell(shell,state,ui):
    nodes={n['props']['id']:n for n in walk(shell) if isinstance(n['props'].get('id'),str)}
    values={id_:nodes[id_]['props'].get('value') for id_ in FORM_IDS}
    config=state.get('config') or state.get('confirmed_config') or {}
    for key in FIELDS:
        if key in values and key in config:
            v=config[key]
            values[key]=(['on'] if v else []) if key in ('routing','routing_fixed','xgb_clip') else v
    values.update({'col-'+k:v for k,v in config.get('mapping',{}).items()})
    values['routing-k']=config.get('routing_parameters',{}).get('k_routing',2.5)
    values['routing-x']=config.get('routing_parameters',{}).get('x_routing',.25)
    if config.get('algorithm') in ALGORITHMS:
        i=list(ALGORITHMS).index(config['algorithm'])
        values.update({f'algo-{i}-{key}':value for key,value in config.get('algo_params',{}).items()})
    values.update({'llm-'+k:v for k,v in state.get('llm_public',{}).items() if v is not None})
    values.update({k:v for k,v in ui.get('forms',{}).items() if k in FORM_IDS})
    for key,value in values.items(): nodes[key]['props']['value']=value
    nodes['llm-key']['props'].update(value='',placeholder='密钥不随快照保存',disabled=True)
    for i,key in enumerate(ALGORITHMS):
        nodes[f'algo-form-{i}']['props']['style']={} if key==values['algorithm'] and values['parameter_mode']=='calibrate' else {'display':'none'}
    nodes['sce-note']['props']['style']={} if values['algorithm']=='SCE-UA' and values['parameter_mode']=='calibrate' else {'display':'none'}
    nodes['parameter-uploads']['props']['style']={} if values['parameter_mode']=='import' else {'display':'none'}
    nodes['upstream-field']['props']['style']={} if values['routing'] else {'display':'none'}
    files=[]
    for item in state.get('file_metadata',[]):
        size=f"{item['size']/1024:.1f} KB" if item['size'] is not None else '大小未保留'
        files.append(html.Div([html.Strong(item['name']),html.Small(f"{item['kind']} · {size} · {item['status']}"),
                               html.Button('移除',disabled=True,className='remove')],className='file-item'))
    nodes['file-list']['props']['children']=component_tree(files)
    nodes['job-status']['props']['children']=state.get('phase') or '等待确认数据'
    nodes['top-status']['props']['children']='运行中' if state.get('busy') else '就绪'
    progress=state.get('progress')
    nodes['job-progress']['props'].update(value=round(progress*100) if progress is not None else None,
                                          style={} if state.get('busy') or progress is not None else {'display':'none'})
    for key,value in ui.get('messages',{}).items():
        if key in ('task-message','session-note','upload-message','llm-message','llm-status','config-note','export-message'):
            nodes[key]['props']['children']=str(value or '')
    nodes['snapshot-message']['props']['children']='离线展示：配置只读，计算、上传及重置操作不可用。'
    nodes['sidebar']['props']['children'].append(component_tree(group('结果实际使用的配置',[
        html.P('当前侧栏显示保存时的表单值；下方为结果实际使用的配置。',className='hint'),
        html.H3('已确认输入'),html.Pre(json.dumps(scientific_config(state.get('confirmed_config')),ensure_ascii=False,indent=2)),
        html.H3('模型模拟'),html.Pre(json.dumps(scientific_config(state['simulation']['configuration']) if state['simulation'] else '尚未运行',ensure_ascii=False,indent=2))])))
    return values


def mathjax_script():
    # Dash ships the complete TeX/SVG distribution; reuse it without a CDN dependency.
    source=(Path(dash.__file__).parent/'dcc'/'async-mathjax.js').read_text(encoding='utf-8')
    source=re.sub(r'var console=__webpack_require__\(\d+\),process=__webpack_require__\(\d+\);',
                  'var console=window.console,process={env:{},browser:true,nextTick:fn=>queueMicrotask(fn)};',source)
    return r'''window.MathJax={loader:{load:[]},startup:{typeset:false},tex:{packages:['base','ams','newcommand','noundefined','configmacros'],inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']]},options:{enableMenu:false}};
(function(){const previous=self.webpackChunkdash_core_components;const chunks=[];self.webpackChunkdash_core_components=chunks;
'''+source+'''
const entry=Object.values(chunks[0][1]).sort((a,b)=>b.toString().length-a.toString().length)[0];
entry({exports:{}},{});self.webpackChunkdash_core_components=previous;})();'''


def build_snapshot(state,ui_state=None):
    state=deepcopy(state)
    ui={k:v for k,v in (ui_state or {}).items() if k in ('module','choice','series','phase','metric','format','forms','details','graphs','messages','scroll','viewport')}
    for key,value in dict(module='data',series=[],phase='all',metric='NSE',format='html').items(): ui.setdefault(key,value)
    ui['forms']={k:v for k,v in ui.get('forms',{}).items() if k in FORM_IDS}
    tree=component_tree(layout())
    shell=[n for n in tree['props']['children'] if isinstance(n,dict) and (n['type']=='Header' or n['props'].get('id')=='workspace')]
    for node in shell:
        if node['props'].get('id')=='workspace': node['props']['style']={}
    forms=configure_shell(shell,state,ui)
    views,charts={},{}
    for module in MODULES:
        charts[module]=charts_for(state,module)
        choice=charts[module][0]['title'] if charts[module] else None
        views[module]=component_tree(render(state,module,choice,[], 'all','NSE'))
    portraits=[component_tree(semantic_detail(state['prepared'],i)) for i in range(len(state['prepared']['codes']))] if state['prepared'] else []
    metrics=(state['simulation']['metrics']+[r for p in state['posts'].values() for r in p['metrics']]) if state['simulation'] else []
    payload=dict(shell=shell,views=views,charts=charts,portraits=portraits,metrics=metrics,
                 calib_names=sorted(state['simulation']['calib_names']) if state['simulation'] else [],
                 ui=ui,forms=forms,created=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                 busy=bool(state.get('busy')),reports=state['reports'],semantic_selection=state.get('semantic_selection',0),
                 runtime={key:state.get(key) for key in ('phase','progress','busy','config_error')},
                 corrected=state['posts'].get('xgb',{}).get('corrected_flow',{}),files=state.get('file_metadata',[]))
    data=json.dumps(payload,ensure_ascii=False,cls=PlotlyJSONEncoder)
    for secret in state.get('redactions',[])+[state.get('llm',{}).get('api_key','')]:
        if secret: data=data.replace(secret,'[密钥已隐藏]')
    data=data.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    style=(ASSETS/'theme.css').read_text(encoding='utf-8')+'\n'+(ASSETS/'snapshot.css').read_text(encoding='utf-8')
    runtime=(ASSETS/'snapshot_runtime.js').read_text(encoding='utf-8')
    scripts=get_plotlyjs()+'\n'+(mathjax_script() if any('$' in text or '\\(' in text or '\\[' in text for text in state['reports'].values()) else '')
    scripts=scripts.replace('</script','<\\/script')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' blob:; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>HydroTune · 离线工作台快照</title><style>'''+style+'''</style><script>'''+scripts+'''</script></head><body><div id="snapshot-root"></div>
<script id="snapshot-data" type="application/json">'''+data+'''</script><script>'''+runtime+'''</script></body></html>'''

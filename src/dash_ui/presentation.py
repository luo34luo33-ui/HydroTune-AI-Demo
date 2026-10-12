"""Render server-side scientific results with a consistent Plotly theme."""
from copy import deepcopy
from dash import dcc, html
import pandas as pd
import plotly.graph_objects as go
from src.app.dash_analysis import combine_charts, summary
from .semantics import NOTE, code_rows, options, portrait


def card(title, *children):
    return html.Section([html.Div([html.H2(title)],className='card-heading'),*children],className='card')


def folding_card(title, *children):
    return html.Details([html.Summary(title),html.Div(children,className='folding-body')],className='card folding-card',open=False)


def empty(title, text):
    return html.Div([html.Span('HYDROTUNE',className='eyebrow'),html.H2(title),html.P(text)],className='empty-state card')


def table(rows):
    if not rows:
        return html.P('暂无可用结果。',className='hint')
    columns = list(rows[0])
    def value(v):
        if isinstance(v,float):
            return f'{v:.4f}' if pd.notna(v) and abs(v) != float('inf') else '—'
        return str(v)
    return html.Div(html.Table([html.Thead(html.Tr([html.Th(c) for c in columns])),
                               html.Tbody([html.Tr([html.Td(value(row[c])) for c in columns]) for row in rows])]),className='table-scroll')


def markdown(text):
    return dcc.Markdown(text or '尚未生成智能报告。使用上方按钮根据当前计算证据生成。',mathjax=True,className='markdown',dangerously_allow_html=False)


def charts_for(state, module):
    if module == 'data':
        charts = list(state['prepared']['charts']) if state['prepared'] else []
        if state['basin']:
            charts.append({'title':'流域概况','figure':state['basin']['figure']})
        return charts
    if not state['simulation']:
        return []
    charts=combine_charts(state['simulation'],state['posts'] if module=='post' else {})
    if module=='post':
        charts.append({'title':'模型性能比较','figure':performance_figure(state,[], 'all','NSE').to_plotly_json()})
    return charts


def performance_figure(state, series, phase, metric):
    rows=state['simulation']['metrics']+[r for post in state['posts'].values() for r in post['metrics']]
    if phase!='all':
        rows=[r for r in rows if r['类型']==phase]
    if series:
        rows=[r for r in rows if r['模型'] in series]
    totals=summary(rows)
    figure=go.Figure()
    for category in sorted({r['类型'] for r in totals}):
        values=[r for r in totals if r['类型']==category]
        figure.add_bar(x=[r['模型'] for r in values],y=[r[metric] for r in values],name=category)
    figure.update_layout(template='plotly_white',barmode='group',yaxis_title=metric)
    return figure


def filtered_figure(state, module, choice, series, phase, metric='NSE'):
    if module=='post' and choice=='模型性能比较' and state['simulation']:
        return performance_figure(state,series,phase,metric)
    charts = charts_for(state,module)
    chart = next((c for c in charts if c['title']==choice),charts[0] if charts else None)
    if chart is None:
        return None
    if phase != 'all' and state['simulation'] and module != 'data':
        is_calib = chart['title'] in state['simulation']['calib_names']
        if (phase=='率定') != is_calib:
            return None
    fig = go.Figure(deepcopy(chart['figure']))
    if series:
        fig.data = [trace for trace in fig.data if trace.name in series]
    fig.update_layout(uirevision=choice, paper_bgcolor='white',plot_bgcolor='white',
                      font={'family':'Arial, Microsoft YaHei','color':'#172B4D'},
                      legend={'orientation':'h','y':-.25,'x':0,'font':{'size':10},'entrywidth':.33,'entrywidthmode':'fraction'},
                      title={'text':chart['title'],'x':.02,'y':.98,'font':{'size':13}},
                      margin={'l':50,'r':50,'t':45,'b':110},height=460)
    return fig


def basin_card(state):
    if state['basin']:
        basin = state['basin']
        labels = {'area_km2':'流域面积 (km²)','min_m':'最低高程 (m)',
                  'max_m':'最高高程 (m)','mean_m':'平均高程 (m)','crs':'坐标系'}
        return card('流域概况',table([{labels.get(k,k):v for k,v in basin.items() if k!='figure'}]),
                    dcc.Graph(figure=basin['figure'],config={'responsive':True,'displaylogo':False}))
    return empty('流域概况',state['space_error'] or '尚未确认流域空间数据。可上传 SHP ZIP，DEM 为可选项。')


def semantic_detail(prepared, index):
    view = portrait(prepared, index)
    return [html.P(f"来源：{view['source']} · {view['period']}",className='hint'),
            html.Div([html.Span(tag,className='semantic-tag') for tag in view['tags']],className='semantic-tags'),
            html.Div([kpi(label,value,unit) for label,value,unit in view['metrics']],className='kpi-grid'),
            html.P(view['sentence'],className='notice'),
            dcc.Graph(figure=view['figure'],config={'responsive':True,'displaylogo':False}),
            html.P('浅青区域：起点至主峰；浅蓝区域：主峰至终点。多峰场次以主峰划分，仅作过程定位。',className='hint')]


def semantic_card(state):
    prepared = state['prepared']
    if not prepared['codes']:
        return card('洪水语义画像',html.P('当前没有可展示的洪水编码。',className='hint'))
    index = state.get('semantic_selection',0)
    return card('洪水语义画像',html.P('选择一场洪水，查看已有编码与对应过程。',className='hint'),
                dcc.Dropdown(id='semantic-event',options=options(prepared),value=index,clearable=False),
                html.Div(semantic_detail(prepared,index),id='semantic-detail'),
                html.P(NOTE,className='hint'),
                html.Details([html.Summary('完整语义编码表'),table(code_rows(prepared)),
                              html.P('代表性评分为现有规则评分，不表示概率或模型置信度。',className='hint')]))


def render(state,module,choice,series,phase,metric):
    prepared,simulation = state['prepared'],state['simulation']
    if prepared is None:
        if module == 'data' and (state['basin'] or state['space_error']):
            return [html.Div('数据配置已变化，请重新确认并分析。' if state['stale_data'] else
                             '当前仅展示空间数据；水文统计与模型模拟需另行上传水文时序。',className='notice'),basin_card(state)]
        return empty('从一份数据开始','在左侧选择水文或流域空间文件，然后点击“确认并分析”。')
    notices = []
    if state['stale_data']:
        notices.append(html.Div('数据配置已变化。下方为此前结果，请重新确认并分析。',className='notice warning'))
    elif state['stale_simulation'] and module!='data':
        notices.append(html.Div('模拟配置已变化。下方为此前结果，请重新运行模型。',className='notice warning'))
    if module=='data':
        count = sum(len(e['flow']) for e in prepared['events'])
        missing = sum(e['missing']['flow'] for e in prepared['events'])
        content = [html.Div([kpi('水文文件',len(prepared['events']),'个'),kpi('数据记录',count,'条'),
                             kpi('原始流量有效率',f'{100*(count-missing)/count:.1f}','%'),kpi('识别洪水',len(prepared['codes']),'场')],className='kpi-grid'),
                   card('基础数据概况',table(prepared['stats']),*[html.P(w,className='hint') for w in prepared['warnings']]),
                   card('洪水统计',table(prepared['semantic_events'])),
                   semantic_card(state),
                   card('智能预分析报告',markdown(state['reports'].get('pre-report')))]
        content.append(basin_card(state))
    elif module=='simulation':
        if simulation is None:
            return notices+[empty('准备运行水文模型','确认左侧模型、算法与参数，然后点击“启动模型运行”。')]
        rows = simulation['metrics']
        content = [card('率定与验证效果',table(summary(rows))),folding_card('模型参数',table([
            {'模型':name,'参数':key,'值':value} for name,params in simulation['parameters'].items() for key,value in params.items()])),
                   folding_card('逐场次评价指标',table(rows)),card('阶段分析报告',markdown(state['reports'].get('stage-report'))),
                   html.P('仅展示优化器真实反馈；当前未记录完整收敛轨迹，不绘制推算的收敛曲线。',className='hint'),
                   *[html.Div(w,className='notice warning') for w in simulation['warnings']]]
    elif module=='post':
        if simulation is None:
            return notices+[empty('先完成模型模拟','后处理使用已经完成的真实模拟结果。')]
        content = [html.Div('BMA 按本场观测 NSE 计算权重；多场 XGB 使用真实误差滞后，属于回顾性评估。',className='notice')]
        for kind,title in [('bma','BMA 模型融合'),('xgb','XGB 误差校正')]:
            post=state['posts'].get(kind)
            result_card=folding_card if kind=='bma' else card
            if not post:
                content.append(result_card(title,html.P('尚未运行。使用上方按钮启动。',className='hint')))
            else:
                content.append(result_card(title,table(post['weights']) if kind=='bma' else html.P('校正前指标见下方模型比较。',className='hint'),
                                    table(summary(post['metrics'])),*[html.Div(w,className='notice warning') for w in post['warnings']]))
        rows=simulation['metrics']+[r for post in state['posts'].values() for r in post['metrics']]
        if phase!='all':
            rows=[r for r in rows if r['类型']==phase]
        if series:
            rows=[r for r in rows if r['模型'] in series]
        totals=summary(rows)
        chart=[] if choice=='模型性能比较' else [dcc.Graph(figure=performance_figure(state,series,phase,metric),config={'responsive':True,'displaylogo':False})]
        content.append(card('模型性能比较',html.P('指标按完整场次计算。',className='hint'),table(totals),*chart))
    else:
        return notices+[card('综合智能分析报告',markdown(state['reports'].get('report')))]
    figure=filtered_figure(state,module,choice,series,phase,metric)
    if figure is not None:
        content.insert(1,card('模型性能比较' if choice=='模型性能比较' else '水文过程',dcc.Graph(id='selected-chart',figure=figure,config={'responsive':True,'displaylogo':False})))
    elif module!='report':
        content.append(html.P('当前筛选条件下没有过程图。',className='hint'))
    return notices+content


def kpi(label,value,unit):
    return html.Div([html.Span(label),html.Div([html.Strong(str(value)),html.Small(unit)])],className='kpi')

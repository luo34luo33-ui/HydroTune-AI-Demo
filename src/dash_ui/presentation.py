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
    stage_colors={'率定':'#092B4C','验证':'#08A6AD'}
    unit=' · m³/s' if metric in ('RMSE','MAE') else ' · %' if metric=='PBIAS' else ''
    for category in sorted({r['类型'] for r in totals}):
        values=[r for r in totals if r['类型']==category]
        raw_values=[r[metric] for r in values]
        figure.add_bar(x=[r['模型'] for r in values],y=[abs(value) if pd.notna(value) else value for value in raw_values] if metric=='NSE' else raw_values,
                       customdata=raw_values,name=category,
                       marker=dict(color=stage_colors.get(category,'#7893AA'),cornerradius=5,line_width=0),
                       texttemplate='%{y:.3f}',textposition='outside',cliponaxis=False,
                       textfont=dict(size=12,color='#334155'),
                       hovertemplate='<b>%{x}</b><br>'+category+' · 平均 '+metric+': %{customdata:.3f}'+unit+'<extra></extra>')
    model_names=list(dict.fromkeys(r['模型'] for r in totals))
    figure.update_layout(template='plotly_white',barmode='group',bargap=.36,bargroupgap=.14,
                         height=420,paper_bgcolor='white',plot_bgcolor='white',
                         title=dict(text='<b>|NSE|</b> · 场次平均 NSE 的绝对值' if metric=='NSE' else f'<b>{metric}</b> · 场次平均表现',x=.02,font=dict(size=16,color='#092B4C')),
                         font=dict(family='Arial, Microsoft YaHei',size=12,color='#475569'),
                         margin=dict(l=65,r=30,t=65,b=85),
                         xaxis=dict(title=None,tickmode='array',tickvals=model_names,
                                    ticktext=[name.replace(' Model','').replace('-XGB',' + XGB') for name in model_names],
                                    tickangle=0,showgrid=False,showline=True,linecolor='#E2E8F0',automargin=True),
                         yaxis=dict(title='|NSE|' if metric=='NSE' else '平均 '+metric+unit,gridcolor='#EDF2F7',gridwidth=1,
                                    zeroline=True,zerolinecolor='#CBD5E1',zerolinewidth=1,
                                    rangemode='tozero',tickformat='.2f' if metric in ('NSE','KGE') else '.3~g',automargin=True),
                         legend=dict(orientation='h',x=.5,xanchor='center',y=-.18,
                                     bgcolor='rgba(0,0,0,0)',font=dict(size=12),tracegroupgap=16),
                         hoverlabel=dict(bgcolor='white',bordercolor='#DCE5EE',font_size=12))
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
    basin, image = state['basin'], state.get('basin_display')
    config=state.get('confirmed_config') or state.get('config') or {}
    prepared=state['prepared']
    facts=[]
    if basin:
        labels={'area_km2':'边界计算面积','min_m':'最低高程','max_m':'最高高程','mean_m':'平均高程','crs':'坐标系'}
        for key,label in labels.items():
            if key in basin:
                value=basin[key]
                unit=' km²' if key=='area_km2' else ' m' if key.endswith('_m') else ''
                facts.append((label,f'{value:,.2f}'+unit if isinstance(value,(int,float)) else str(value)))
    if 'area' in config:
        facts.append(('建模流域面积',f"{config['area']:,.2f} km²"))
    if prepared:
        facts.extend([('时间尺度','小时尺度' if prepared['timestep']=='hourly' else '日尺度'),
                      ('洪水场次',f"{len(prepared['codes'])} 场"),
                      ('预热期',f"{prepared.get('warmup_hours',0)} 小时")])
    if image:
        visual=html.Figure([html.Img(src=image['src'],alt='流域概况补充展示图'),
                            html.Figcaption('补充展示图 · '+image['name'])],className='basin-visual')
    elif basin:
        figure=go.Figure(basin['figure'])
        figure.update_layout(autosize=True,height=None,margin=dict(l=45,r=35,t=35,b=45))
        visual=html.Div(dcc.Graph(id='basin-chart',figure=figure,style={'height':'100%','width':'100%'},
                       config={'responsive':True,'displaylogo':False}),className='basin-visual')
    else:
        visual=html.Div([html.Span('BASIN OVERVIEW',className='eyebrow'),html.H3('流域空间概况'),
                        html.P(state['space_error'] or '上传流域边界和 DEM 数据，自动分析流域概况。')],className='basin-visual basin-placeholder')
    return card('流域概况',html.Div([visual,html.Aside([html.Span('流域基本情况',className='eyebrow'),
                html.Dl([html.Div([html.Dt(label),html.Dd(value)],className='basin-fact') for label,value in facts]),
                html.P('确认空间或水文数据后展示基本情况。',className='hint') if not facts else None],className='basin-facts')],className='basin-overview'))


def semantic_detail(prepared, index):
    view = portrait(prepared, index)
    return [html.P(f"来源：{view['source']} · {view['period']}",className='hint'),
            html.Div([html.Span(tag,className='semantic-tag') for tag in view['tags']],className='semantic-tags'),
            html.Div([kpi(label,value,unit) for label,value,unit in view['metrics']],className='kpi-grid'),
            html.P(view['sentence'],className='notice'),
            dcc.Graph(id='semantic-chart',figure=view['figure'],config={'responsive':True,'displaylogo':False})]


def semantic_card(state):
    prepared = state['prepared']
    if not prepared['codes']:
        return card('洪水语义画像',html.P('当前没有可展示的洪水编码。',className='hint'))
    index = state.get('semantic_selection',0)
    return card('洪水语义画像',html.P('选择一场洪水，查看已有编码与对应过程。',className='hint'),
                dcc.Dropdown(id='semantic-event',options=options(prepared),value=index,clearable=False),
                html.Div(semantic_detail(prepared,index),id='semantic-detail'),
                html.Details([html.Summary('完整语义编码表'),table(code_rows(prepared)),
                              html.P(NOTE,className='hint'),
                              html.P('代表性评分为现有规则评分，不表示概率或模型置信度。',className='hint')]))


def render(state,module,choice,series,phase,metric):
    prepared,simulation = state['prepared'],state['simulation']
    if prepared is None:
        if module == 'data':
            return [basin_card(state),html.Div('数据配置已变化，请重新确认并分析。' if state['stale_data'] else
                             '上传水文时序并确认后，可查看水文统计与洪水语义画像。',className='notice')]
        return empty('从一份数据开始','在左侧选择水文或流域空间文件，然后点击“确认并分析”。')
    notices = []
    if state['stale_data']:
        notices.append(html.Div('数据配置已变化。下方为此前结果，请重新确认并分析。',className='notice warning'))
    elif state['stale_simulation'] and module!='data':
        notices.append(html.Div('模拟配置已变化。下方为此前结果，请重新运行模型。',className='notice warning'))
    if module=='data':
        count = sum(len(e['flow']) for e in prepared['events'])
        missing = sum(e['missing']['flow'] for e in prepared['events'])
        content = [basin_card(state),html.Div([kpi('水文文件',len(prepared['events']),'个'),kpi('数据记录',count,'条'),
                             kpi('原始流量有效率',f'{100*(count-missing)/count:.1f}','%'),kpi('识别洪水',len(prepared['codes']),'场')],className='kpi-grid'),
                   folding_card('基础数据概况',table(prepared['stats']),*[html.P(w,className='hint') for w in prepared['warnings']]),
                   folding_card('洪水统计',table(prepared['semantic_events'])),
                   semantic_card(state),
                   card('智能预分析报告',markdown(state['reports'].get('pre-report')))]
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
        content = [html.Div('BMA 按各模型所有有效场次（含率定与验证）的平均 NSE 计算统一权重，应用于全部场次，属于回顾性融合评价；多场 XGB 使用真实误差滞后，属于回顾性评估。',className='notice')]
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
        chart=[] if choice=='模型性能比较' else [dcc.Graph(id='performance-chart',figure=performance_figure(state,series,phase,metric),config={'responsive':True,'displaylogo':False})]
        content.append(card('模型性能比较',html.P('指标按完整场次计算。',className='hint'),table(totals),*chart))
    else:
        return notices+[card('综合智能分析报告',markdown(state['reports'].get('report')))]
    figure=filtered_figure(state,module,choice,series,phase,metric)
    if figure is not None:
        content.insert(2 if module=='data' else 1,card('模型性能比较' if choice=='模型性能比较' else '水文过程',dcc.Graph(id='selected-chart',figure=figure,config={'responsive':True,'displaylogo':False})))
    elif module!='report':
        content.append(html.P('当前筛选条件下没有过程图。',className='hint'))
    return notices+content


def kpi(label,value,unit):
    return html.Div([html.Span(label),html.Div([html.Strong(str(value)),html.Small(unit)])],className='kpi')

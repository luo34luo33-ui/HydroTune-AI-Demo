"""Persistent forms, landing page and modular workbench shell."""
from dash import dcc, html
from src.app.dash_analysis import UPLOAD_MODES
from src.models.registry import ModelRegistry
from .configuration import ALGORITHMS

TITLE = 'HydroTune：大语言模型智能体水文平台'
TAGLINE = '自动化水文建模 | LLM驱动的智能诊断 | 建模效果优化'
MODULES = {'data':'数据预分析', 'simulation':'水文模型模拟', 'post':'后处理结果', 'report':'智能分析报告'}


def field(label, component):
    return html.Label([html.Span(label), component], className='field')


def number(id_, value, low, high, step=1):
    return dcc.Input(id=id_, type='number', value=value, min=low, max=high, step=step,
                     persistence=True, persistence_type='session', debounce=True)


def dropdown(id_, options, value, **kwargs):
    return dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                        persistence=True, persistence_type='session', **kwargs)


def check(id_, label, value=False):
    return dcc.Checklist(id=id_, options=[{'label':label,'value':'on'}], value=['on'] if value else [],
                         persistence=True, persistence_type='session')


def upload(id_, text, multiple=False):
    return dcc.Upload(id=id_, children=html.Div([html.Strong(text), html.Small('点击选择或拖入文件')]),
                      multiple=multiple, className='upload-box')


def group(title, children, opened=False):
    return html.Details([html.Summary(title), html.Div(children, className='group-body')], open=opened)


def sidebar():
    models = ModelRegistry.get_all_models()
    return html.Aside([
        html.Div([html.Span('WORKSPACE', className='eyebrow'), html.H2('平台配置')], className='sidebar-title'),
        group('LLM 配置', [
            field('服务商', dropdown('llm-provider', [{'label':'自动（服务器配置）','value':'auto'}, {'label':'MiniMax','value':'minimax'},{'label':'DeepSeek','value':'deepseek'}], 'auto')),
            field('MiniMax 平台', dropdown('llm-region', [{'label':'中国','value':'cn'},{'label':'国际','value':'intl'}], 'cn')),
            field('MiniMax 模型', dcc.Input(id='llm-model', value='MiniMax-M2.7', debounce=True)),
            field('API Key', dcc.Input(id='llm-key', type='password', placeholder='留空使用服务器配置', autoComplete='off')),
            html.P('DeepSeek 使用服务器已配置模型；输入密钥仅在当前会话生效。', className='hint'),
            html.Div([html.Button('保存配置', id='save-llm'), html.Button('测试连接', id='test-llm', className='secondary')], className='button-row'),
            html.Div(id='llm-message', role='status'), html.Div(id='llm-status', className='hint', role='status'),
        ]),
        group('当前注册模型', [
            dcc.Checklist(id='models',options=[{'label':html.Div([html.Strong(name),html.Small(
                (model.__doc__ or model.model_type).strip().splitlines()[0].split(' - ')[0][:70]),
                html.Small('支持小时与日尺度' if model.supports_hourly else '仅支持日尺度')],className='model-option'),'value':name} for name,model in models.items()],
                value=list(models),persistence='hbv',persistence_type='session'),
        ],True),
        group('运行与数据配置', [
            field('上传模式', dropdown('mode', list(UPLOAD_MODES), UPLOAD_MODES[0])),
            field('流域面积 · km²', number('area',150.7944,.1,100000,.0001)),
            field('时间尺度', dropdown('timestep',[{'label':'日尺度','value':'daily'},{'label':'小时尺度','value':'hourly'}],'daily')),
            field('预热期 · 小时', number('warmup_hours',0,0,720)),
            field('参数来源', dropdown('parameter_mode',[{'label':'自动率定','value':'calibrate'},{'label':'导入参数','value':'import'}],'calibrate')),
            group('数据列映射', [field(label, dcc.Input(id='col-'+key,value=key,debounce=True,persistence=True,persistence_type='session')) for key,label in
                               [('date','时间列'),('precip','降水列 · mm/步'),('evap','蒸发列 · mm/步'),('flow','流量列 · m³/s'),('upstream','上游出库流量列 · m³/s')]]),
            group('上游出库汇流', [check('routing','启用上游汇流'),check('routing_fixed','固定马斯京根参数'),
                                  field('k · 计算时间步',number('routing-k',2.5,.5,10,.1)),field('x · 坦化系数',number('routing-x',.25,0,.5,.01))]),
        ], True),
        group('上传与确认', [
            upload('hydro-upload','水文数据 · CSV / Excel',True),
            upload('boundary-upload','流域边界 · SHP ZIP'),
            upload('dem-upload','地形高程 · GeoTIFF（可选）'),
            html.Div(id='upload-message',role='status'), html.Div(id='file-list'),
            html.Div([html.P('导入参数模式：选择后点击确认，参数随输入一起生效。',className='hint'),
                      *[html.Div([html.Small(name),upload({'type':'parameter-upload','model':name},'参数 CSV')]) for name in models]],id='parameter-uploads'),
            html.Button('确认并分析',id='confirm',className='primary full'),
            html.P('仅执行数据预分析；模型率定和智能报告单独启动。',className='hint'),
        ],True),
        group('率定算法', [
            field('优化方法',dropdown('algorithm',list(ALGORITHMS),'两阶段算法(推荐)')),
            field('最大迭代配置',number('max_iter',10,5,200)),
            html.P('两阶段算法按模型维度分配退火与局部优化次数。',className='hint'),
            *[html.Div([field(label,number(f'algo-{i}-{name}',default,low,high,step)) for name,label,default,low,high,step in fields],
                       id=f'algo-form-{i}') for i,fields in enumerate(ALGORITHMS.values())],
            html.P('SCE-UA 固定 5 个复形，每复形点数为参数维度的 2 倍。',id='sce-note',className='hint'),
        ]),
        group('XGB 校正配置', [field(label,number(key,value,low,high,step)) for key,label,value,low,high,step in [
            ('xgb_depth','树深度',10,3,15,1),('xgb_rate','学习率',.05,.01,.5,.01),
            ('xgb_estimators','树数量',300,50,500,50),('xgb_seed','随机种子',2025,1,9999,1),
            ('xgb_quantile','残差中心分位',.9,.8,.98,.02),('xgb_ratio','最大校正比例',.45,.1,1,.05)]]+[check('xgb_clip','限制预测误差范围',True)]),
        html.Div(id='config-note',className='hint',role='status'),
    ],className='sidebar',id='sidebar')


def layout():
    descriptions = ['洪水统计、智能预分析、流域概况。','模型率定、模拟评价、多模型比较。','BMA 融合、XGB 误差校正、多模型对比。','模型性能分析、LLM 智能诊断及优化建议。']
    return html.Div([
        dcc.Location(id='url',refresh=False),dcc.Interval(id='poll',interval=1000),
        dcc.Store(id='render-key'),dcc.Store(id='file-list-key'),dcc.Store(id='had-confirmed',storage_type='session'),
        dcc.Download(id='download-report'),dcc.Download(id='download-chart'),dcc.Download(id='download-corrected'),
        html.Header([dcc.Link([html.Strong('HydroTune'),html.Span('：大语言模型智能体水文平台')],href='/',className='brand'),
                     html.Div(TAGLINE,className='tagline'),
                     html.Div([dcc.Link('进入工作台 →',href='/workspace?module=data',className='workbench-link'),dcc.Link('返回首页',href='/',className='home-link'),html.Span(id='top-status',className='status-chip')],className='header-right')],className='topbar'),
        html.Main([
            html.Section([
                html.Div([html.Span('HYDROLOGY · AI · SCIENTIFIC WORKSPACE',className='eyebrow'),
                          html.H1('HydroTune'),html.H2('让水文建模更智能、更高效'),
                          html.P('集成水文数据预分析、模型率定与模拟、多模型后处理及 LLM 智能诊断，为水文研究提供连贯、高效的建模工作台。',className='hero-subtitle'),
                          dcc.Link('开始使用 →',href='/workspace?module=data',className='primary hero-button'),
                          html.Span('从数据到科学洞察',className='hero-caption')],className='hero-copy'),
                html.Figure([html.Img(src='/assets/landing-cover.svg',alt='数字流域示意：立体地形、等高线与河网',className='hero-cover'),
                             html.Figcaption('数字流域概念示意 · 非实际流域数据')],className='hero-visual')
            ],className='landing-hero'),
            html.Div([html.Div([html.Span('科研工作台',className='eyebrow'),html.H2('连接数据、模型与科学洞察')]),html.P('四个工作模块，一条清晰的研究路径。',className='hint')],className='section-heading'),
            html.Div([dcc.Link([html.Span(className='feature-icon icon-'+key,**{'aria-hidden':'true'}),html.H3(label),html.P(descriptions[i]),html.Span('进入模块 ↗',className='feature-link')],
                              href='/workspace?module='+key,className='feature-card') for i,(key,label) in enumerate(MODULES.items())],className='feature-grid'),
            html.Footer('HydroTune · 以真实计算支持水文科研决策',className='landing-footer')
        ],id='landing',className='landing'),
        html.Div([
            html.Details([html.Summary('平台配置',className='drawer-toggle'),sidebar()],id='configuration-drawer',open=True),
            html.Main([
                html.Div([html.Div([html.Span('RESEARCH WORKBENCH',className='eyebrow'),html.H1('水文科研工作台')]),html.Span('数据 → 模拟 → 后处理 → 诊断',className='hint')],className='workspace-heading'),
                dcc.Tabs(id='module-tabs',value='data',children=[dcc.Tab(label=label,value=key,className='module-tab',selected_className='module-tab selected') for key,label in MODULES.items()],className='module-tabs'),
                html.Div([html.Div(id='job-status',role='status'),html.Progress(id='job-progress',max=100),html.Div(id='task-message',role='status'),html.Div(id='session-note',role='status')],className='task-strip'),
                html.Div([
                    html.Div([html.Button('生成预分析报告',id='pre-report')],id='actions-data'),
                    html.Div([html.Button('启动模型运行',id='simulate',className='primary'),html.Button('生成阶段报告',id='stage-report',className='secondary')],id='actions-simulation'),
                    html.Div([html.Button('运行 BMA',id='bma'),html.Button('运行 XGB',id='xgb'),html.Button('下载校正 CSV',id='corrected-download',className='secondary')],id='actions-post'),
                    html.Div([html.Button('生成 / 重新生成综合报告',id='report',className='primary'),html.Button('下载 Markdown',id='report-download',className='secondary'),dcc.Clipboard(id='report-copy',title='复制 Markdown'),html.Span('复制 Markdown',className='hint')],id='actions-report'),
                ],className='module-actions'),
                html.Details([html.Summary('图表筛选与导出'),html.Div([
                    field('图表 / 文件',dcc.Dropdown(id='chart-choice',clearable=False)),
                    field('显示序列',dcc.Dropdown(id='series-choice',multi=True,placeholder='全部序列')),
                    field('阶段',dropdown('phase-choice',[{'label':'全部','value':'all'},{'label':'率定','value':'率定'},{'label':'验证','value':'验证'}],'all')),
                    field('评价指标',dropdown('metric-choice',['NSE','KGE','RMSE','MAE','PBIAS'],'NSE')),
                    field('时间范围',dcc.DatePickerRange(id='date-range',display_format='YYYY-MM-DD',clearable=True)),
                    field('格式',dropdown('chart-format',['html','png','svg'],'html')),
                    html.Button('下载当前图表',id='chart-download',className='secondary'),html.Div(id='export-message',role='status')
                ],className='filter-grid')],id='chart-tools',className='card chart-tools'),
                dcc.Loading(html.Div(id='module-view'),type='circle',delay_show=250),
                html.Footer('HydroTune · 真实数据 · 可追溯计算 · 智能分析',className='workspace-footer'),
            ],className='workspace-main')
        ],id='workspace',className='workspace',style={'display':'none'})
    ])

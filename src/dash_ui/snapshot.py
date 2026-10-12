"""Read-only, offline exhibition pages from completed workbench results."""
import json
from datetime import datetime
from html import escape

from markdown_it import MarkdownIt
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

from src.app.dash_analysis import summary
from .presentation import charts_for


def build_snapshot(state):
    parts, scripts = [], []
    markdown = MarkdownIt('commonmark', {'html': False}).enable('table')

    def table(rows):
        if not rows:
            return '<p class="hint">暂无结果。</p>'
        columns = list(rows[0])
        def cell(value):
            return escape(f'{value:.4f}' if isinstance(value, float) else str(value))
        return '<div class="table-scroll"><table><thead><tr>' + ''.join(
            '<th>'+escape(key)+'</th>' for key in columns) + '</tr></thead><tbody>' + ''.join(
            '<tr>'+''.join('<td>'+cell(row[key])+'</td>' for key in columns)+'</tr>' for row in rows)+'</tbody></table></div>'

    def section(title, body, anchor=''):
        parts.append(f'<section id="{anchor}"><h2>{escape(title)}</h2>{body}</section>')

    def warnings(items):
        return ''.join('<p class="notice">'+escape(item)+'</p>' for item in items)

    def charts(module):
        body = ''
        for chart in charts_for(state,module):
            id_ = 'plot-'+str(len(scripts))
            payload = json.dumps(chart['figure'],cls=PlotlyJSONEncoder,ensure_ascii=False).replace('<','\\u003c')
            scripts.append(f'{{const f={payload}; Plotly.newPlot("{id_}",f.data,{{...f.layout,width:undefined,autosize:true}},{{responsive:true,displaylogo:false}});}}')
            body += f'<h3>{escape(chart["title"])}</h3><div class="chart" id="{id_}"></div>'
        return body

    if state['stale_data'] or state['stale_simulation']:
        section('结果状态','<p class="notice">配置已变化，以下保留此前运行结果，尚未按新配置重新运行。</p>')
    prepared, simulation = state['prepared'],state['simulation']
    body = '<p class="hint">未运行水文数据预分析。</p>'
    if prepared:
        body = table(prepared['stats'])+'<h3>洪水统计</h3>'+table(prepared['semantic_events'])+warnings(prepared['warnings'])
    basin = state['basin']
    if basin:
        labels = {'area_km2':'流域面积 (km²)','min_m':'最低高程 (m)','max_m':'最高高程 (m)','mean_m':'平均高程 (m)','crs':'坐标系'}
        body += '<h3>流域概况</h3>'+table([{labels[key]:value for key,value in basin.items() if key!='figure'}])
    body += warnings([state['space_error']] if state['space_error'] else [])+charts('data')
    section('数据预分析',body,'data')
    body = '<p class="hint">模型模拟尚未运行。</p>'
    if simulation:
        body = '<h3>率定与验证效果</h3>'+table(summary(simulation['metrics']))+'<details><summary>逐场次评价指标</summary>'+table(simulation['metrics'])+'</details>'
        body += '<details><summary>模型参数</summary>'+table([{'模型':name,'参数':key,'值':value} for name,params in simulation['parameters'].items() for key,value in params.items()])+'</details>'
        body += warnings(simulation['warnings'])+charts('simulation')
    section('水文模型模拟',body,'simulation')
    body = '<p class="notice">BMA 按本场观测 NSE 计算权重；多场 XGB 使用真实误差滞后，属于回顾性评估。</p>'
    for kind,label in [('bma','BMA 模型融合'),('xgb','XGB 误差校正')]:
        result = state['posts'].get(kind)
        body += '<details><summary>'+label+'</summary>' if kind=='bma' else '<h3>'+label+'</h3>'
        if result:
            body += (table(result['weights']) if kind=='bma' else '')+table(result['metrics'])+warnings(result['warnings'])
        else:
            body += '<p class="hint">尚未运行。</p>'
        if kind=='bma':
            body += '</details>'
    if state['posts']:
        body += charts('post')
    section('后处理结果',body,'post')
    body = ''
    for key,label in [('pre-report','预分析报告'),('stage-report','阶段报告'),('report','综合智能分析报告')]:
        body += '<h3>'+label+'</h3>'+(markdown.render(state['reports'][key]) if state['reports'].get(key) else '<p class="hint">尚未生成。</p>')
    section('智能分析报告',body,'report')
    # Export only confirmed scientific configuration, never session credentials or uploads.
    config = simulation['configuration'] if simulation else state['confirmed_config']
    section('运行配置','<details><summary>查看本次结果对应的配置</summary><pre>'+escape(json.dumps(config,ensure_ascii=False,indent=2))+'</pre></details>')
    created = datetime.now().astimezone().isoformat(timespec='seconds')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HydroTune · 运行结果展示</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#F4F7FB;color:#172B4D;font:15px/1.8 Arial,"Microsoft YaHei",sans-serif}
header{background:linear-gradient(110deg,#092B4C,#08A6AD);color:white;padding:36px max(24px,calc((100vw - 1280px)/2))}header h1{margin:0;font-size:32px}header p{margin:8px 0}
nav{display:flex;gap:22px;flex-wrap:wrap}nav a{color:white}main{max-width:1328px;padding:24px;margin:auto}section{background:white;border:1px solid #e2eaf0;border-radius:12px;padding:24px;margin:0 0 20px;box-shadow:0 3px 12px #092b4c08;scroll-margin-top:20px}h2{color:#092B4C;border-bottom:1px solid #eaf0f5;padding-bottom:12px}h3{font-size:17px}summary{cursor:pointer;font-weight:600}details{margin:16px 0}.hint{color:#64748b}.notice{padding:12px 16px;background:#eaf7f7;border-left:3px solid #08A6AD;border-radius:6px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px 12px;border-bottom:1px solid #edf1f5;text-align:left;white-space:nowrap}th{background:#f4f8fb}.chart{width:100%;min-height:460px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;padding:16px}img{max-width:100%}footer{padding:16px;text-align:center;color:#64748b}@media(max-width:600px){main{padding:12px}section{padding:16px}.chart{min-height:360px}}
</style><script>'''+get_plotlyjs()+'''</script></head><body><header><h1>HydroTune · 运行结果展示</h1>
<p>大语言模型智能体水文平台 · 离线交互展示</p><p>保存时间：'''+escape(created)+'''</p><nav>
<a href="#data">数据预分析</a><a href="#simulation">水文模型模拟</a><a href="#post">后处理结果</a><a href="#report">智能分析报告</a>
</nav></header><main>'''+''.join(parts)+'''</main><footer>HydroTune · 本页保留保存时的计算结果；未运行步骤已明确标注。</footer><script>'''+''.join(scripts)+'''</script></body></html>'''

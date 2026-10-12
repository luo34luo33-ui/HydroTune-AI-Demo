"""Editable, self-contained HTML from one completed analysis result."""

import json
from html import escape

from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder


def build_snapshot(result: dict) -> str:
    """Embed actual result data; edits affect presentation, never the originals."""
    serializable = json.loads(json.dumps({'original': result, 'edits': {}},
                                         cls=PlotlyJSONEncoder, ensure_ascii=False),
                              parse_constant=lambda _: None)
    payload = json.dumps(serializable, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    title = escape(f"HydroTune 运行快照 · {result['created_at']}")
    return f'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif;background:#f2f7fa;color:#17324d;margin:0}}
header{{background:#102a43;color:white;padding:24px 5vw}} main{{max-width:1200px;margin:auto;padding:28px}}
section{{background:white;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 3px 20px #1232}}
input,textarea{{font:inherit;border:1px solid #cbd5e1;border-radius:7px;padding:7px;box-sizing:border-box}}
textarea{{width:100%;min-height:130px}} input{{max-width:100%}} button{{font:inherit;background:#087e8b;color:white;border:0;border-radius:7px;padding:9px 15px;cursor:pointer}}
table{{border-collapse:collapse;width:100%;overflow:auto}} th,td{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left}}
.hint{{color:#52677b}} .chart{{min-height:380px}} .compact{{width:105px}}
</style><script>{get_plotlyjs()}</script></head><body>
<header><h1 id="heading"></h1><div id="time"></div></header><main>
<p id="description" class="hint"></p>
<section><h2>快照编辑</h2><p class="hint">仅修改展示内容；原始计算结果保留在下方。离线编辑后点击“下载修改后的 HTML”。</p>
<label>标题 <input id="edit-title"></label><p><label>说明<br><textarea id="edit-description"></textarea></label></p>
<button id="save">下载修改后的 HTML</button></section>
<section><h2>本次配置</h2><pre id="configuration"></pre></section>
<section><h2>预分析</h2><div id="quality"></div><textarea id="edit-preanalysis"></textarea></section>
<section><h2>指标汇总</h2><pre id="summary"></pre></section>
<section><h2>指标展示值</h2><div style="overflow:auto"><table id="metrics"></table></div></section>
<section><h2>模拟过程图</h2><div id="charts"></div></section>
<section><h2>智能报告</h2><textarea id="edit-report" style="min-height:300px"></textarea></section>
<section><details><summary>查看原始计算结果</summary><pre id="original" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></section>
</main><script id="snapshot-data" type="application/json">{payload}</script>
<script>
const state=JSON.parse(document.getElementById('snapshot-data').textContent);
const source=state.original, edits=state.edits;
const $=id=>document.getElementById(id);
const title=edits.title ?? 'HydroTune 运行快照';
$('heading').textContent=title; $('time').textContent=source.created_at;
$('edit-title').value=title; $('edit-description').value=edits.description ?? '';
$('description').textContent=edits.description ?? '';
$('edit-preanalysis').value=edits.preanalysis ?? source.preanalysis_report;
$('edit-report').value=edits.report ?? source.report;
$('configuration').textContent=JSON.stringify(source.configuration,null,2);
$('quality').textContent=Object.entries(source.quality).map(([k,v])=>`${{k}}: ${{v}}`).join('  ·  ');
$('summary').textContent=JSON.stringify(source.summary ?? [],null,2);
$('original').textContent=JSON.stringify(source,null,2);
const fields=['文件','类型','模型','NSE','KGE','RMSE','MAE','PBIAS'];
const table=$('metrics');
const head=document.createElement('tr'); fields.forEach(k=>{{let th=document.createElement('th');th.textContent=k;head.appendChild(th)}});table.appendChild(head);
source.metrics.forEach((row,i)=>{{
  const tr=document.createElement('tr');
  fields.forEach(k=>{{let td=document.createElement('td');
    if(['文件','类型','模型'].includes(k))td.textContent=row[k] ?? '';
    else {{let input=document.createElement('input');input.className='compact';input.dataset.metric=i;input.dataset.field=k;
      input.value=edits.metrics?.[i]?.[k] ?? (row[k] == null ? '' : Number(row[k]).toFixed(4));td.appendChild(input)}}
    tr.appendChild(td)}});table.appendChild(tr)
}});
source.charts.forEach((chart,i)=>{{
  const wrap=document.createElement('div');const label=document.createElement('label');label.textContent='图表标题 ';
  const input=document.createElement('input');input.dataset.chart=i;input.value=edits.charts?.[i] ?? chart.title;
  label.appendChild(input);wrap.appendChild(label);
  const traces=chart.figure.data.map((trace,j)=>{{
    const name=edits.chartTraces?.[i]?.[j] ?? trace.name ?? `序列 ${{j+1}}`;
    const traceLabel=document.createElement('label');traceLabel.textContent=' 图例 ';
    const traceInput=document.createElement('input');traceInput.dataset.chartTrace=`${{i}}:${{j}}`;traceInput.value=name;
    traceLabel.appendChild(traceInput);wrap.appendChild(traceLabel);
    return {{...trace,name}};
  }});
  const plot=document.createElement('div');plot.className='chart';wrap.appendChild(plot);$('charts').appendChild(wrap);
  Plotly.newPlot(plot,traces,{{...chart.figure.layout,title:input.value}},{{responsive:true}});
  input.addEventListener('input',()=>Plotly.relayout(plot,{{title:input.value}}));
  wrap.querySelectorAll('[data-chart-trace]').forEach(traceInput=>{{
    traceInput.addEventListener('input',()=>Plotly.restyle(plot,{{name:traceInput.value}},[Number(traceInput.dataset.chartTrace.split(':')[1])]));
  }});
}});
$('edit-title').addEventListener('input',()=>{{$('heading').textContent=$('edit-title').value}});
$('edit-description').addEventListener('input',()=>{{$('description').textContent=$('edit-description').value}});
$('save').addEventListener('click',()=>{{
  state.edits={{title:$('edit-title').value,description:$('edit-description').value,
    preanalysis:$('edit-preanalysis').value,report:$('edit-report').value,metrics:{{}},charts:{{}},chartTraces:{{}}}};
  document.querySelectorAll('[data-metric]').forEach(input=>{{
    (state.edits.metrics[input.dataset.metric] ??={{}})[input.dataset.field]=input.value;
  }});
  document.querySelectorAll('[data-chart]').forEach(input=>{{state.edits.charts[input.dataset.chart]=input.value}});
  document.querySelectorAll('[data-chart-trace]').forEach(input=>{{
    const [chart,trace]=input.dataset.chartTrace.split(':');
    (state.edits.chartTraces[chart] ??={{}})[trace]=input.value;
  }});
  $('snapshot-data').textContent=JSON.stringify(state).replaceAll('</','<'+String.fromCharCode(92)+'/');
  const blob=new Blob(['<!doctype html>\\n'+document.documentElement.outerHTML],{{type:'text/html;charset=utf-8'}});
  const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='hydrotune-snapshot-edited.html';link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);
}});
</script></body></html>'''

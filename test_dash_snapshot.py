"""Behavior tests for coherent, private, offline workbench snapshots."""
import json
from pathlib import Path
import re
import subprocess
import shutil
from unittest.mock import patch

import pytest
from dash import no_update
from src.dash_ui.application import create_app
from src.dash_ui import tasks
from src.dash_ui.snapshot import build_snapshot, capture_state, walk
from test_dash_workbench import config, files
from src.app import dash_analysis as analysis


def payload(document):
    return json.loads(re.search(r'<script id="snapshot-data" type="application/json">(.*?)</script>',document,re.S).group(1))


@pytest.mark.parametrize('phase,busy,progress', [('等待确认数据',False,None),('排队中',True,None),
                          ('模型运行',True,.42),('任务失败：示例错误',False,None),('完成',False,1)])
def test_preserve_real_status_and_safe_configuration(config,phase,busy,progress):
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        state.update(config=config,phase=phase,busy=busy,progress=progress,llm={'api_key':'PRIVATE-KEY'},
                     pending={'one':{'name':'event.csv','kind':'hydro','content':b'PRIVATE-RAW'}},
                     confirmed_file_metadata=[{'key':'removed','name':'removed.csv','kind':'hydro','size':123}])
        ui=dict(module='post',forms={'area':'564','llm-key':'CLIENT-SECRET','algo-1-w':.33},
                details={'sidebar/LLM 配置':True},graphs={'selected-chart':{'layout':{'xaxis':{'range':[2,8]}},'visibility':[{'name':'实测流量','visible':'legendonly'}]}},
                scroll={'y':120,'sidebar':80},messages={'task-message':'真实消息'})
        snapshot=capture_state(state)
        document=build_snapshot(snapshot,ui)
    data=payload(document)
    nodes={node['props'].get('id'):node['props'] for node in walk(data['shell']) if node['props'].get('id')}
    assert data['busy']==busy and nodes['job-status']['children']==phase
    assert nodes['job-progress']['value']==(round(progress*100) if progress is not None else None)
    assert nodes['area']['value']=='564' and nodes['algo-1-w']['value']==.33
    assert data['ui']['scroll']==ui['scroll'] and data['ui']['graphs']==ui['graphs']
    assert data['ui']['details']==ui['details']
    assert {f['name'] for f in data['files']}=={'event.csv','removed.csv'}
    assert data['files'][0]['status']=='待确认'
    assert all(token not in document for token in ('PRIVATE-KEY','CLIENT-SECRET','PRIVATE-RAW'))
    assert "connect-src 'none'" in document and '<script src=' not in document


def test_snapshot_capture_rejects_unrendered_revision_but_allows_busy_state():
    app=create_app()
    with app.server.test_request_context('/'):
        state=tasks.current()
        state.update(revision=3,busy=True,phase='真实进度',progress=.2)
        save=next(v['callback'].__wrapped__ for k,v in app.callback_map.items() if 'download-results.data' in k)
        ui=dict(ready=True,revision=2,render_key=[2,'data',None,None,'all','NSE'],module='data',series=[],phase='all',metric='NSE')
        assert save(ui)[0] is no_update
        ui.update(revision=3,render_key=[3,'data',None,None,'all','NSE'])
        with patch.object(tasks,'submit',side_effect=AssertionError('snapshot started a job')):
            result,message=save(ui)
        assert payload(result['content'])['busy'] is True
        assert state['busy'] and state['progress']==.2
        assert '冻结' in message


def test_snapshot_keeps_report_text_and_blocks_active_markup(files,config):
    app=create_app()
    report='# 真实报告\n\n<script>alert(1)</script>\n\n![外部图](https://example.invalid/image.png)\n\n$$Q=1$$'
    with app.server.test_request_context('/'):
        state=tasks.current()
        state.update(prepared=analysis.preprocess(files,config),reports={'report':report},
                     confirmed_config=config,config={**config,'area':564},stale_data=True)
        document=build_snapshot(capture_state(state),{'forms':{'area':600}})
    data=payload(document)
    assert data['reports']['report']==report
    markdown=[n['props']['html'] for n in walk(data['views']['report']) if n['type']=='SnapshotMarkdown']
    assert '<h1>真实报告</h1>' in markdown[0]
    assert '<img' not in markdown[0] and '<script>' not in markdown[0]
    assert 'window.MathJax=' in document
    assert data['forms']['area']==600
    assert '配置已变化' in str(data['views']['data'])


@pytest.mark.skipif(shutil.which('node') is None,reason='Node.js is only needed for the JavaScript parity test, not the application')
def test_offline_aggregation_matches_python_summary():
    rows=[{'模型':model,'文件':str(i),'类型':category,'NSE':value,'KGE':value,'RMSE':value,'MAE':value,'PBIAS':value}
          for model in ('Tank Model','HBV Model') for category in ('率定','验证') for i,value in enumerate((.4,.8,-999))]
    code="""const fs=require('fs'),vm=require('vm');const sandbox={window:{},document:{getElementById:()=>null}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
const rows=JSON.parse(process.argv[2]);process.stdout.write(JSON.stringify(sandbox.window.HydroSnapshot.aggregate(rows,['Tank Model'],'验证')));"""
    result=subprocess.run(['node','-e',code,str(Path('src/dash_ui/assets/snapshot_runtime.js')),json.dumps(rows)],check=True,capture_output=True,text=True,encoding='utf-8')
    assert json.loads(result.stdout)==analysis.summary([r for r in rows if r['模型']=='Tank Model' and r['类型']=='验证'])


def test_markdown_preserves_tex_delimiters_and_subscripts():
    from src.dash_ui.snapshot import markdown_html
    for expression in (r'\(Q_{obs}=\sum_i Q_i\)',r'\[Q=\frac{V}{t}\]',r'$$Q_{obs}=1$$'):
        assert expression in markdown_html('公式：'+expression)
    expression='$$\nQ_{obs}=1\n\n+2\n$$'
    assert expression in markdown_html(expression)

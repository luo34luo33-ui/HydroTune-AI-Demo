/* Standalone viewer: no Dash renderer, HTTP requests, model calls, or polling. */
(function () {
    'use strict';
    const metricNames=['NSE','KGE','RMSE','MAE','PBIAS'];
    function aggregate(rows,series,phase) {
        const groups=new Map();
        rows.filter(r=>(!series.length||series.includes(r['模型']))&&(phase==='all'||phase===r['类型'])).forEach(r=>{
            const key=r['类型']+'\u0000'+r['模型'];
            if(!groups.has(key))groups.set(key,[]);
            groups.get(key).push(r);
        });
        return [...groups.keys()].sort().map(key=>{
            const values=groups.get(key),out={'类型':values[0]['类型'],'模型':values[0]['模型']};
            metricNames.forEach(name=>{const valid=values.map(r=>r[name]).filter(v=>Number.isFinite(v)&&v> -900);
                out[name]=valid.length?valid.reduce((a,b)=>a+b,0)/valid.length:null;});
            return out;
        });
    }
    function performance(rows,series,phase,metric,baseLayout) {
        const totals=aggregate(rows,series,phase);
        return {data:[...new Set(totals.map(r=>r['类型']))].sort().map(category=>{
            const values=totals.filter(r=>r['类型']===category);
            return {type:'bar',name:category,x:values.map(r=>r['模型']),y:values.map(r=>r[metric])};
        }),layout:{...(baseLayout||{}),barmode:'group',yaxis:{...(baseLayout?.yaxis||{}),title:{text:metric}},paper_bgcolor:'white',plot_bgcolor:'white'}};
    }
    window.HydroSnapshot={aggregate,performance};
    const source=document.getElementById('snapshot-data');
    if(!source)return;
    const saved=JSON.parse(source.textContent),ui=saved.ui;
    const performanceLayout=(saved.charts.post||[]).find(c=>c.title==='模型性能比较')?.figure.layout;
    document.body.classList.add('snapshot-workbench');
    const root=document.getElementById('snapshot-root');
    let active=ui.module,choice=ui.choice,series=ui.series||[],phase=ui.phase||'all',metric=ui.metric||'NSE';
    let semantic=saved.semantic_selection||0,queue=[],initial=true;
    const details={...(ui.details||{})},graphMemory=new Map();
    const labels={data:'数据预分析',simulation:'水文模型模拟',post:'后处理结果',report:'智能分析报告'};
    const downloadActions=new Set(['chart-download','report-download','corrected-download']);
    const clone=value=>JSON.parse(JSON.stringify(value));
    function detailKey(el){if(el.id)return el.id;const title=el.querySelector(':scope > summary');
        return (el.closest('#sidebar')?'sidebar/':'view/')+(title?title.textContent.trim():'');}
    function captureDetails(){root.querySelectorAll('details').forEach(el=>details[detailKey(el)]=el.open);}
    function restoreDetails(){root.querySelectorAll('details').forEach(el=>{
        const key=detailKey(el);if(Object.hasOwn(details,key))el.open=details[key];
        el.addEventListener('toggle',()=>details[key]=el.open);
    });}
    function append(el,children){(Array.isArray(children)?children:[children]).forEach(child=>{const node=make(child);if(node)el.append(node);});}
    function make(node){
        if(node===null||node===undefined||typeof node==='boolean')return null;
        if(Array.isArray(node)){const fragment=document.createDocumentFragment();node.forEach(n=>{const el=make(n);if(el)fragment.append(el);});return fragment;}
        if(typeof node!=='object')return document.createTextNode(String(node));
        const type=node.type,p=node.props||{};
        if(['Store','Download','Interval','Location'].includes(type))return null;
        let el;
        if(type==='Loading')return make(p.children);
        if(type==='SnapshotMarkdown'){el=document.createElement('div');el.innerHTML=p.html;}
        else if(type==='Graph'){
            el=document.createElement('div');el.className='snapshot-chart';queue.push({el,figure:p.figure,id:p.id});
        }else if(type==='Dropdown'){
            el=document.createElement('select');el.className='snapshot-select';el.multiple=!!p.multi;
            el.disabled=!(p.id==='semantic-event'||['chart-choice','series-choice','phase-choice','metric-choice','chart-format'].includes(p.id));
            (p.options||[]).forEach(option=>{const o=typeof option==='object'?option:{label:option,value:option};
                const opt=document.createElement('option');opt.value=String(o.value);opt.textContent=typeof o.label==='string'?o.label:String(o.value);
                opt.selected=p.multi?(p.value||[]).includes(o.value):p.value===o.value;el.append(opt);});
        }else if(type==='Checklist'){
            el=document.createElement('div');
            (p.options||[]).forEach(option=>{const label=document.createElement('label'),check=document.createElement('input');
                check.type='checkbox';check.checked=(p.value||[]).includes(option.value);check.disabled=true;label.append(check);append(label,option.label);el.append(label);});
        }else if(type==='Input'){
            el=document.createElement('input');el.type=p.type||'text';el.value=p.type==='password'?'':p.value??'';el.readOnly=true;
            if(p.type==='password')el.placeholder='密钥不随快照保存';
        }else if(type==='Upload'){
            el=document.createElement('div');el.title='离线快照不能上传文件';append(el,p.children);
        }else if(type==='Tabs'){
            el=document.createElement('div');(p.children||[]).forEach(tab=>{const b=document.createElement('button');
                b.textContent=tab.props.label;b.className='module-tab';b.dataset.module=tab.props.value;
                b.onclick=()=>switchModule(tab.props.value,true);el.append(b);});
        }else if(type==='Clipboard'){
            el=document.createElement('button');el.textContent='⧉';el.title='复制 Markdown';el.setAttribute('aria-label','复制 Markdown');el.onclick=copyReport;
        }else{
            el=document.createElement(type==='Link'?'a':type.toLowerCase());append(el,p.children);
        }
        if(p.id)el.id=p.id;
        if(p.className)el.className+=' '+p.className;
        if(p.style)Object.entries(p.style).forEach(([key,value])=>{
            el.style[key]=typeof value==='number'&&!['opacity','zIndex','flexGrow','flexShrink','fontWeight','order'].includes(key)?value+'px':value;
        });
        if(p.open!==undefined)el.open=p.open;
        if(p.placeholder&&el.type!=='password')el.placeholder=p.placeholder;
        if(p.role)el.setAttribute('role',p.role);
        if(type==='Img'&&p.src&&p.src.startsWith('data:image/')){el.src=p.src;el.alt=p.alt||'';}
        if(type==='Progress'){
            if(p.value!==null&&p.value!==undefined){el.value=p.value;el.max=p.max||100;}
            else {const frozen=document.createElement('div');frozen.id=el.id;frozen.style.cssText=el.style.cssText;frozen.className='snapshot-progress-unknown';
                frozen.setAttribute('role','progressbar');frozen.setAttribute('aria-valuetext','进度未知（保存时刻）');frozen.title='无法获取精确进度；这是保存时刻的状态';el=frozen;}
        }
        if(type==='Button'){
            el.type='button';el.disabled=!downloadActions.has(p.id)||(p.id==='report-download'&&!saved.reports.report)||(p.id==='corrected-download'&&!Object.keys(saved.corrected).length);el.title=el.disabled?'离线快照不执行此操作':'';
            if(downloadActions.has(p.id))el.onclick=()=>handleDownload(p.id);
        }
        if(type==='Link'){
            el.href='#'+active;
            if((p.href||'').includes('workspace'))el.onclick=e=>{e.preventDefault();switchModule('data',true);};
            else {el.setAttribute('aria-disabled','true');el.title='当前为离线工作台快照';el.onclick=e=>e.preventDefault();}
        }
        if(type==='Dropdown'){
            el.addEventListener('change',()=>{
                if(p.id==='semantic-event'){semantic=Number(el.value);showSemantic();return;}
                if(p.id==='chart-choice')choice=el.value;
                if(p.id==='series-choice')series=[...el.selectedOptions].map(o=>o.value);
                if(p.id==='phase-choice')phase=el.value;
                if(p.id==='metric-choice')metric=el.value;
                updateCharts();
            });
        }
        return el;
    }
    function memoryKey(id){return active+'/'+(id==='selected-chart'?choice:id==='semantic-chart'?semantic:'')+'/'+id;}
    function viewState(el){const layout={};Object.keys(el.layout||{}).forEach(key=>{
        if(/^[xy]axis\d*$/.test(key))layout[key]={range:el.layout[key].range,autorange:el.layout[key].autorange};
        else if(/^scene\d*$/.test(key))layout[key]={camera:el.layout[key].camera};
    });return {layout,visibility:(el.data||[]).map(t=>({name:t.name,visible:t.visible}))};}
    function mergeView(figure,state){if(!state)return figure;
        Object.entries(state.layout||{}).forEach(([key,value])=>figure.layout[key]={...figure.layout[key],...value});
        (state.visibility||[]).forEach(item=>figure.data.filter(t=>t.name===item.name).forEach(t=>{if(item.visible!==undefined)t.visible=item.visible;}));return figure;}
    async function plot(item,figure){
        if(!item.el.isConnected||item.el.closest('[hidden]'))return;
        const id=item.id||item.el.id,key=memoryKey(id),f=clone(figure||item.figure);
        const captured=initial&&active===ui.module?(ui.graphs||{})[id]:null;
        mergeView(f,graphMemory.get(key)||captured);
        f.layout={...f.layout,width:undefined,autosize:true};
        const config={responsive:true,displaylogo:false,showLink:false,modeBarButtonsToRemove:['sendDataToCloud','editInChartStudio']};
        if(item.el.data)await Plotly.react(item.el,f.data,f.layout,config);else await Plotly.newPlot(item.el,f.data,f.layout,config);
        if(!item.el.dataset.tracked){
            item.el.on('plotly_relayout',()=>graphMemory.set(memoryKey(id),viewState(item.el)));
            item.el.on('plotly_restyle',()=>graphMemory.set(memoryKey(id),viewState(item.el)));
            item.el.dataset.tracked='1';
        }
    }
    function selectedFigure(){
        const chart=(saved.charts[active]||[]).find(c=>c.title===choice);
        if(!chart)return null;
        if(active!=='data'&&phase!=='all'&&choice!=='模型性能比较'&&((phase==='率定')!==saved.calib_names.includes(choice)))return null;
        const fig=active==='post'&&choice==='模型性能比较'?performance(saved.metrics,series,phase,metric,performanceLayout):clone(chart.figure);
        if(series.length&&choice!=='模型性能比较')fig.data=fig.data.filter(t=>series.includes(t.name));
        fig.layout={...fig.layout,uirevision:choice,paper_bgcolor:'white',plot_bgcolor:'white',
            font:{family:'Arial, Microsoft YaHei',color:'#172B4D'},
            legend:{orientation:'h',y:-.25,x:0,font:{size:10},entrywidth:.33,entrywidthmode:'fraction'},
            title:{text:choice,x:.02,y:.98,font:{size:13}},margin:{l:50,r:50,t:45,b:110},height:460};
        return fig;
    }
    function tableNode(rows){const table=document.createElement('table');if(!rows.length)return document.createTextNode('暂无可用结果。');
        const keys=Object.keys(rows[0]),head=document.createElement('thead'),tr=document.createElement('tr');
        keys.forEach(key=>{const th=document.createElement('th');th.textContent=key;tr.append(th);});head.append(tr);table.append(head);
        const body=document.createElement('tbody');rows.forEach(row=>{const r=document.createElement('tr');keys.forEach(key=>{
            const cell=document.createElement('td'),v=row[key];cell.textContent=v===null?'—':typeof v==='number'?v.toFixed(4):String(v);r.append(cell);});body.append(r);});table.append(body);return table;}
    async function updateCharts(){
        const selected=document.getElementById('selected-chart'),figure=selectedFigure();
        if(selected){selected.closest('section').hidden=!figure;
            if(figure)await plot({el:selected,id:'selected-chart'},figure);
            const title=selected.closest('section').querySelector('h2');if(title)title.textContent=choice==='模型性能比较'?'模型性能比较':'水文过程';}
        const card=[...document.querySelectorAll('#module-view section')].find(el=>el.querySelector('h2')?.textContent==='模型性能比较'&&!el.contains(selected));
        if(active==='post'&&card){
            const existing=card.querySelector('.table-scroll');if(existing)existing.replaceChildren(tableNode(aggregate(saved.metrics,series,phase)));
            const comparison=card.querySelector('#performance-chart');
            if(comparison){comparison.hidden=choice==='模型性能比较';if(!comparison.hidden)await plot({el:comparison,id:'performance-chart'},performance(saved.metrics,series,phase,metric,performanceLayout));}
        }
        const message=document.getElementById('snapshot-filter-empty');if(message)message.remove();
        if(!figure&&active!=='report'&&(saved.charts[active]||[]).length){const p=document.createElement('p');p.id='snapshot-filter-empty';p.className='hint';p.textContent='当前筛选条件下没有过程图。';document.getElementById('module-view').append(p);}
    }
    async function showSemantic(){
        const container=document.getElementById('semantic-detail');if(!container||!saved.portraits.length)return;
        const previous=container.querySelector('.js-plotly-plot');if(previous)Plotly.purge(previous);
        const before=queue.length;container.replaceChildren(make(saved.portraits[semantic]));
        await Promise.all(queue.splice(before).map(item=>plot(item)));
    }
    function setOptions(id,options,value,multiple=false){const el=document.getElementById(id);el.replaceChildren();
        options.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;o.selected=multiple?value.includes(v):value===v;el.append(o);});}
    function controls(){const charts=saved.charts[active]||[],titles=charts.map(c=>c.title);
        if(!titles.includes(choice))choice=titles[0]||null;
        setOptions('chart-choice',titles,choice);
        const names=[...new Set(charts.filter(c=>c.title!=='模型性能比较').flatMap(c=>c.figure.data.map(t=>t.name)))];
        setOptions('series-choice',names,series,true);
        const select=document.getElementById('series-choice');select.hidden=true;
        let popover=document.getElementById('snapshot-series');
        if(!popover){popover=document.createElement('details');popover.id='snapshot-series';popover.className='snapshot-multiselect';select.after(popover);}
        const summary=document.createElement('summary');summary.textContent=series.length?series.join('、'):'全部序列';summary.className='snapshot-select';
        const menu=document.createElement('div');menu.className='snapshot-series-menu';
        names.forEach(name=>{const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=series.includes(name);
            check.onchange=()=>{[...select.options].find(o=>o.value===name).selected=check.checked;
                series=[...select.selectedOptions].map(o=>o.value);summary.textContent=series.length?series.join('、'):'全部序列';updateCharts();};
            label.append(check,document.createTextNode(name));menu.append(label);});
        popover.replaceChildren(summary,menu);
        ['phase','metric'].forEach(key=>document.getElementById(key+'-choice').value=key==='phase'?phase:metric);
        document.getElementById('chart-format').value=ui.format||'html';
        const semanticSelect=document.getElementById('semantic-event');if(semanticSelect)semanticSelect.value=String(semantic);
    }
    async function switchModule(module,push){
        if(!labels[module])return;
        captureDetails();
        document.querySelectorAll('#module-view .js-plotly-plot').forEach(el=>{graphMemory.set(memoryKey(el.id),viewState(el));Plotly.purge(el);});
        active=module;queue=[];
        document.querySelectorAll('[data-module]').forEach(el=>{el.classList.toggle('selected',el.dataset.module===module);el.setAttribute('aria-selected',String(el.dataset.module===module));});
        Object.keys(labels).forEach(key=>document.getElementById('actions-'+key).style.display=key===module?'':'none');
        document.getElementById('chart-tools').style.display=module==='report'?'none':'';
        document.getElementById('module-view').replaceChildren(make(saved.views[module]));controls();restoreDetails();
        const graphs=queue;queue=[];
        await Promise.all(graphs.filter(item=>!['selected-chart','semantic-chart','performance-chart'].includes(item.id)).map(item=>plot(item)));
        await showSemantic();await updateCharts();
        if(window.MathJax?.startup?.promise){await MathJax.startup.promise;await MathJax.typesetPromise([document.getElementById('module-view')]);}
        if(push)location.hash=module;
        if(initial){initial=false;requestAnimationFrame(()=>{
            const scroll=ui.scroll||{};window.scrollTo(scroll.x||0,scroll.y||0);document.getElementById('sidebar').scrollTop=scroll.sidebar||0;
        });}
    }
    function blobDownload(content,name,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');
        a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    async function copyReport(){const text=saved.reports.report||'';if(navigator.clipboard?.writeText){try{await navigator.clipboard.writeText(text);return;}catch(_){/* file URLs may require the fallback */}}
        const area=document.createElement('textarea');area.value=text;document.body.append(area);area.select();document.execCommand('copy');area.remove();}
    async function handleDownload(id){
        const message=document.getElementById('export-message');
        if(id==='chart-download')message.textContent='正在导出…';
        try{
            if(id==='report-download'){if(saved.reports.report)blobDownload(saved.reports.report,'HydroTune-report.md','text/markdown');return;}
            if(id==='corrected-download'){
                const records=['文件,记录,实测流量_m3s,模拟流量_m3s,校正流量_m3s'];
                Object.entries(saved.corrected).forEach(([name,value])=>{
                    function array(data){if(Array.isArray(data))return data;const bytes=Uint8Array.from(atob(data.bdata),c=>c.charCodeAt(0));return Array.from(new Float64Array(bytes.buffer));}
                    const obs=array(value.observed),sim=array(value.simulated),corrected=array(value.corrected);
                    obs.forEach((v,i)=>records.push('"'+name.replaceAll('"','""')+'",'+i+','+[v,sim[i],corrected[i]].join(',')));
                });blobDownload('\ufeff'+records.join('\n'),'HydroTune-XGB.csv','text/csv');return;
            }
            const el=document.getElementById('selected-chart');if(!el?.data){message.textContent='当前没有可导出的图表。';return;}
            const format=document.getElementById('chart-format').value,name='HydroTune-'+active+'-'+(choice||'chart').replace(/[^\p{L}\p{N}_.-]/gu,'_');
            if(format==='html'){
                const script=document.querySelector('head script').textContent;
                const figure=JSON.stringify({data:el.data,layout:el.layout},(key,value)=>ArrayBuffer.isView(value)?Array.from(value):value).replaceAll('<','\\u003c');
                blobDownload('<!doctype html><meta charset="utf-8"><title>'+name+'</title><div id="plot"></div><script>'+script.replaceAll('</script','<\\/script')+'<\/script><script>const f='+figure+';Plotly.newPlot("plot",f.data,{...f.layout,width:undefined,autosize:true},{responsive:true,displaylogo:false});<\/script>',name+'.html','text/html');
            }else await Plotly.downloadImage(el,{format,filename:name,width:1400,height:700});
            message.textContent='图表已导出。';
        }catch(_){message.textContent='当前浏览器无法导出静态图，请改用 HTML。';}
    }
    root.append(make(saved.shell));queue=[];
    const badge=document.createElement('div');badge.className='snapshot-badge';
    badge.textContent='离线快照 · '+saved.created.replace('T',' ')+' · '+(saved.busy?'保存时刻的运行状态；任务不会继续执行':'保存时刻的工作台状态');
    document.querySelector('.task-strip').prepend(badge);
    // Capture the initial expanded/collapsed state before switchModule records defaults.
    restoreDetails();
    window.addEventListener('hashchange',()=>{const module=location.hash.slice(1);if(module!==active&&labels[module])switchModule(module,false);});
    window.addEventListener('resize',()=>document.querySelectorAll('#module-view .js-plotly-plot').forEach(el=>Plotly.Plots.resize(el)));
    document.addEventListener('click',event=>{const popover=document.getElementById('snapshot-series');if(popover&&!popover.contains(event.target))popover.open=false;});
    switchModule(active,false).catch(error=>{const p=document.createElement('p');p.className='notice warning';p.textContent='快照展示出现错误：'+error.message;root.prepend(p);});
})();

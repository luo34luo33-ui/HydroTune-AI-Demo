/* Only presentation state and explicitly named non-sensitive fields leave the UI. */
(function () {
    function detailKey(el) {
        if (el.id) return el.id;
        const title = el.querySelector(':scope > summary');
        return (el.closest('#sidebar') ? 'sidebar/' : 'view/') + (title ? title.textContent.trim() : '');
    }
    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        hydrotune: {
            capture: function (clicks, renderKey, revision, module, choice, series, phase, metric, format, values, ids) {
                if (!clicks) return window.dash_clientside.no_update;
                const forms = {};
                ids.forEach((id, i) => {
                    const input = document.getElementById(id);
                    forms[id] = input && input.tagName === 'INPUT' && input.type !== 'password' ? input.value : values[i];
                });
                const details = {};
                document.querySelectorAll('#workspace details').forEach(el => details[detailKey(el)] = el.open);
                const graphs = {};
                document.querySelectorAll('#module-view .js-plotly-plot').forEach(el => {
                    const wrapper = el.closest('[id]');
                    const id = el.id || (wrapper && wrapper.id);
                    if (!id) return;
                    const layout = {};
                    Object.keys(el.layout || {}).forEach(key => {
                        if (/^[xy]axis\d*$/.test(key)) {
                            const axis = el.layout[key];
                            layout[key] = {range:axis.range, autorange:axis.autorange};
                        } else if (/^scene\d*$/.test(key)) layout[key] = {camera:el.layout[key].camera};
                    });
                    graphs[id] = {layout, visibility:(el.data || []).map(t => ({name:t.name, visible:t.visible}))};
                });
                const messages = {};
                ['task-message','session-note','upload-message','llm-message','llm-status','config-note','export-message'].forEach(id => {
                    const el = document.getElementById(id); messages[id] = el ? el.textContent : '';
                });
                const sidebar = document.getElementById('sidebar');
                return {clicks, revision, render_key:renderKey, ready:!document.querySelector('#module-view[data-dash-is-loading="true"], #module-view [data-dash-is-loading="true"]'),
                    module, choice, series:series || [], phase, metric, format, forms, details, graphs, messages,
                    scroll:{x:window.scrollX,y:window.scrollY,sidebar:sidebar ? sidebar.scrollTop : 0},
                    viewport:{width:window.innerWidth,height:window.innerHeight}};
            }
        }
    });
})();

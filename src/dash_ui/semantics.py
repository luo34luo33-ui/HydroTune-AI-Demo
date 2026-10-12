"""Deterministic flood portraits shared by the workbench and offline page."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.app.dash_analysis import _chart


LABELS = {
    'event_id': '场次编码', 'magnitude': '相对量级', 'peak_flow': '洪峰 (m³/s)',
    'flood_volume': '洪量 (万m³)', 'duration': '历时 (步数)', 'rise_type': '涨水特征',
    'recession_type': '退水特征', 'peak_shape': '峰型', 'rainfall_driven': '降雨驱动标签',
    'representative_score': '代表性规则评分', 'selection_reason': '推荐理由',
    'start_date': '编码参考开始日期', 'end_date': '编码参考结束日期',
    'baseflow': '基流 (m³/s)', 'peak_ratio': '峰值比', 'total_precip': '累计降雨 (mm)',
}
TRANSLATIONS = {
    'magnitude': dict(small='偏小', medium='中等', large='偏大', extreme='极大'),
    'rise_type': dict(rapid='快涨', moderate='平稳上涨', slow='缓涨'),
    'recession_type': dict(rapid='快退', moderate='平稳退水', slow='缓退'),
    'peak_shape': {'rounded': '圆钝峰', 'sharp': '尖峰', 'multi-peak': '多峰'},
    'rainfall_driven': dict(high='高', medium='中', low='低'),
}
NOTE = '标签来自现有规则编码。量级为当前样本内相对分类，非洪水等级或重现期；降雨驱动标签不代表因果结论。洪量沿用扣除估计基流后的计算；历时按记录步数计。'


def code_rows(prepared):
    return [{'序号': index + 1, **{LABELS[key]: TRANSLATIONS.get(key, {}).get(value, value)
            for key, value in code.to_dict().items()}}
            for index, code in enumerate(prepared['codes'])]


def options(prepared):
    return [{'label': f"#{i+1} · {code.event_id} · {boundary.get('file_name', prepared['events'][0]['name'])}", 'value': i}
            for i, (code, boundary) in enumerate(zip(prepared['codes'], prepared['semantic_boundaries']))]


def portrait(prepared, index):
    code = prepared['codes'][index]
    boundary = prepared['semantic_boundaries'][index]
    start, end, peak = boundary['start_idx'], boundary['end_idx'], boundary['peak_idx']
    dates = pd.concat([event['dates'] for event in prepared['events']], ignore_index=True).iloc[start:end+1].reset_index(drop=True)
    flow = np.concatenate([event['flow'] for event in prepared['events']])[start:end+1]
    rain = np.concatenate([event['precip'] for event in prepared['events']])[start:end+1]
    event = dict(dates=dates, flow=flow, precip=rain)
    figure = go.Figure(_chart(event, {}, '洪水语义画像 · 降雨与径流', 0))
    parsed = pd.to_datetime(dates, errors='coerce')
    dated = parsed.notna().all()
    x = list(parsed) if dated else list(range(len(flow)))
    peak -= start
    figure.add_vrect(x0=x[0], x1=x[peak], fillcolor='#08A6AD', opacity=.09, line_width=0, layer='below')
    figure.add_vrect(x0=x[peak], x1=x[-1], fillcolor='#092B4C', opacity=.05, line_width=0, layer='below')
    markers = {}
    for position, label in [(0, '起点'), (peak, '主峰'), (len(flow)-1, '终点')]:
        markers.setdefault(position, []).append(label)
    figure.add_scatter(x=[x[p] for p in markers], y=[flow[p] for p in markers],
                       mode='markers+text', text=[' / '.join(v) for v in markers.values()],
                       textposition='top center', marker=dict(size=9, color='#08A6AD', line=dict(color='white', width=2)),
                       showlegend=False, name='场次标记', hovertemplate='%{text}<br>%{y:.2f} m³/s<extra></extra>')
    figure.update_layout(height=410, margin=dict(l=55,r=55,t=55,b=75),
                         legend=dict(orientation='h', y=-.2), font=dict(family='Arial, Microsoft YaHei'),
                         paper_bgcolor='white', plot_bgcolor='white')
    tags = [f"相对量级 · {TRANSLATIONS['magnitude'][code.magnitude]}",
            TRANSLATIONS['rise_type'][code.rise_type], TRANSLATIONS['recession_type'][code.recession_type],
            TRANSLATIONS['peak_shape'][code.peak_shape], f"降雨驱动 · {TRANSLATIONS['rainfall_driven'][code.rainfall_driven]}"]
    unit = '小时' if prepared['timestep'] == 'hourly' else '天'
    metrics = [('洪峰', f'{code.peak_flow:,.2f}', 'm³/s'), ('洪量', f'{code.flood_volume:,.2f}', '万m³'),
               ('历时', str(code.duration), unit), ('累计降雨', f'{code.total_precip:,.2f}', 'mm')]
    period = f'{parsed.iloc[0]:%Y-%m-%d %H:%M} — {parsed.iloc[-1]:%Y-%m-%d %H:%M}' if dated else '按记录序号展示；编码日期为参考时间，不代表真实日期。'
    sentence = f"本场洪水呈{tags[1]}、{tags[2]}特征，峰型为{tags[3]}，相对量级{TRANSLATIONS['magnitude'][code.magnitude]}。"
    return dict(figure=figure.to_plotly_json(), tags=tags, metrics=metrics, period=period,
                sentence=sentence, source=boundary.get('file_name', prepared['events'][0]['name']))

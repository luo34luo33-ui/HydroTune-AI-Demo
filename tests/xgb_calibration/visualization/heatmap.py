# -*- coding: utf-8 -*-
"""
场次选择热力图
用于分析哪些场次被频繁选择用于率定
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import get_cmap
from pathlib import Path


def plot_selection_heatmap(
    results_df: pd.DataFrame,
    flood_events: list,
    model: str,
    calib_ratio: str,
    output_dir: Path,
    top_pct: float = 0.2,
):
    """绘制场次选择频率热力图"""
    
    subset = results_df[
        (results_df['model'] == model) & 
        (results_df['calibration_ratio'] == calib_ratio)
    ].copy()
    
    n_top = max(1, int(len(subset) * top_pct))
    subset = subset.nlargest(n_top, 'nse_mean')
    
    flood_to_idx = {f: i for i, f in enumerate(flood_events)}
    selection_count = np.zeros(len(flood_events), dtype=int)
    
    for _, row in subset.iterrows():
        calib_events = row['calib_events'].split(',')
        for event in calib_events:
            event = event.strip()
            if event in flood_to_idx:
                selection_count[flood_to_idx[event]] += 1
    
    freq = selection_count / n_top * 100
    
    n_floods = len(flood_events)
    y_positions = np.arange(n_floods)
    
    sorted_idx = np.argsort(freq)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    cmap = get_cmap('RdYlGn')
    colors = cmap(freq[sorted_idx] / 100)
    
    ax.barh(y_positions, freq[sorted_idx], color=colors, edgecolor='none', height=0.8)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([flood_events[i] for i in sorted_idx], fontsize=7)
    ax.set_xlabel('出现频率 (%)', fontsize=12)
    ax.set_ylabel('洪水场次', fontsize=12)
    
    model_name_cn = {'tank': 'Tank', 'hbv': 'HBV', 'xaj': '新安江'}
    ratio_cn = {'5场': '5场', '10场': '10场', '15场': '15场'}
    
    title = f'{model_name_cn.get(model, model)}模型 - {ratio_cn.get(calib_ratio, calib_ratio)}率定\nTop 20% 高NSE组合中洪水被选频率'
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for i, idx in enumerate(sorted_idx):
        if freq[idx] > 0:
            ax.text(freq[idx] + 1, i, f'{freq[idx]:.0f}%', va='center', ha='left', fontsize=7)
    
    n_total = len(flood_events)
    ax.text(0.98, 0.02, f'率定样本数: {calib_ratio.replace("场", "")} | Top {n_top}个组合 | 共{n_total}场洪水', 
           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    filename = f'selection_freq_{model}_{calib_ratio.replace("场", "")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename
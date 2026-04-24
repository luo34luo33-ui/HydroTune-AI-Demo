# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import get_cmap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

INPUTS_DIR = Path('E:/HydroTune-AI-Demo/tests/inputs')
RESULTS_FILE = Path('E:/HydroTune-AI-Demo/tests/outputs/xgb_correction/xgb_correction_results.csv')
OUTPUT_DIR = Path('E:/HydroTune-AI-Demo/tests/outputs/xgb_correction/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_flood_events():
    flood_files = sorted(INPUTS_DIR.glob('*.csv'))
    flood_ids = [f.stem for f in flood_files]
    return sorted(flood_ids)

def load_results():
    df = pd.read_csv(RESULTS_FILE)
    return df

def get_top_n_selection(df, model, calib_ratio, flood_events, top_pct=0.2):
    subset = df[(df['model'] == model) & (df['calibration_ratio'] == calib_ratio)].copy()
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
    return freq, selection_count, n_top

def plot_frequency_bar(freq, selection_count, n_selected, model, calib_ratio, flood_events, save_path):
    n_floods = len(flood_events)
    y_positions = np.arange(n_floods)
    
    sorted_idx = np.argsort(freq)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    cmap = get_cmap('RdYlGn')
    colors = cmap(freq[sorted_idx] / 100)
    
    bars = ax.barh(y_positions, freq[sorted_idx], color=colors, edgecolor='none', height=0.8)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([flood_events[i] for i in sorted_idx], fontsize=7)
    ax.set_xlabel('出现频率 (%)', fontsize=12, fontname='SimSun')
    ax.set_ylabel('洪水场次', fontsize=12, fontname='SimSun')
    
    model_name_cn = {'tank': 'Tank', 'hbv': 'HBV', 'xaj': '新安江'}
    ratio_cn = {'5场': '5场', '10场': '10场', '15场': '15场'}
    
    title = f'{model_name_cn.get(model, model)}模型 - {ratio_cn.get(calib_ratio, calib_ratio)}率定\nTop 20% 高NSE组合中洪水被选频率'
    ax.set_title(title, fontsize=14, fontname='SimSun', pad=15)
    
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for i, idx in enumerate(sorted_idx):
        if freq[idx] > 0:
            ax.text(freq[idx] + 1, i, f'{freq[idx]:.0f}%', va='center', ha='left', fontsize=7)
    
    n_total = len(flood_events)
    ax.text(0.98, 0.02, f'率定样本数: {calib_ratio.replace("场", "")} | Top {n_selected}个组合 | 共{n_total}场洪水', 
           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'已保存: {save_path}')

def main():
    print('正在加载数据...')
    flood_events = get_flood_events()
    print(f'共发现 {len(flood_events)} 场洪水')
    
    df = load_results()
    print(f'共加载 {len(df)} 条实验记录')
    
    models = ['tank', 'hbv', 'xaj']
    calib_ratios = ['5场', '10场', '15场']
    
    for model in models:
        for calib_ratio in calib_ratios:
            print(f'\n正在处理 {model} - {calib_ratio}...')
            freq, count, n_top = get_top_n_selection(df, model, calib_ratio, flood_events, top_pct=0.2)
            
            filename = f'selection_freq_{model}_{calib_ratio.replace("场", "")}.png'
            save_path = OUTPUT_DIR / filename
            
            plot_frequency_bar(freq, count, n_top, model, calib_ratio, flood_events, save_path)
    
    print('\n所有图表绘制完成！')

if __name__ == '__main__':
    main()
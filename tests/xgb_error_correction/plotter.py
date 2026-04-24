# -*- coding: utf-8 -*-
"""
绘图模块
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from xgb_error_correction.config import OUTPUT_DIR


PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results() -> pd.DataFrame:
    """加载累积结果"""
    csv_path = os.path.join(OUTPUT_DIR, 'xgb_correction_results.csv')
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(csv_path, encoding='gbk')
    return pd.DataFrame()


def append_result(result: Dict):
    """追加单条结果到CSV"""
    csv_path = os.path.join(OUTPUT_DIR, 'xgb_correction_results.csv')
    df_new = pd.DataFrame([result])
    
    # 确保列顺序一致
    expected_cols = [
        'model', 'calibration_ratio', 'nse_mean', 'nse_std', 'time_seconds',
        'calib_nse', 'calib_events', 'run_iter',
        'calib_nse_raw', 'non_calib_nse_raw', 'all_nse_raw',
        'calib_nse_corrected', 'non_calib_nse_corrected', 'all_nse_corrected'
    ]
    for col in expected_cols:
        if col not in df_new.columns:
            df_new[col] = None
    
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_existing = pd.read_csv(csv_path, encoding='gbk')
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(csv_path, index=False, encoding='utf-8')


def plot_nse_comparison():
    """绘制NSE对比柱状图"""
    df = load_results()
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    ratios = ['5场', '10场', '15场', '75%基准']
    x = np.arange(len(ratios))
    width = 0.25
    
    colors = {'tank': '#e74c3c', 'hbv': '#3498db', 'xaj': '#2ecc71'}
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        nses = []
        stds = []
        for ratio in ratios:
            r = model_data[model_data['calibration_ratio'] == ratio]
            if len(r) > 0:
                nses.append(r['nse_mean'].mean())
                stds.append(r['nse_std'].mean() if 'nse_std' in r.columns else 0)
            else:
                nses.append(0)
                stds.append(0)
        
        ax.bar(x + i * width, nses, width, label=model.upper(), 
               color=colors.get(model, '#95a5a6'), yerr=stds, capsize=3)
    
    ax.set_xlabel('Calibration Ratio')
    ax.set_ylabel('NSE')
    ax.set_title('NSE Comparison: XGBoost Error Correction vs Benchmark')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['5 Events', '10 Events', '15 Events', '75% Benchmark'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'nse_comparison.png'), dpi=150)
    plt.close()


def plot_time_efficiency():
    """绘制时间效率柱状图"""
    df = load_results()
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    ratios = ['5场', '10场', '15场', '75%基准']
    x = np.arange(len(ratios))
    width = 0.25
    
    colors = {'tank': '#e74c3c', 'hbv': '#3498db', 'xaj': '#2ecc71'}
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        times = []
        for ratio in ratios:
            r = model_data[model_data['calibration_ratio'] == ratio]
            if len(r) > 0:
                times.append(r['time_seconds'].mean())
            else:
                times.append(0)
        
        ax.bar(x + i * width, times, width, label=model.upper(), 
               color=colors.get(model, '#95a5a6'))
    
    ax.set_xlabel('Calibration Ratio')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Efficiency Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['5 Events', '10 Events', '15 Events', '75% Benchmark'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_efficiency.png'), dpi=150)
    plt.close()


def plot_convergence():
    """绘制收敛曲线（按run_iter）"""
    df = load_results()
    if df.empty or 'run_iter' not in df.columns:
        return
    
    # 检查是否有足够的数据
    has_data = False
    for model in ['tank', 'hbv', 'xaj']:
        model_data = df[df['model'] == model]
        for ratio in ['5场', '10场', '15场']:
            ratio_data = model_data[model_data['calibration_ratio'] == ratio]
            if len(ratio_data) > 1:
                has_data = True
                break
    
    if not has_data:
        
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['tank', 'hbv', 'xaj']
    
    for ax, model in zip(axes, models):
        model_data = df[df['model'] == model]
        
        for ratio in ['5场', '10场', '15场']:
            ratio_data = model_data[model_data['calibration_ratio'] == ratio]
            if len(ratio_data) > 0:
                label_map = {'5场': '5Ev', '10场': '10Ev', '15场': '15Ev'}
                ax.plot(ratio_data['run_iter'], ratio_data['nse_mean'], 
                       marker='o', label=label_map.get(ratio, ratio))
        
        ax.set_xlabel('Run Iteration')
        ax.set_ylabel('NSE')
        ax.set_title(f'{model.upper()} Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'convergence.png'), dpi=150)
    plt.close()
    


def update_plots():
    """更新所有图表"""
    plot_nse_comparison()
    plot_time_efficiency()
    plot_convergence()
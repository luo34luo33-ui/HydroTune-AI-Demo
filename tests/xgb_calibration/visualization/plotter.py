# -*- coding: utf-8 -*-
"""
绘图工具
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR


PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


class Plotter:
    """绘图器"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or PLOTS_DIR
    
    def load_results(self) -> pd.DataFrame:
        """加载累积结果"""
        csv_path = os.path.join(OUTPUT_DIR, 'xgb_correction_results.csv')
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(csv_path, encoding='gbk')
        return pd.DataFrame()
    
    def append_result(self, result: Dict):
        """追加单条结果到CSV"""
        csv_path = os.path.join(OUTPUT_DIR, 'xgb_correction_results.csv')
        df_new = pd.DataFrame([result])
        
        expected_cols = [
            'model', 'calibration_ratio', 'nse_mean', 'nse_std', 'time_seconds',
            'calib_nse', 'calib_events', 'run_iter',
            'calib_nse_raw', 'non_calib_nse_raw',
            'calib_nse_corrected', 'non_calib_nse_corrected'
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
    
    def plot_nse_comparison(self):
        """绘制NSE对比柱状图"""
        df = self.load_results()
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
            for ratio in ratios:
                r = model_data[model_data['calibration_ratio'] == ratio]
                if len(r) > 0:
                    nses.append(r['nse_mean'].mean())
                else:
                    nses.append(0)
            
            ax.bar(x + i * width, nses, width, label=model.upper(), 
                   color=colors.get(model, '#95a5a6'))
        
        ax.set_xlabel('Calibration Ratio')
        ax.set_ylabel('NSE')
        ax.set_title('NSE Comparison: XGBoost Error Correction vs Benchmark')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['5 Events', '10 Events', '15 Events', '75% Benchmark'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'nse_comparison.png'), dpi=150)
        plt.close()
    
    def plot_time_efficiency(self):
        """绘制时间效率柱状图"""
        df = self.load_results()
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
        plt.savefig(os.path.join(self.output_dir, 'time_efficiency.png'), dpi=150)
        plt.close()
    
    def update_plots(self):
        """更新所有图表"""
        self.plot_nse_comparison()
        self.plot_time_efficiency()
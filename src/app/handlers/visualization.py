# -*- coding: utf-8 -*-
"""
可视化模块
从 app.py 结果展示部分提取
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
from src.llm_api import get_language


class VisualizationHandler:
    """Result Visualization Handler"""
    
    @staticmethod
    def plot_flow_comparison(observed: np.ndarray, simulated: np.ndarray, 
                             title: str = None, timestep: str = 'daily'):
        """Plot flow time series comparison"""
        lang = get_language()
        if title is None:
            title = "Flow Time Series Comparison" if lang == 'en' else "流量过程对比"
        
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        
        if lang == 'en':
            xlabel_text = "Time (days)" if timestep == 'daily' else "Time (h)"
            flow_label = "Observed Flow"
            sim_label = "Simulated Flow"
        else:
            xlabel_text = "时间(天)" if timestep == 'daily' else "时间(h)"
            flow_label = "观测流量"
            sim_label = "模拟流量"
        
        ax.plot(observed, 'b-', linewidth=1.5, label=flow_label)
        ax.plot(simulated, 'r--', linewidth=1.5, label=sim_label)
        ax.fill_between(range(len(observed)), 0, observed, alpha=0.2, color='blue')
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel(r'Flow ($m^3/s$)' if lang == 'en' else r'流量 ($m^3/s$)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_scatter(observed: np.ndarray, simulated: np.ndarray,
                     title: str = None):
        """Plot scatter plot"""
        lang = get_language()
        if title is None:
            title = "Observed vs Simulated Scatter Plot" if lang == 'en' else "观测-模拟散点图"
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        max_val = max(np.max(observed), np.max(simulated)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 Line' if lang == 'en' else '1:1线')
        ax.scatter(observed, simulated, alpha=0.5, s=20)
        
        ax.set_xlabel('Observed Flow ($m^3/s$)' if lang == 'en' else '观测流量 ($m^3/s$)', fontsize=12)
        ax.set_ylabel('Simulated Flow ($m^3/s$)' if lang == 'en' else '模拟流量 ($m^3/s$)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    @staticmethod
    def plot_metrics_table(calibration_results: Dict[str, Dict]):
        """绘制指标表格"""
        from src.hydro_calc import calc_nse, calc_rmse, calc_kge, calc_pbias
        
        metrics_data = []
        for model_name, result in calibration_results.items():
            if result is None:
                continue
            
            simulated = result.get('simulated')
            if simulated is None:
                continue
            
            try:
                obs = result.get('observed', result.get('calib_data', [None, None, None])[2])
                if obs is None:
                    continue
                
                nse = calc_nse(obs, simulated)
                rmse = calc_rmse(obs, simulated)
                kge = calc_kge(obs, simulated)
                pbias = calc_pbias(obs, simulated)
                
                metrics_data.append({
                    '模型': model_name,
                    'NSE': f"{nse:.4f}",
                    'KGE': f"{kge:.4f}",
                    'RMSE': f"{rmse:.2f}",
                    'Pbias': f"{pbias:.2f}%",
                })
            except:
                continue
        
        if metrics_data:
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            st.table(df)
        
        return metrics_data
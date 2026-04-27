# -*- coding: utf-8 -*-
"""
可视化模块
从 app.py 结果展示部分提取
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List


class VisualizationHandler:
    """结果可视化处理器"""
    
    @staticmethod
    def plot_flow_comparison(observed: np.ndarray, simulated: np.ndarray, 
                             title: str = "流量过程对比", timestep: str = 'daily'):
        """绘制流量过程对比图"""
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        
        xlabel_text = "时间(天)" if timestep == 'daily' else "时间(h)"
        
        ax.plot(observed, 'b-', linewidth=1.5, label='观测流量')
        ax.plot(simulated, 'r--', linewidth=1.5, label='模拟流量')
        ax.fill_between(range(len(observed)), 0, observed, alpha=0.2, color='blue')
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel(r'流量 ($m^3/s$)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_scatter(observed: np.ndarray, simulated: np.ndarray,
                     title: str = "观测-模拟散点图"):
        """绘制散点图"""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        max_val = max(np.max(observed), np.max(simulated)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1线')
        ax.scatter(observed, simulated, alpha=0.5, s=20)
        
        ax.set_xlabel('观测流量 ($m^3/s$)', fontsize=12)
        ax.set_ylabel('模拟流量 ($m^3/s$)', fontsize=12)
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
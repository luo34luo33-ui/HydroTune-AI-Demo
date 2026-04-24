# -*- coding: utf-8 -*-
"""
过程线对比绘图
展示实测流量、原始模拟、XGB校正后模拟的三曲线对比
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import WARMUP_STEPS
from core.data_loader import calc_nse


def get_plots_dir():
    """动态获取绘图输出目录"""
    from config import OUTPUT_DIR
    return os.path.join(OUTPUT_DIR, 'hydrographs')


def plot_comparison(
    event: Dict,
    sim: np.ndarray,
    corrected_sim: Optional[np.ndarray],
    params: Dict,
    model_name: str,
    selector_name: str,
    n_calib: int,
    output_subdir: str = None,
) -> Dict[str, float]:
    """绘制过程线对比图
    
    Args:
        event: 场次数据，包含 precip, flow
        sim: 原始模拟流量
        corrected_sim: XGB校正后模拟流量（可选）
        params: 率定参数
        model_name: 模型名称
        selector_name: 选择器名称
        n_calib: 率定场次数
        output_subdir: 输出子目录
        
    Returns:
        包含各曲线NSE的字典
    """
    precip = event['precip'][WARMUP_STEPS:]
    obs = event['flow'][WARMUP_STEPS:]
    sim_plot = sim[WARMUP_STEPS:]
    
    from core.data_loader import calc_nse
    nse_raw = calc_nse(obs, sim_plot)
    
    if corrected_sim is not None:
        corrected_plot = corrected_sim[WARMUP_STEPS:]
        nse_corrected = calc_nse(obs, corrected_plot)
    else:
        corrected_plot = None
        nse_corrected = None
    
    time_axis = np.arange(len(obs))
    
    fig, ax_flow = plt.subplots(figsize=(14, 6))
    ax_precip = ax_flow.twinx()
    ax_precip.spines['top'].set_position(('outward', 0))
    
    ax_flow.plot(time_axis, obs, label='Observed', linewidth=1.5, color='blue')
    ax_flow.plot(time_axis, sim_plot, label='Simulated (Raw)', linewidth=1.5, color='red', linestyle='--')
    
    if corrected_plot is not None:
        ax_flow.plot(time_axis, corrected_plot, label='Simulated (Corrected)', 
                     linewidth=1.5, color='green')
    
    precip_max = precip.max() if precip.max() > 0 else 1
    ax_precip.bar(time_axis, precip, width=1.0, color='lightblue', edgecolor='none', alpha=0.7)
    ax_precip.set_ylim(precip_max * 5, 0)
    ax_precip.set_ylabel('Precipitation (mm)', fontsize=9)
    
    param_lines = [f"{k}: {v:.3f}" for k, v in list(params.items())[:8]]
    param_text = "\n".join(param_lines)
    ax_flow.text(0.01, 0.99, param_text, transform=ax_flow.transAxes, fontsize=6,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    nse_text = f"NSE (Raw): {nse_raw:.4f}"
    if nse_corrected is not None:
        nse_text += f"\nNSE (Corrected): {nse_corrected:.4f}"
    ax_flow.text(0.99, 0.99, nse_text, transform=ax_flow.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax_flow.set_title(f"{model_name.upper()} - {selector_name} - {n_calib} events - {event['name']}")
    ax_flow.legend(loc='upper left', fontsize=9)
    ax_flow.grid(True, alpha=0.3)
    ax_flow.set_xlabel('Time Step (hour)', fontsize=10)
    ax_flow.set_ylabel('Flow (m³/s)', fontsize=10)
    
    plt.tight_layout()
    
    subdir = output_subdir or model_name
    out_dir = os.path.join(get_plots_dir(), subdir)
    os.makedirs(out_dir, exist_ok=True)
    
    filename = f"{model_name}_{selector_name}_{n_calib}_{event['name']}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    import pandas as pd
    csv_data = {
        'time': time_axis,
        'observed': obs,
        'simulated_raw': sim_plot,
    }
    if corrected_plot is not None:
        csv_data['simulated_corrected'] = corrected_plot
    pd.DataFrame(csv_data).to_csv(
        os.path.join(out_dir, filename.replace('.png', '.csv')),
        index=False
    )
    
    return {
        'nse_raw': nse_raw,
        'nse_corrected': nse_corrected,
    }


def plot_event_list(
    events: list,
    results: list,
    model_name: str,
    selector_name: str,
    n_calib: int,
    test_event_indices: set = None,
) -> Dict[str, float]:
    """批量绘制场次过程线对比图
    
    Args:
        events: 场次数据列表
        results: 对应的模拟结果列表
        model_name: 模型名称
        selector_name: 选择器名称
        n_calib: 率定场次数
        test_event_indices: XGB测试集场次索引（仅对这些场次计算统计）
        
    Returns:
        汇总的NSE统计
    """
    nse_raw_list = []
    nse_corrected_list = []
    n_events_plotted = 0
    
    for idx, (event, result) in enumerate(zip(events, results)):
        if test_event_indices is not None and idx not in test_event_indices:
            continue
        
        sim = result['sim']
        corrected_sim = result.get('sim_corrected', None)
        
        if corrected_sim is None:
            continue
        
        plot_comparison(
            event=event,
            sim=sim,
            corrected_sim=corrected_sim,
            params=result.get('params', {}),
            model_name=model_name,
            selector_name=selector_name,
            n_calib=n_calib,
        )
        
        nse_dict = {
            'nse_raw': calc_nse(event['flow'][WARMUP_STEPS:], sim[WARMUP_STEPS:]),
            'nse_corrected': calc_nse(event['flow'][WARMUP_STEPS:], corrected_sim[WARMUP_STEPS:]),
        }
        
        if nse_dict['nse_raw'] > -10:
            nse_raw_list.append(nse_dict['nse_raw'])
        if nse_dict['nse_corrected'] > -10:
            nse_corrected_list.append(nse_dict['nse_corrected'])
        n_events_plotted += 1
    
    return {
        'nse_raw_mean': np.mean(nse_raw_list) if nse_raw_list else -9999,
        'nse_raw_std': np.std(nse_raw_list) if nse_raw_list else 0,
        'nse_corrected_mean': np.mean(nse_corrected_list) if nse_corrected_list else -9999,
        'nse_corrected_std': np.std(nse_corrected_list) if nse_corrected_list else 0,
        'n_events': n_events_plotted,
    }
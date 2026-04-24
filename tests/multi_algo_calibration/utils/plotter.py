# -*- coding: utf-8 -*-
"""
绘图工具
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from ..configs import OUTPUT_PLOTS, OUTPUT_DATA, WARMUP_STEPS


def plot_event(
    event: Dict,
    sim: np.ndarray,
    params: Dict,
    model_name: str,
    algo_name: str,
    output_subdir: str,
) -> float:
    """绘制单个场次的拟合图"""
    precip_plot = event['precip'][WARMUP_STEPS:]
    obs_plot = event['flow'][WARMUP_STEPS:]
    sim_plot = sim[WARMUP_STEPS:]
    
    from .data_loader import calc_nse
    nse = calc_nse(obs_plot, sim_plot)
    time_axis = np.arange(len(obs_plot))
    
    fig, ax_flow = plt.subplots(figsize=(12, 5))
    ax_precip = ax_flow.twinx()
    ax_precip.spines['top'].set_position(('outward', 0))
    
    ax_flow.plot(time_axis, obs_plot, label='Obs', linewidth=1.5, color='blue')
    ax_flow.plot(time_axis, sim_plot, label='Sim', linewidth=1.5, color='red')
    
    ax_precip.bar(time_axis, precip_plot, width=1.0, color='lightblue', edgecolor='none', alpha=0.7)
    ax_precip.set_ylim(max(precip_plot) * 5, 0)
    ax_precip.set_ylabel('Precip (mm)', fontsize=9)
    
    param_text = "\n".join([f"{k}: {v:.3f}" for k, v in params.items()])
    ax_flow.text(0.02, 0.98, param_text, transform=ax_flow.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_flow.text(0.98, 0.98, f"NSE: {nse:.4f}", transform=ax_flow.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax_flow.set_title(f"{model_name}-{algo_name} {event['name']}")
    ax_flow.legend(loc='upper left')
    ax_flow.grid(True, alpha=0.3)
    ax_flow.set_xlabel('Time Step (h)')
    ax_flow.set_ylabel('Flow (m³/s)')
    plt.tight_layout()
    
    filename = f"{model_name.lower()}_{algo_name.lower()}_{event['name']}.png"
    plt.savefig(os.path.join(OUTPUT_PLOTS, output_subdir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    pd.DataFrame({'obs': obs_plot, 'sim': sim_plot}).to_csv(
        os.path.join(OUTPUT_DATA, output_subdir, filename.replace('.png', '.csv')),
        index=False
    )
    
    return nse
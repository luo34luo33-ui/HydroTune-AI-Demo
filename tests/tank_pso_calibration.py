# -*- coding: utf-8 -*-
"""
Tank模型 + PSO算法率定
独立运行，不依赖项目其他模块
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

sys.path.insert(0, os.path.dirname(__file__))
from calibration_base import (
    DATA_DIR, COL_MAPPING, CATCHMENT_AREA, WARMUP_STEPS, MAX_ITERATIONS,
    OUTPUT_PARAMS, OUTPUT_PLOTS, OUTPUT_DATA, MUSKINGUM_BOUNDS
)
from algos.pso import optimize_pso
from models.tank import run_tank_model, TANK_PARAM_BOUNDS
from calibration_params import PSO_PARAMS


def calc_nse(observed, simulated):
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    if len(obs) == 0:
        return -9999
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return -9999
    return 1 - np.sum((obs - sim) ** 2) / denom


def muskingum_routing(upstream_flow, k, x):
    n = len(upstream_flow)
    if n == 0:
        return np.array([])
    dt = 1.0
    denom = k * (1 - x) + 0.5 * dt
    if denom == 0:
        return upstream_flow.copy()
    C0 = (-k * x + 0.5 * dt) / denom
    C1 = (k * x + 0.5 * dt) / denom
    C2 = (k * (1 - x) - 0.5 * dt) / denom
    routed = np.zeros(n)
    routed[0] = upstream_flow[0]
    for t in range(1, n):
        routed[t] = C0 * upstream_flow[t] + C1 * upstream_flow[t-1] + C2 * routed[t-1]
    return np.maximum(routed, 0)


def load_flood_events():
    csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
    events = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        rename_map = {COL_MAPPING[k]: k for k in COL_MAPPING if COL_MAPPING[k] in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
        if 'precip' not in df.columns or 'flow' not in df.columns:
            continue
        if 'evap' not in df.columns:
            df['evap'] = 0.0
        events.append({
            'name': os.path.basename(fpath).replace('.csv', ''),
            'precip': df['precip'].fillna(0).values,
            'evap': df['evap'].fillna(0).values,
            'flow': df['flow'].fillna(0).values,
            'upstream': df['upstream'].fillna(0).values if 'upstream' in df.columns else None,
        })
    print(f"加载了 {len(events)} 场洪水数据")
    return events


def run_calibration():
    print("=" * 60)
    print("Tank模型 + PSO算法 率定")
    print("=" * 60)
    
    events = load_flood_events()
    param_names = list(TANK_PARAM_BOUNDS.keys()) + ['k_routing', 'x_routing']
    bounds = list(TANK_PARAM_BOUNDS.values()) + [MUSKINGUM_BOUNDS['k_routing'], MUSKINGUM_BOUNDS['x_routing']]
    
    def objective(params_array):
        model_params = {k: v for k, v in zip(param_names[:-2], params_array[:-2])}
        try:
            nse_list = []
            for evt in events:
                sim = run_tank_model(evt['precip'], evt['evap'], model_params, CATCHMENT_AREA, 1.0)
                if evt['upstream'] is not None:
                    routed = muskingum_routing(evt['upstream'], params_array[-2], params_array[-1])
                    sim = sim + routed
                obs = evt['flow'][WARMUP_STEPS:]
                sim = sim[WARMUP_STEPS:]
                nse = calc_nse(obs, sim)
                if not np.isnan(nse):
                    nse_list.append(nse)
            return -np.mean(nse_list) if nse_list else 1e10
        except:
            return 1e10
    
    best_x, best_fun = optimize_pso(objective, bounds, MAX_ITERATIONS, len(param_names), PSO_PARAMS)
    best_params = {k: v for k, v in zip(param_names, best_x)}
    best_nse = -best_fun
    
    print(f"\n率定完成！最优NSE: {best_nse:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    
    pd.DataFrame([best_params]).to_csv(os.path.join(OUTPUT_PARAMS, "tank_pso_params.csv"), index=False)
    
    for evt in events:
        sim = run_tank_model(evt['precip'], evt['evap'], best_params, CATCHMENT_AREA, 1.0)
        if evt['upstream'] is not None:
            sim = sim + muskingum_routing(evt['upstream'], best_params['k_routing'], best_params['x_routing'])
        
        precip_plot = evt['precip'][WARMUP_STEPS:]
        obs_plot = evt['flow'][WARMUP_STEPS:]
        sim_plot = sim[WARMUP_STEPS:]
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
        
        param_text = "\n".join([f"{k}: {v:.3f}" for k, v in best_params.items()])
        ax_flow.text(0.02, 0.98, param_text, transform=ax_flow.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_flow.text(0.98, 0.98, f"NSE: {nse:.4f}", transform=ax_flow.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax_flow.set_title(f"Tank-PSO {evt['name']}")
        ax_flow.legend(loc='upper left')
        ax_flow.grid(True, alpha=0.3)
        ax_flow.set_xlabel('Time Step (h)')
        ax_flow.set_ylabel('Flow (m³/s)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS, "tank", f"tank_pso_flood_{evt['name']}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        pd.DataFrame({'observed': obs_plot, 'simulated': sim_plot}).to_csv(
            os.path.join(OUTPUT_DATA, "tank", f"tank_pso_simulation_{evt['name']}.csv"), index=False)
    
    print("完成！")


if __name__ == "__main__":
    run_calibration()
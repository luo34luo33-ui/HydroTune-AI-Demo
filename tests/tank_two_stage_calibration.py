# -*- coding: utf-8 -*-
"""
Tank模型 + 两阶段算法率定
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
from calibration_params import TWO_STAGE_PARAMS
from algos.two_stage import optimize_two_stage
from models.tank import run_tank_model, TANK_PARAM_ORDER, TANK_PARAM_BOUNDS


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
        I = upstream_flow[t]
        I_prev = upstream_flow[t - 1]
        Q_prev = routed[t - 1]
        routed[t] = C0 * I + C1 * I_prev + C2 * Q_prev
    return np.maximum(routed, 0)


def load_flood_events():
    csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
    events = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        
        rename_map = {}
        for std_name, orig_name in COL_MAPPING.items():
            if orig_name and orig_name in df.columns:
                rename_map[orig_name] = std_name
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        if 'precip' not in df.columns or 'flow' not in df.columns:
            continue
        
        if 'evap' not in df.columns:
            df['evap'] = 0.0
        
        precip = df['precip'].fillna(0).values
        evap = df['evap'].fillna(0).values
        flow = df['flow'].fillna(0).values
        upstream = df['upstream'].fillna(0).values if 'upstream' in df.columns else None
        
        events.append({
            'name': fname.replace('.csv', ''),
            'precip': precip,
            'evap': evap,
            'flow': flow,
            'upstream': upstream,
        })
    
    print(f"加载了 {len(events)} 场洪水数据")
    return events


def run_calibration():
    print("=" * 60)
    print("Tank模型 + 两阶段算法 率定")
    print("=" * 60)
    
    events = load_flood_events()
    if not events:
        print("错误：未找到有效数据")
        return
    
    param_names = list(TANK_PARAM_BOUNDS.keys()) + ['k_routing', 'x_routing']
    bounds = list(TANK_PARAM_BOUNDS.values()) + [
        MUSKINGUM_BOUNDS['k_routing'],
        MUSKINGUM_BOUNDS['x_routing']
    ]
    
    def objective(params_array):
        model_params = {k: v for k, v in zip(param_names[:-2], params_array[:-2])}
        k_rout = params_array[-2]
        x_rout = params_array[-1]
        
        try:
            nse_list = []
            for evt in events:
                sim = run_tank_model(evt['precip'], evt['evap'], model_params, CATCHMENT_AREA, 1.0)
                
                if evt['upstream'] is not None and len(evt['upstream']) > 0:
                    routed = muskingum_routing(evt['upstream'], k_rout, x_rout)
                    sim = sim + routed
                
                obs = evt['flow'][WARMUP_STEPS:] if WARMUP_STEPS > 0 else evt['flow']
                sim = sim[WARMUP_STEPS:] if WARMUP_STEPS > 0 else sim
                
                nse = calc_nse(obs, sim)
                if not np.isnan(nse) and not np.isinf(nse):
                    nse_list.append(nse)
            
            avg_nse = np.mean(nse_list) if nse_list else -1e10
            return -avg_nse
        except Exception:
            return 1e10
    
    print("开始率定优化...")
    best_x, best_fun = optimize_two_stage(objective, bounds, MAX_ITERATIONS, len(param_names))
    best_nse = -best_fun
    
    best_params = {k: v for k, v in zip(param_names, best_x)}
    
    print(f"\n率定完成！最优NSE: {best_nse:.4f}")
    print("最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    
    param_df = pd.DataFrame([best_params])
    param_df['NSE'] = best_nse
    param_df.to_csv(os.path.join(OUTPUT_PARAMS, "tank_two_stage_params.csv"), index=False)
    print(f"\n参数已保存到: {OUTPUT_PARAMS}/tank_two_stage_params.csv")
    
    os.makedirs(os.path.join(OUTPUT_PLOTS, "tank"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA, "tank"), exist_ok=True)
    
    print("\n生成各场次洪水过程线...")
    for evt in events:
        sim = run_tank_model(evt['precip'], evt['evap'], best_params, CATCHMENT_AREA, 1.0)
        
        if evt['upstream'] is not None:
            routed = muskingum_routing(evt['upstream'], best_params['k_routing'], best_params['x_routing'])
            sim = sim + routed
        
        obs = evt['flow']
        precip_plot = evt['precip'][WARMUP_STEPS:]
        obs_plot = obs[WARMUP_STEPS:]
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
        
        ax_flow.set_xlabel('Time Step (h)')
        ax_flow.set_ylabel('Flow (m³/s)')
        ax_flow.set_title(f"Tank-TwoStage {evt['name']}")
        ax_flow.legend(loc='upper left')
        ax_flow.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fname = f"tank_two_stage_flood_{evt['name']}.png"
        plt.savefig(os.path.join(OUTPUT_PLOTS, "tank", fname), dpi=150, bbox_inches='tight')
        plt.close()
        
        result_df = pd.DataFrame({
            'time': range(len(sim_plot)),
            'observed': obs_plot,
            'simulated': sim_plot,
        })
        result_df.to_csv(os.path.join(OUTPUT_DATA, "tank", f"tank_two_stage_simulation_{evt['name']}.csv"), index=False)
    
    print("完成！")


if __name__ == "__main__":
    run_calibration()
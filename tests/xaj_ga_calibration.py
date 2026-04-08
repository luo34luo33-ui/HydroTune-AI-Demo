# -*- coding: utf-8 -*-
"""XAJ模型 + GA算法率定"""
import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
sys.path.insert(0, os.path.dirname(__file__))
from calibration_base import DATA_DIR, COL_MAPPING, CATCHMENT_AREA, WARMUP_STEPS, MAX_ITERATIONS, OUTPUT_PARAMS, OUTPUT_PLOTS, OUTPUT_DATA, MUSKINGUM_BOUNDS
from algos.ga import optimize_ga
from models.xaj import run_xaj_model, XAJ_PARAM_BOUNDS
from calibration_params import GA_PARAMS

def calc_nse(o, s):
    m = ~(np.isnan(o) | np.isnan(s))
    if m.sum() == 0: return -9999
    d = np.sum((o[m] - np.mean(o[m]))**2)
    return 1 - np.sum((o[m] - s[m])**2) / d if d > 0 else -9999

def musk(u, k, x):
    n = len(u)
    if n == 0: return np.array([])
    d = k * (1 - x) + 0.5
    C0, C1, C2 = (-k*x+0.5)/d, (k*x+0.5)/d, (k*(1-x)-0.5)/d
    r = np.zeros(n)
    r[0] = u[0]
    for t in range(1, n): r[t] = C0*u[t] + C1*u[t-1] + C2*r[t-1]
    return np.maximum(r, 0)

def load_events():
    events = []
    for f in sorted(glob(os.path.join(DATA_DIR, "*.csv"))):
        try:
            df = pd.read_csv(f)
            df = df.rename(columns={COL_MAPPING[k]: k for k in COL_MAPPING if COL_MAPPING[k] in df.columns})
            if 'precip' not in df.columns or 'flow' not in df.columns: continue
            if 'evap' not in df.columns: df['evap'] = 0.0
            events.append({'name': os.path.basename(f).replace('.csv',''), 'precip': df['precip'].fillna(0).values, 'evap': df['evap'].fillna(0).values, 'flow': df['flow'].fillna(0).values, 'upstream': df['upstream'].fillna(0).values if 'upstream' in df.columns else None})
        except: continue
    return events

def run_calibration():
    print("XAJ+GA")
    events = load_events()
    pnames = list(XAJ_PARAM_BOUNDS.keys()) + ['k_routing', 'x_routing']
    bounds = list(XAJ_PARAM_BOUNDS.values()) + [MUSKINGUM_BOUNDS['k_routing'], MUSKINGUM_BOUNDS['x_routing']]
    
    def obj(p):
        mp = {k:v for k,v in zip(pnames[:-2], p[:-2])}
        try:
            ns = []
            for e in events:
                s = run_xaj_model(e['precip'], e['evap'], mp, CATCHMENT_AREA)
                if e['upstream'] is not None: s += musk(e['upstream'], p[-2], p[-1])
                n = calc_nse(e['flow'][WARMUP_STEPS:], s[WARMUP_STEPS:])
                if not np.isnan(n): ns.append(n)
            return -np.mean(ns) if ns else 1e10
        except: return 1e10
    
    bx, bf = optimize_ga(obj, bounds, MAX_ITERATIONS, len(pnames), GA_PARAMS)
    bp = {k:v for k,v in zip(pnames, bx)}
    print(f"NSE:{-bf:.4f}")
    pd.DataFrame([bp]).to_csv(os.path.join(OUTPUT_PARAMS, "xaj_ga_params.csv"), index=False)
    
    for e in events:
        s = run_xaj_model(e['precip'], e['evap'], bp, CATCHMENT_AREA)
        if e['upstream'] is not None: s += musk(e['upstream'], bp['k_routing'], bp['x_routing'])
        
        precip_plot = e['precip'][WARMUP_STEPS:]
        obs_plot = e['flow'][WARMUP_STEPS:]
        sim_plot = s[WARMUP_STEPS:]
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
        
        param_text = "\n".join([f"{k}: {v:.3f}" for k, v in bp.items()])
        ax_flow.text(0.02, 0.98, param_text, transform=ax_flow.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_flow.text(0.98, 0.98, f"NSE: {nse:.4f}", transform=ax_flow.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax_flow.set_title(f"XAJ-GA {e['name']}")
        ax_flow.legend(loc='upper left')
        ax_flow.grid(True, alpha=0.3)
        ax_flow.set_xlabel('Time Step (h)')
        ax_flow.set_ylabel('Flow (m³/s)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS, "xaj", f"xaj_ga_{e['name']}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        pd.DataFrame({'obs':obs_plot, 'sim':sim_plot}).to_csv(os.path.join(OUTPUT_DATA, "xaj", f"xaj_ga_{e['name']}.csv"), index=False)
    print("完成")

if __name__ == "__main__": run_calibration()
# -*- coding: utf-8 -*-
"""
Tank模型各种率定算法测试
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import glob

sys.path.insert(0, os.path.dirname(__file__))

from src.models.registry import ModelRegistry
from src.hydro_calc import calibrate_model_fast, calc_nse

DATA_DIR = r"E:\HydroTune-AI-Demo\example_data\60场+提前72h换列名"
PARAM_FILE = r"E:\HydroTune-AI-Demo\example_data\params_tank_template.csv"

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

test_file = csv_files[0]
print(f"Testing with: {os.path.basename(test_file)}")

df = pd.read_csv(test_file)

precip = df['avg_rain'].fillna(0).values
flow_arr = df['GZ_in'].values
evap_arr = df['E0'].values

print(f"Precip: sum={precip.sum():.2f}, mean={precip.mean():.2f}")
print(f"Flow: sum={flow_arr.sum():.2f}, mean={flow_arr.mean():.2f}")

param_df = pd.read_csv(PARAM_FILE, encoding='gbk')
tank_params = {}
for col in param_df.columns:
    if col == param_df.columns[0]:
        continue
    if col in ['k_routing', 'x_routing']:
        continue
    try:
        tank_params[col] = float(param_df[col].values[0])
    except:
        pass

print(f"\nTank params: {tank_params}")

model = ModelRegistry.get_model("Tank水箱模型(完整版)")
spatial_data = {'area': 150.0, 'del_t': 24.0}
warmup = 72
k_routing = 5.0
x_routing = 0.13

algorithms = [
    ('two_stage', 'Two-Stage(Dual+L-BFGS)'),
    ('pso', 'PSO Particle Swarm'),
    ('ga', 'Genetic Algorithm'),
    ('sce', 'SCE-UA'),
    ('de', 'Differential Evolution'),
]

print(f"\n{'='*60}")
print(f"Testing calibration algorithms (max_iter=10)")
print(f"{'='*60}")

results = []

for algo_key, algo_name in algorithms:
    print(f"\n--- Testing: {algo_name} ---")
    start_time = time.time()
    
    try:
        result = calibrate_model_fast(
            "Tank水箱模型(完整版)",
            precip,
            evap_arr,
            flow_arr,
            max_iter=10,
            spatial_data=spatial_data,
            timestep='daily',
            algorithm=algo_key,
            algo_params={'n_particles': 20} if algo_key == 'pso' else None,
            upstream_flow=None,
            enable_routing=False,
            calib_events=None,
            warmup_steps=warmup,
            manual_routing_params=None
        )
        
        if result:
            params, nse, simulated = result
            elapsed = time.time() - start_time
            
            print(f"  NSE = {nse:.4f}")
            print(f"  Time = {elapsed:.2f}s")
            print(f"  Params: {params}")
            
            results.append({
                'algorithm': algo_name,
                'nse': nse,
                'time': elapsed,
                'params': params
            })
        else:
            print(f"  [ERROR] Calibration failed")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"Summary")
print(f"{'='*60}")
for r in results:
    print(f"{r['algorithm']:25s} | NSE: {r['nse']:8.4f} | Time: {r['time']:6.2f}s")
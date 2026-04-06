# -*- coding: utf-8 -*-
"""
Tank model debugging script
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.models.registry import ModelRegistry
from src.hydro_calc import calibrate_model_fast, calc_nse

DATA_DIR = r"E:\HydroTune-AI-Demo\example_data\60场+提前72h换列名"
PARAM_FILE = r"E:\HydroTune-AI-Demo\example_data\params_tank_template.csv"

column_mapping = {
    'precip': 'avg_rain',
    'flow': 'GZ_in',
    'evap': 'E0',
    'upstream': 'GB_out'
}

catchment_area = 150.0
warmup_steps = 72
k_routing = 5.0
x_routing = 0.13
enable_routing = True

file_name = "60场+提前72h换列名最新_插值_20100806.csv"
file_path = os.path.join(DATA_DIR, file_name)

print(f"Test file: {file_name}")

for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"[OK] File encoding: {enc}")
        break
    except UnicodeDecodeError:
        continue

print(f"\nData columns: {df.columns.tolist()}")
print(f"Data rows: {len(df)}")

precip_arr = np.array(df[column_mapping['precip']].values, dtype=float)
flow_arr = np.array(df[column_mapping['flow']].values, dtype=float)
evap_arr = np.array(df[column_mapping['evap']].values, dtype=float)
upstream_arr = np.array(df[column_mapping['upstream']].values, dtype=float) if column_mapping['upstream'] in df.columns else None

print(f"\n*** DATA ISSUE DETECTED ***")
print(f"Precip (avg_rain):")
print(f"  Total: {len(precip_arr)}, Valid: {np.sum(~np.isnan(precip_arr))}, NaN: {np.sum(np.isnan(precip_arr))}")
print(f"  Sum: {np.nansum(precip_arr):.2f}, Mean: {np.nanmean(precip_arr):.2f}")
print(f"  First 20 values: {precip_arr[:20]}")

print(f"\nFlow (GZ_in):")
print(f"  Total: {len(flow_arr)}, Valid: {np.sum(~np.isnan(flow_arr))}, NaN: {np.sum(np.isnan(flow_arr))}")
print(f"  Sum: {np.nansum(flow_arr):.2f}, Mean: {np.nanmean(flow_arr):.2f}")

print(f"\nEvap (E0):")
print(f"  Total: {len(evap_arr)}, Valid: {np.sum(~np.isnan(evap_arr))}, NaN: {np.sum(np.isnan(evap_arr))}")
print(f"  Sum: {np.nansum(evap_arr):.2f}, Mean: {np.nanmean(evap_arr):.2f}")

if upstream_arr is not None:
    print(f"\nUpstream (GB_out):")
    print(f"  Total: {len(upstream_arr)}, Valid: {np.sum(~np.isnan(upstream_arr))}, NaN: {np.sum(np.isnan(upstream_arr))}")
    print(f"  Sum: {np.nansum(upstream_arr):.2f}, Mean: {np.nanmean(upstream_arr):.2f}")

print(f"\n=== PROBLEM: Most precip data is NaN! ===")
print(f"This causes model output to be zero/invalid, leading to NSE=-9999")
print(f"\nTrying to use another rain column with more valid data...")

rain_cols = [c for c in df.columns if 'rain' in c.lower()]
best_rain_col = None
best_valid_count = 0
for c in rain_cols:
    valid_count = df[c].notna().sum()
    print(f"  {c}: {valid_count} valid values")
    if valid_count > best_valid_count:
        best_valid_count = valid_count
        best_rain_col = c

print(f"\nBest rain column: {best_rain_col} ({best_valid_count} valid values)")

print(f"\nReading Tank params...")
for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
    try:
        param_df = pd.read_csv(PARAM_FILE, encoding=enc)
        break
    except UnicodeDecodeError:
        continue

print(f"Param columns: {param_df.columns.tolist()}")

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

print(f"Tank params: {tank_params}")

print(f"\nInitializing Tank model...")
model = ModelRegistry.get_model("Tank水箱模型(完整版)")

print(f"\nRunning model with best rain column...")
spatial_data = {'area': catchment_area, 'del_t': 24.0}

best_precip_arr = np.array(df[best_rain_col].values, dtype=float)
best_precip_arr = np.nan_to_num(best_precip_arr, nan=0.0)

print(f"Precip after filling NaN with 0:")
print(f"  Sum: {best_precip_arr.sum():.2f}, Mean: {best_precip_arr.mean():.2f}")

try:
    simulated = model.run(best_precip_arr, evap_arr, tank_params, spatial_data, None, warmup_steps)
    print(f"  Simulated: len={len(simulated)}, sum={simulated.sum():.2f}, mean={simulated.mean():.2f}")
except Exception as e:
    print(f"  [ERROR] Model run failed: {e}")
    import traceback
    traceback.print_exc()
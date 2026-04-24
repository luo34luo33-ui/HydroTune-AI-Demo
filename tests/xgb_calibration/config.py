# -*- coding: utf-8 -*-
"""
全局配置
"""
import os
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs")

COL_MAPPING = {
    'date': 'Time',
    'precip': 'avg_rain',
    'evap': 'E0',
    'flow': 'GZ_in',
    'upstream': 'GB_out',
}

CATCHMENT_AREA = 584.0

WARMUP_STEPS = 72
MAX_ITERATIONS = 10

CALIBRATION_RATIOS = [5, 10, 15]
BENCHMARK_RATIO = 0.75
N_RUNS = 10

RANDOM_SEED = 42

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 300,
}

LSTM_PARAMS = {
    'units': 64,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
}

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_BASE, exist_ok=True)

_current_output_dir = None

def get_run_output_dir(run_name: str = None, update_global: bool = True) -> str:
    """获取本次运行的输出目录
    
    Args:
        run_name: 自定义运行名称（可选），如 'tank_10events'
        update_global: 是否更新全局OUTPUT_DIR变量
        
    Returns:
        输出目录路径，格式: outputs/2024-01-15_14-30-00 或 outputs/tank_10events
    """
    global _current_output_dir
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        _current_output_dir = os.path.join(OUTPUT_BASE, run_name)
    else:
        _current_output_dir = os.path.join(OUTPUT_BASE, timestamp)
    
    if update_global:
        global OUTPUT_DIR
        OUTPUT_DIR = _current_output_dir
    
    os.makedirs(_current_output_dir, exist_ok=True)
    return _current_output_dir

OUTPUT_DIR = get_run_output_dir(update_global=False)

N_ERROR_LAGS = 5
N_PRECIP_LAGS = 3

MIX_WARMUP_STEPS = 3

STEP_LAG_RANGE = [1, 2, 3, 4, 5, 6, 7]

DEBUG_MODE = False
if DEBUG_MODE:
    MAX_ITERATIONS = 5
    N_RUNS = 2
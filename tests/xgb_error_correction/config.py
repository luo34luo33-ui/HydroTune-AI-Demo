# -*- coding: utf-8 -*-
"""
配置文件
"""
import os
import numpy as np

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs")

# 列名映射
COL_MAPPING = {
    'date': 'Time',
    'precip': 'avg_rain',
    'evap': 'E0',
    'flow': 'GZ_in',
    'upstream': 'GB_out',
}

# 流域参数
CATCHMENT_AREA = 584.0  # km²

# 率定参数
WARMUP_STEPS = 72  # 预热期步数(h)
MAX_ITERATIONS = 30  # SCE-UA迭代次数（正常水平）
CALIBRATION_RATIOS = [5, 10, 15]  # 率定场次数
BENCHMARK_RATIO = 0.75  # 75%基准
N_RUNS = 30  # 随机次数（正式运行用10次）

# 随机种子
RANDOM_SEED = 42

# XGBoost参数（默认）
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 300,
}

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "xgb_correction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特征配置
N_ERROR_LAGS = 5  # 误差滞后阶数
N_PRECIP_LAGS = 3  # 降水滞后阶数

# 调试模式（减少迭代次数和运行次数）
DEBUG_MODE = False
if DEBUG_MODE:
    MAX_ITERATIONS = 5
    N_RUNS = 2
    
# 正式运行配置
if not DEBUG_MODE:
    N_RUNS = 30  # 正式运行10次
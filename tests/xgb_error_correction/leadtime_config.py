# -*- coding: utf-8 -*-
"""
预见期退化实验配置
"""
import os
import sys
import numpy as np

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
MAX_ITERATIONS = 30

RANDOM_SEED = 42

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 300,
}

LINEAR_PARAMS = {
    'fit_intercept': True,
}

N_ERROR_LAGS = 5
N_PRECIP_LAGS = 3

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG_MODE = True

if DEBUG_MODE:
    LEADTIMES = [1, 3, 6]
    CORRECTOR_MODELS = ['xgb']
    STRATEGIES = ['oracle', 'free', 'k_window']
    K_VALUES = [0, 3, 6]
    CALIB_RATIOS = [5]
    SEEDS = [0, 1]
    MAX_ITERATIONS = 5
else:
    LEADTIMES = [1, 2, 3, 4, 5, 6, 7]
    CORRECTOR_MODELS = ['xgb', 'lr', 'lstm']
    STRATEGIES = ['oracle', 'free', 'k_window']
    K_VALUES = [0, 1, 2, 3, 5, 7]
    CALIB_RATIOS = [75, 3, 5, 10]
    SEEDS = list(range(10))

if len(sys.argv) > 1:
    DEBUG_MODE = False
    LEADTIMES = [int(x) for x in sys.argv[1].split(',')] if sys.argv[1] != '' else LEADTIMES
    if len(sys.argv) > 2:
        CORRECTOR_MODELS = sys.argv[2].split(',')
    if len(sys.argv) > 3:
        STRATEGIES = sys.argv[3].split(',')
    if len(sys.argv) > 4:
        CALIB_RATIOS = [int(x) for x in sys.argv[4].split(',')]

N_RUNS = len(SEEDS)

TIMEOUT_SECONDS = 300
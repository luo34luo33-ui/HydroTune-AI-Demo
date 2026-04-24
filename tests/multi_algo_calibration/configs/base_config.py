# -*- coding: utf-8 -*-
"""
率定测试基础配置
用于CLI模式下独立运行率定，不依赖项目其他模块
"""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "inputs")

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

MUSKINGUM_BOUNDS = {
    'k_routing': (0.5, 5.0),
    'x_routing': (0.0, 0.5),
}

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUTPUT_PARAMS = os.path.join(OUTPUT_BASE, "params")
OUTPUT_PLOTS = os.path.join(OUTPUT_BASE, "plots")
OUTPUT_DATA = os.path.join(OUTPUT_BASE, "data")

for d in [OUTPUT_PARAMS, OUTPUT_PLOTS, OUTPUT_DATA]:
    os.makedirs(d, exist_ok=True)
for subdir in ["tank", "hbv", "xaj"]:
    os.makedirs(os.path.join(OUTPUT_PLOTS, subdir), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA, subdir), exist_ok=True)
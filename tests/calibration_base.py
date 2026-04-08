# -*- coding: utf-8 -*-
"""
率定测试基础配置
用于CLI模式下独立运行率定，不依赖项目其他模块
"""

import os

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "inputs")

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
MAX_ITERATIONS = 30  # 迭代次数

# 马斯京根参数边界
MUSKINGUM_BOUNDS = {
    'k_routing': (0.5, 5.0),
    'x_routing': (0.0, 0.5),
}

# 输出目录
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT_PARAMS = os.path.join(OUTPUT_BASE, "params")
OUTPUT_PLOTS = os.path.join(OUTPUT_BASE, "plots")
OUTPUT_DATA = os.path.join(OUTPUT_BASE, "data")

# 确保输出目录存在
for d in [OUTPUT_PARAMS, OUTPUT_PLOTS, OUTPUT_DATA]:
    os.makedirs(d, exist_ok=True)
for subdir in ["tank", "hbv", "xaj"]:
    os.makedirs(os.path.join(OUTPUT_PLOTS, subdir), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA, subdir), exist_ok=True)
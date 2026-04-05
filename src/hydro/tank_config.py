# -*- coding: utf-8 -*-
"""
Tank模型参数配置
从 tank-model-structured 子模块迁移
"""

import numpy as np

TANK_PARAMETER_ORDER = [
    't0_is', 't0_boc', 't0_soc_uo', 't0_soc_lo', 't0_soh_uo', 't0_soh_lo',
    't1_is', 't1_boc', 't1_soc', 't1_soh',
    't2_is', 't2_boc', 't2_soc', 't2_soh',
    't3_is', 't3_soc'
]

MUSKINGUM_PARAMETER_ORDER = ['k', 'x']

NUM_PARAMETER = {
    'Subbasin': len(TANK_PARAMETER_ORDER),
    'Reach': len(MUSKINGUM_PARAMETER_ORDER)
}

MODEL_PARAMS = {
    't0_is': {
        'default': 50.0,
        'min': 0.01,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-0 初始蓄水量',
        'tank': 0,
        'param_type': 'initial_storage'
    },
    't0_boc': {
        'default': 0.3,
        'min': 0.1,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-0 底孔出流系数',
        'tank': 0,
        'param_type': 'bottom_outlet'
    },
    't0_soc_lo': {
        'default': 0.3,
        'min': 0.1,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-0 侧孔出流系数（下层）',
        'tank': 0,
        'param_type': 'side_outlet_lower'
    },
    't0_soc_uo': {
        'default': 0.3,
        'min': 0.1,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-0 侧孔出流系数（上层）',
        'tank': 0,
        'param_type': 'side_outlet_upper'
    },
    't0_soh_lo': {
        'default': 25.0,
        'min': 0.0,
        'max': 50.0,
        'unit': 'mm',
        'description': 'Tank-0 侧孔高度（下层）',
        'tank': 0,
        'param_type': 'outlet_height_lower'
    },
    't0_soh_uo': {
        'default': 87.5,
        'min': 75.0,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-0 侧孔高度（上层）',
        'tank': 0,
        'param_type': 'outlet_height_upper'
    },
    't1_is': {
        'default': 50.0,
        'min': 0.01,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-1 初始蓄水量',
        'tank': 1,
        'param_type': 'initial_storage'
    },
    't1_boc': {
        'default': 0.25,
        'min': 0.01,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-1 底孔出流系数',
        'tank': 1,
        'param_type': 'bottom_outlet'
    },
    't1_soc': {
        'default': 0.25,
        'min': 0.01,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-1 侧孔出流系数',
        'tank': 1,
        'param_type': 'side_outlet'
    },
    't1_soh': {
        'default': 50.0,
        'min': 0.0,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-1 侧孔高度',
        'tank': 1,
        'param_type': 'outlet_height'
    },
    't2_is': {
        'default': 50.0,
        'min': 0.01,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-2 初始蓄水量',
        'tank': 2,
        'param_type': 'initial_storage'
    },
    't2_boc': {
        'default': 0.25,
        'min': 0.01,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-2 底孔出流系数',
        'tank': 2,
        'param_type': 'bottom_outlet'
    },
    't2_soc': {
        'default': 0.25,
        'min': 0.01,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-2 侧孔出流系数',
        'tank': 2,
        'param_type': 'side_outlet'
    },
    't2_soh': {
        'default': 50.0,
        'min': 0.0,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-2 侧孔高度',
        'tank': 2,
        'param_type': 'outlet_height'
    },
    't3_is': {
        'default': 50.0,
        'min': 0.01,
        'max': 100.0,
        'unit': 'mm',
        'description': 'Tank-3 初始蓄水量（基流）',
        'tank': 3,
        'param_type': 'initial_storage'
    },
    't3_soc': {
        'default': 0.25,
        'min': 0.01,
        'max': 0.5,
        'unit': '-',
        'description': 'Tank-3 侧孔出流系数（基流）',
        'tank': 3,
        'param_type': 'side_outlet'
    },
}

tank_lb = np.array([MODEL_PARAMS[p]['min'] for p in TANK_PARAMETER_ORDER])
tank_ub = np.array([MODEL_PARAMS[p]['max'] for p in TANK_PARAMETER_ORDER])
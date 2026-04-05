# -*- coding: utf-8 -*-
"""
默认参数配置 - 所有模型的默认参数值统一管理
"""
from typing import Dict

# ===== Tank模型默认参数 =====
TANK_DEFAULT_PARAMS: Dict[str, float] = {
    't0_is': 50.0,
    't0_boc': 0.3,
    't0_soc_uo': 0.3,
    't0_soc_lo': 0.3,
    't0_soh_uo': 87.5,
    't0_soh_lo': 25.0,
    't1_is': 50.0,
    't1_boc': 0.25,
    't1_soc': 0.25,
    't1_soh': 50.0,
    't2_is': 50.0,
    't2_boc': 0.25,
    't2_soc': 0.25,
    't2_soh': 50.0,
    't3_is': 50.0,
    't3_soc': 0.25,
}

# ===== HBV模型默认参数 =====
HBV_DEFAULT_PARAMS: Dict[str, float] = {
    'dd': 6.10,
    'fc': 195.0,
    'beta': 2.6143,
    'c': 0.07,
    'k0': 0.163,
    'l': 4.87,
    'k1': 0.029,
    'k2': 0.049,
    'kp': 0.050,
    'pwp': 106.0,
}

# ===== XAJ模型默认参数 =====
XAJ_DEFAULT_PARAMS: Dict[str, float] = {
    'k': 0.8,
    'b': 0.3,
    'im': 0.01,
    'um': 20.0,
    'lm': 70.0,
    'dm': 60.0,
    'c': 0.15,
    'sm': 20.0,
    'ex': 1.5,
    'ki': 0.3,
    'kg': 0.4,
    'cs': 0.8,
    'l': 1,
    'ci': 0.8,
    'cg': 0.98,
}

# ===== XAJ V3模型默认参数 (新核心) =====
XAJ_V3_DEFAULT_PARAMS: Dict[str, float] = {
    'B': 0.3,
    'C': 0.2,
    'WM': 150.0,
    'WUM': 23.87,
    'WLM': 60.0,
    'IM': 0.02,
    'SM': 30.92,
    'EX': 1.12,
    'K': 1.2,
    'KG': 0.37,
    'KI': 0.2,
    'CG': 0.998,
    'CI': 0.85,
    'CS': 0.72,
    'L': 1,
    'X': 0.27,
    'K_res': 4.88,
    'X_res': 0.14,
    'n': 5,
    'Area': 584.0,
}

# ===== 马斯京根演算默认参数 =====
ROUTING_DEFAULT_PARAMS: Dict[str, float] = {
    'k_routing': 2.5,
    'x_routing': 0.25,
}


def get_default_params(model_name: str) -> Dict[str, float]:
    """根据模型名称获取默认参数"""
    if 'Tank' in model_name:
        return TANK_DEFAULT_PARAMS.copy()
    elif 'HBV' in model_name:
        return HBV_DEFAULT_PARAMS.copy()
    elif 'V3' in model_name:
        return XAJ_V3_DEFAULT_PARAMS.copy()
    elif '新安江' in model_name or 'XAJ' in model_name:
        return XAJ_DEFAULT_PARAMS.copy()
    else:
        return {}
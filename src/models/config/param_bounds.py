# -*- coding: utf-8 -*-
"""
参数边界配置 - 所有模型的参数取值范围统一管理
"""
from typing import Dict, Tuple

# ===== Tank模型参数边界 =====
TANK_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    't0_is': (0.01, 100.0),
    't0_boc': (0.1, 0.5),
    't0_soc_uo': (0.1, 0.5),
    't0_soc_lo': (0.1, 0.5),
    't0_soh_uo': (75.0, 100.0),
    't0_soh_lo': (0.0, 50.0),
    't1_is': (0.01, 100.0),
    't1_boc': (0.01, 0.5),
    't1_soc': (0.01, 0.5),
    't1_soh': (0.0, 100.0),
    't2_is': (0.01, 100.0),
    't2_boc': (0.01, 0.5),
    't2_soc': (0.01, 0.5),
    't2_soh': (0.0, 100.0),
    't3_is': (0.01, 100.0),
    't3_soc': (0.01, 0.5),
}

# ===== HBV模型参数边界 =====
HBV_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'dd': (3.0, 7.0),
    'fc': (100.0, 200.0),
    'beta': (1.0, 7.0),
    'c': (0.01, 0.07),
    'k0': (0.05, 0.2),
    'l': (2.0, 5.0),
    'k1': (0.01, 0.1),
    'k2': (0.01, 0.05),
    'kp': (0.01, 0.05),
    'pwp': (90.0, 180.0),
}

# ===== XAJ模型参数边界 =====
XAJ_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'k': (0.5, 1.5),
    'b': (0.1, 0.8),
    'im': (0.0, 0.15),
    'um': (10.0, 60.0),
    'lm': (50.0, 150.0),
    'dm': (40.0, 120.0),
    'c': (0.1, 0.5),
    'sm': (10.0, 80.0),
    'ex': (1.0, 3.0),
    'ki': (0.05, 0.45),
    'kg': (0.05, 0.45),
    'cs': (0.5, 0.98),
    'l': (0, 20),
    'ci': (0.5, 0.98),
    'cg': (0.9, 0.999),
}

# ===== XAJ V3模型参数边界 (新核心) =====
XAJ_V3_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'B': (0.1, 0.5),
    'C': (0.1, 0.3),
    'WM': (100.0, 200.0),
    'WUM': (10.0, 40.0),
    'WLM': (40.0, 100.0),
    'IM': (0.0, 0.1),
    'SM': (10.0, 80.0),
    'EX': (1.0, 2.0),
    'K': (0.8, 1.5),
    'KG': (0.2, 0.5),
    'KI': (0.1, 0.3),
    'CG': (0.90, 0.999),
    'CI': (0.7, 0.95),
    'CS': (0.15, 0.85),
    'L': (0, 5),
    'X': (0.0, 0.5),
    'K_res': (1.0, 8.0),
    'X_res': (0.0, 0.5),
    'n': (3, 6)
}

# ===== 马斯京根演算参数边界 =====
ROUTING_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'k_routing': (0.5, 5.0),
    'x_routing': (0.0, 0.5),
}


def get_param_bounds(model_name: str) -> Dict[str, Tuple[float, float]]:
    """根据模型名称获取参数边界"""
    if 'Tank' in model_name:
        return TANK_PARAM_BOUNDS.copy()
    elif 'HBV' in model_name:
        return HBV_PARAM_BOUNDS.copy()
    elif 'V3' in model_name:
        return XAJ_V3_PARAM_BOUNDS.copy()
    elif '新安江' in model_name or 'XAJ' in model_name:
        return XAJ_PARAM_BOUNDS.copy()
    else:
        return {}
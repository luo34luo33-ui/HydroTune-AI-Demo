# -*- coding: utf-8 -*-
"""
HBV模型
独立实现，不依赖项目其他模块
"""
import numpy as np
from typing import Dict

HBV_PARAM_BOUNDS = {
    'fc': (100.0, 200.0),
    'beta': (1.0, 7.0),
    'c': (0.01, 0.07),
    'k0': (0.05, 0.2),
    'l': (2.0, 5.0),
    'k1': (0.01, 0.1),
    'k2': (0.01, 0.05),
    'kp': (0.01, 0.05),
    'lp': (0.3, 1.0),
}

HBV_PARAM_ORDER = [
    'fc', 'beta', 'c', 'k0', 'l', 'k1', 'k2', 'kp', 'lp'
]


def run_hbv_model(
    precip: np.ndarray,
    evap: np.ndarray,
    params: Dict[str, float],
    area: float = 584.0,
) -> np.ndarray:
    """
    运行HBV模型
    
    Args:
        precip: 降水序列 (mm)
        evap: 蒸发序列 (mm)
        params: 参数字典
        area: 流域面积 (km²)
        
    Returns:
        模拟流量序列 (m³/s)
    """
    p = params
    fc = p.get('fc', 195.0)
    beta = p.get('beta', 2.6)
    c = p.get('c', 0.07)
    k0 = p.get('k0', 0.163)
    l = p.get('l', 4.87)
    k1 = p.get('k1', 0.029)
    k2 = p.get('k2', 0.049)
    kp = p.get('kp', 0.050)
    lp = p.get('lp', 0.5)
    
    pwp = lp * fc
    
    n = len(precip)
    if n == 0:
        return np.array([])
    
    sm = np.zeros(n)
    suz = np.zeros(n)
    slz = np.zeros(n)
    q0_arr = np.zeros(n)
    q1_arr = np.zeros(n)
    q2_arr = np.zeros(n)
    
    pet = evap * c
    for t in range(n):
        prev_sm = sm[t - 1] if t > 0 else 0
        ae = pet[t] if prev_sm > pwp else pet[t] * max(sm[t - 1], 0) / max(pwp, 1e-10)
        ae = max(ae, 0)
        
        ratio = max(sm[t - 1], 0) / max(fc, 1e-10)
        ratio = min(ratio, 1.0)
        effective = precip[t] * (ratio ** beta)
        
        sm[t] = max(sm[t - 1] + precip[t] - ae - effective, 0)
        
        recharge = effective * (1 - ((fc - sm[t]) / fc) ** 2)
        recharge = max(recharge, 0)
        
        q0_arr[t] = k0 * max(sm[t] - l, 0)
        q1_arr[t] = k1 * max(suz[t - 1], 0) if t > 0 else 0
        q2_arr[t] = k2 * max(slz[t - 1], 0) if t > 0 else 0
        
        uz_exchange = kp * (slz[t - 1] - suz[t - 1]) if t > 0 else 0
        
        suz[t] = max(suz[t - 1] + recharge - q1_arr[t] + uz_exchange, 0) if t > 0 else max(recharge, 0)
        slz[t] = max(slz[t - 1] - q2_arr[t] - uz_exchange, 0) if t > 0 else 0
    
    unit_conv = (area * 1000) / 86400
    discharge = unit_conv * (q0_arr + q1_arr + q2_arr)
    
    return np.maximum(discharge, 0)
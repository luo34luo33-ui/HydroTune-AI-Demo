# -*- coding: utf-8 -*-
"""
Tank水箱模型
独立实现，不依赖项目其他模块
"""
import numpy as np
from typing import Dict, Tuple

TANK_PARAM_BOUNDS = {
    't0_is': (0.0, 50.0),
    't0_boc': (0.15, 0.5),
    't0_soc_uo': (0.2, 0.6),
    't0_soc_lo': (0.15, 0.5),
    't0_soh_uo': (50.0, 120.0),
    't0_soh_lo': (10.0, 50.0),
    't1_is': (0.0, 50.0),
    't1_boc': (0.1, 0.4),
    't1_soc': (0.1, 0.4),
    't1_soh': (20.0, 80.0),
    't2_is': (0.0, 50.0),
    't2_boc': (0.05, 0.3),
    't2_soc': (0.05, 0.3),
    't2_soh': (10.0, 60.0),
    't3_is': (0.0, 50.0),
    't3_soc': (0.001, 0.05),
}

TANK_PARAM_ORDER = [
    't0_is', 't0_boc', 't0_soc_uo', 't0_soc_lo', 't0_soh_uo', 't0_soh_lo',
    't1_is', 't1_boc', 't1_soc', 't1_soh',
    't2_is', 't2_boc', 't2_soc', 't2_soh',
    't3_is', 't3_soc'
]


def run_tank_model(
    precip: np.ndarray,
    evap: np.ndarray,
    params: Dict[str, float],
    area: float = 584.0,
    del_t: float = 1.0,
) -> np.ndarray:
    """
    运行Tank水箱模型
    
    Args:
        precip: 降水序列 (mm)
        evap: 蒸发序列 (mm)
        params: 参数字典
        area: 流域面积 (km²)
        del_t: 时间步长 (小时)
        
    Returns:
        模拟流量序列 (m³/s)
    """
    p = params
    t0_is = p.get('t0_is', 10.0)
    t0_boc = p.get('t0_boc', 0.3)
    t0_soc_uo = p.get('t0_soc_uo', 0.4)
    t0_soc_lo = p.get('t0_soc_lo', 0.3)
    t0_soh_uo = p.get('t0_soh_uo', 80.0)
    t0_soh_lo = p.get('t0_soh_lo', 30.0)
    
    t1_is = p.get('t1_is', 10.0)
    t1_boc = p.get('t1_boc', 0.25)
    t1_soc = p.get('t1_soc', 0.25)
    t1_soh = p.get('t1_soh', 50.0)
    
    t2_is = p.get('t2_is', 10.0)
    t2_boc = p.get('t2_boc', 0.15)
    t2_soc = p.get('t2_soc', 0.15)
    t2_soh = p.get('t2_soh', 35.0)
    
    t3_is = p.get('t3_is', 10.0)
    t3_soc = p.get('t3_soc', 0.02)
    
    n = len(precip)
    if n == 0:
        return np.array([])
    
    tank_storage = np.zeros((n, 4))
    side_outlet_flow = np.zeros((n, 4))
    bottom_outlet_flow = np.zeros((n, 3))
    
    del_rf_et = precip - evap
    
    tank_storage[0, 0] = max(t0_is, 0)
    tank_storage[0, 1] = max(t1_is, 0)
    tank_storage[0, 2] = max(t2_is, 0)
    tank_storage[0, 3] = max(t3_is, 0)
    
    for t in range(n):
        side_outlet_flow[t, 0] = (
            t0_soc_lo * max(tank_storage[t, 0] - t0_soh_lo, 0) +
            t0_soc_uo * max(tank_storage[t, 0] - t0_soh_uo, 0)
        )
        
        side_outlet_flow[t, 1] = t1_soc * max(tank_storage[t, 1] - t1_soh, 0)
        side_outlet_flow[t, 2] = t2_soc * max(tank_storage[t, 2] - t2_soh, 0)
        side_outlet_flow[t, 3] = t3_soc * tank_storage[t, 3]
        
        bottom_outlet_flow[t, 0] = t0_boc * tank_storage[t, 0]
        bottom_outlet_flow[t, 1] = t1_boc * tank_storage[t, 1]
        bottom_outlet_flow[t, 2] = t2_boc * tank_storage[t, 2]
        
        if t < n - 1:
            tank_storage[t+1, 0] = (
                tank_storage[t, 0] + del_rf_et[t+1] -
                (side_outlet_flow[t, 0] + bottom_outlet_flow[t, 0])
            )
            tank_storage[t+1, 1] = (
                tank_storage[t, 1] + bottom_outlet_flow[t, 0] -
                (side_outlet_flow[t, 1] + bottom_outlet_flow[t, 1])
            )
            tank_storage[t+1, 2] = (
                tank_storage[t, 2] + bottom_outlet_flow[t, 1] -
                (side_outlet_flow[t, 2] + bottom_outlet_flow[t, 2])
            )
            tank_storage[t+1, 3] = (
                tank_storage[t, 3] + bottom_outlet_flow[t, 2] -
                side_outlet_flow[t, 3]
            )
            
            tank_storage[t+1, 0] = max(tank_storage[t+1, 0], 0)
            tank_storage[t+1, 1] = max(tank_storage[t+1, 1], 0)
            tank_storage[t+1, 2] = max(tank_storage[t+1, 2], 0)
            tank_storage[t+1, 3] = max(tank_storage[t+1, 3], 0)
    
    unit_conv = (area * 1000) / (del_t * 3600)
    discharge = unit_conv * side_outlet_flow.sum(axis=1)
    
    return np.maximum(discharge, 0)
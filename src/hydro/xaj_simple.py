# -*- coding: utf-8 -*-
"""
新安江模型 (XAJ) 简化实现
从tests/models/xaj.py迁移，独立于tests目录
"""
import numpy as np
from typing import Dict

XAJ_PARAM_BOUNDS = {
    'k': (0.7, 1.3),
    'b': (0.1, 0.5),
    'im': (0.001, 0.1),
    'um': (10.0, 60.0),
    'lm': (50.0, 150.0),
    'dm': (40.0, 120.0),
    'c': (0.1, 0.5),
    'sm': (10.0, 80.0),
    'ex': (1.0, 2.0),
    'ki': (0.1, 0.5),
    'kg': (0.1, 0.5),
    'cs': (0.5, 0.98),
    'l': (0, 20),
    'ci': (0.5, 0.98),
    'cg': (0.98, 0.999),
}

XAJ_PARAM_ORDER = [
    'k', 'b', 'im', 'um', 'lm', 'dm', 'c',
    'sm', 'ex', 'ki', 'kg', 'cs', 'l', 'ci', 'cg'
]


def _constrain_ki_kg(ki: float, kg: float) -> tuple:
    if ki + kg >= 1.0:
        total = ki + kg
        ki = ki / total * 0.99
        kg = kg / total * 0.99
    return ki, kg


def run_xaj_model(
    precip: np.ndarray,
    evap: np.ndarray,
    params: Dict[str, float],
    area: float = 584.0,
) -> np.ndarray:
    """
    运行新安江模型
    
    Args:
        precip: 降水序列 (mm)
        evap: 蒸发序列 (mm)
        params: 参数字典
        area: 流域面积 (km²)
        
    Returns:
        模拟流量序列 (m³/s)
    """
    p = params
    k = p.get('k', 1.0)
    b = p.get('b', 0.3)
    im = p.get('im', 0.01)
    um = p.get('um', 20.0)
    lm = p.get('lm', 70.0)
    dm = p.get('dm', 60.0)
    c = p.get('c', 0.15)
    sm = p.get('sm', 20.0)
    ex = p.get('ex', 1.5)
    ki = p.get('ki', 0.3)
    kg = p.get('kg', 0.4)
    ki, kg = _constrain_ki_kg(ki, kg)
    cs = p.get('cs', 0.8)
    l = int(p.get('l', 1))
    ci = p.get('ci', 0.8)
    cg = p.get('cg', 0.995)
    
    n = len(precip)
    if n == 0:
        return np.array([])
    
    wm = um + lm + dm
    wum = um
    wlm = lm
    wdm = dm
    
    eu = np.zeros(n)
    el = np.zeros(n)
    ed = np.zeros(n)
    
    wu = np.zeros(n)
    wl = np.zeros(n)
    wd = np.zeros(n)
    iu = np.zeros(n)
    
    sm1 = np.zeros(n)
    sm2 = np.zeros(n)
    sm3 = np.zeros(n)
    
    qi = np.zeros(n)
    qg = np.zeros(n)
    
    for t in range(n):
        pe = precip[t] - evap[t] * k
        
        if pe <= 0:
            if t > 0:
                eum = min(evap[t] * k, wu[t - 1] + iu[t - 1])
            else:
                eum = min(evap[t] * k, wum)
            
            eu[t] = eum
            el[t] = 0
            ed[t] = 0
            
            iuo = iu[t - 1] if t > 0 else wum
            wu[t] = max(wu[t - 1] - eu[t], 0) if t > 0 else max(wum - eu[t], 0)
            iu[t] = max(iuo - eu[t], 0)
            wl[t] = wl[t - 1] if t > 0 else wlm
            wd[t] = wd[t - 1] if t > 0 else wdm
        else:
            iuo = iu[t - 1] if t > 0 else wum
            iu[t] = iuo + pe
            
            if iu[t] <= wum:
                eu[t] = min(evap[t] * k, wu[t - 1] if t > 0 else wum)
                wu[t] = max(wu[t - 1] - eu[t], 0) if t > 0 else max(wum - eu[t], 0)
                el[t] = 0
                ed[t] = 0
            else:
                eu[t] = evap[t] * k
                wu[t] = max(iu[t] - wum, 0)
                iu[t] = wum
                
                if wl[t - 1] if t > 0 else wlm > c * (wl[t - 1] if t > 0 else wlm):
                    eel = evap[t] * k * c
                else:
                    eel = evap[t] * k
                
                if (wl[t - 1] if t > 0 else wlm) > c * (wl[t - 1] if t > 0 else wlm):
                    el[t] = min(eel, wl[t - 1] - c * (wl[t - 1] if t > 0 else wlm)) if t > 0 else min(eel, wlm - c * wlm)
                else:
                    el[t] = 0
                
                wl[t] = max((wl[t - 1] if t > 0 else wlm) - el[t], 0)
                
                if (wd[t - 1] if t > 0 else wdm) > c * (wd[t - 1] if t > 0 else wdm):
                    ed[t] = min(eel * (wd[t - 1] / (c * (wd[t - 1] if t > 0 else wdm))), wd[t - 1] - c * (wd[t - 1] if t > 0 else wdm)) if t > 0 else 0
                else:
                    ed[t] = 0
                
                wd[t] = max((wd[t - 1] if t > 0 else wdm) - ed[t], 0)
        
        sm1[t] = sm * (1 + ex)
        
        if t == 0:
            sm1[0] = sm * 0.5
            sm2[0] = sm * 0.3
            sm3[0] = sm * 0.2
        
        fr = (sm1[t] / sm) ** ex if sm > 0 else 0
        
        if iu[t] > wum:
            r = (iu[t] - wum) ** 2 / (iu[t] - wum + sm1[t])
        else:
            r = 0
        
        qs0 = r / (r + sm) if sm > 0 else 0
        
        if fr > 0:
            qs = fr * qs0
            qss = (1 - fr) * qs0
        else:
            qs = 0
            qss = 0
        
        qi[t] = qs * ki * sm
        qg[t] = qss * kg * sm
        
        if t > l:
            qg[t] = qg[t - l]
            qi[t] = qi[t - l]
    
    unit_conv = (area * 1000) / 3600
    discharge = unit_conv * (qi + qg)
    
    return np.maximum(discharge, 0)
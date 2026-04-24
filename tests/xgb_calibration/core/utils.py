# -*- coding: utf-8 -*-
"""
工具函数
"""
import numpy as np


def musk(u: np.ndarray, k: float, x: float, dt: float = 1.0) -> np.ndarray:
    """马斯京根汇流方法"""
    n = len(u)
    if n == 0:
        return np.array([])
    
    d = k * (1 - x) + 0.5 * dt
    if abs(d) < 1e-12:
        return u
    
    C0 = (-k * x + 0.5 * dt) / d
    C1 = (k * x + 0.5 * dt) / d
    C2 = (k * (1 - x) - 0.5 * dt) / d
    
    r = np.zeros(n)
    r[0] = u[0]
    for t in range(1, n):
        r[t] = C0 * u[t] + C1 * u[t - 1] + C2 * r[t - 1]
    
    return np.maximum(r, 0)
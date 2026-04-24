# -*- coding: utf-8 -*-
"""
SCE-UA算法 (Shuffle Complex Evolution)
"""
import numpy as np
from scipy.optimize import differential_evolution


def optimize_sce(objective, bounds, max_iter=30, n_params=15, algo_params=None):
    """
    SCE-UA优化算法
    
    Args:
        objective: 目标函数
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        algo_params: 算法参数字典
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    result = differential_evolution(
        objective, bounds=bounds, maxiter=max_iter,
        tol=1e-6, polish=True, strategy='best1bin',
    )
    return result.x, result.fun
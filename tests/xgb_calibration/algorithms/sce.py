# -*- coding: utf-8 -*-
"""
SCE-UA算法
"""
import numpy as np
from scipy.optimize import differential_evolution


def optimize_sce(objective, bounds, max_iter=30, n_params=15, algo_params=None):
    """SCE-UA优化算法"""
    result = differential_evolution(
        objective, bounds=bounds, maxiter=max_iter,
        tol=1e-6, polish=True, strategy='best1bin',
    )
    return result.x, result.fun
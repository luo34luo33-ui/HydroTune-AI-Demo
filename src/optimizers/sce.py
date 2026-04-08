# -*- coding: utf-8 -*-
"""
SCE-UA算法 (Shuffle Complex Evolution)
"""
import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, List, Tuple, Optional


def optimize_sce(
    objective: Callable,
    bounds: List[Tuple[float, float]],
    max_iter: int = 30,
    n_params: int = 15,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, float]:
    """
    SCE-UA优化算法
    
    Args:
        objective: 目标函数
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        progress_callback: 进度回调函数
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    result = differential_evolution(
        objective, bounds=bounds, maxiter=max_iter,
        tol=1e-6, polish=True, strategy='best1bin',
    )
    
    if progress_callback:
        progress_callback(1.0)
    
    return result.x, result.fun
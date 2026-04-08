# -*- coding: utf-8 -*-
"""
两阶段优化算法
阶段1: 模拟退火 (dual_annealing)
阶段2: L-BFGS-B 局部优化
"""
import numpy as np
from scipy.optimize import minimize, dual_annealing
from typing import Callable, List, Tuple, Optional


def optimize_two_stage(
    objective: Callable,
    bounds: List[Tuple[float, float]],
    max_iter: int = 30,
    n_params: int = 15,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, float]:
    """
    两阶段优化算法
    
    Args:
        objective: 目标函数
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        progress_callback: 进度回调函数
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    if n_params <= 5:
        stage1_iter = max(5, max_iter // 2)
        stage2_iter = max(20, max_iter * 3)
    elif n_params <= 10:
        stage1_iter = max(8, max_iter)
        stage2_iter = max(30, max_iter * 2)
    else:
        stage1_iter = max(5, max_iter // 2)
        stage2_iter = max(15, max_iter)
    
    total_iter = stage1_iter + stage2_iter
    current_iter = [0]
    
    def update_progress_da(x, e, context):
        if progress_callback:
            current_iter[0] += 1
            progress = min(current_iter[0] / total_iter, 1.0)
            progress_callback(progress)
        return False
    
    def update_progress_min(xk):
        if progress_callback:
            current_iter[0] += 1
            progress = min(current_iter[0] / total_iter, 1.0)
            progress_callback(progress)
    
    np.random.seed(42)
    
    result1 = dual_annealing(
        objective, bounds=bounds, maxiter=stage1_iter,
        initial_temp=5230, restart_temp_ratio=1e-4,
        visit=2.62, accept=-5.0, no_local_search=True,
        callback=update_progress_da if progress_callback else None,
    )
    best_x = result1.x
    best_fun = result1.fun
    
    current_iter[0] = stage1_iter
    
    result2 = minimize(
        objective, x0=best_x, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': stage2_iter, 'ftol': 1e-7},
        callback=update_progress_min if progress_callback else None,
    )
    
    if result2.fun <= best_fun:
        return result2.x, result2.fun
    return best_x, best_fun
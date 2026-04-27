# -*- coding: utf-8 -*-
"""
SCE-UA算法 (Shuffle Complex Evolution - University of Arizona)
参考: Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization.
"""
import numpy as np
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
        objective: 目标函数（最小化）
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        progress_callback: 进度回调函数
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    n_params = len(bounds)
    bounds_arr = np.array(bounds)
    
    n_complexes = 5
    points_per_complex = 2 * n_params
    n_points = n_complexes * points_per_complex
    
    max_loop = max_iter
    pgs = 1
    max_no_improve = max_iter + 1
    
    sf = 0.1
    cf = 1.0
    
    population = _initialize_population(n_points, bounds_arr)
    objective_values = np.array([objective(x) for x in population])
    
    sorted_idx = np.argsort(objective_values)
    population = population[sorted_idx]
    objective_values = objective_values[sorted_idx]
    
    best_x = population[0].copy()
    best_f = objective_values[0]
    
    loop_count = 0
    no_improve_count = 0
    
    while loop_count < max_loop:
        complexes = []
        complex_obj_values = []
        
        for ic in range(n_complexes):
            start_idx = ic * points_per_complex
            end_idx = start_idx + points_per_complex
            complexes.append(population[start_idx:end_idx].copy())
            complex_obj_values.append(objective_values[start_idx:end_idx].copy())
        
        for ic in range(n_complexes):
            complex_pts = complexes[ic]
            complex_vals = complex_obj_values[ic]
            
            for _ in range(pgs):
                if len(complex_pts) < 3:
                    break
                    
                idx_worst = np.argmax(complex_vals)
                worst_pt = complex_pts[idx_worst]
                worst_val = complex_vals[idx_worst]
                
                idx1, idx2, idx3 = _select_three_indices(len(complex_pts), idx_worst)
                pt1, pt2, pt3 = complex_pts[idx1], complex_pts[idx2], complex_pts[idx3]
                
                new_pt = pt1 + cf * (pt2 - pt3)
                new_pt = np.clip(new_pt, bounds_arr[:, 0], bounds_arr[:, 1])
                
                new_val = objective(new_pt)
                
                if new_val < worst_val:
                    complex_pts[idx_worst] = new_pt
                    complex_vals[idx_worst] = new_val
                    
                    if new_val < best_f:
                        best_x = new_pt.copy()
                        best_f = new_val
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    new_pt = (worst_pt + best_x) / 2
                    new_pt = np.clip(new_pt, bounds_arr[:, 0], bounds_arr[:, 1])
                    new_val = objective(new_pt)
                    
                    if new_val < worst_val:
                        complex_pts[idx_worst] = new_pt
                        complex_vals[idx_worst] = new_val
                        
                        if new_val < best_f:
                            best_x = new_pt.copy()
                            best_f = new_val
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        new_pt = _random_point(bounds_arr)
                        new_val = objective(new_pt)
                        complex_pts[idx_worst] = new_pt
                        complex_vals[idx_worst] = new_val
                        no_improve_count += 1
            
            start_idx = ic * points_per_complex
            end_idx = start_idx + points_per_complex
            population[start_idx:end_idx] = complex_pts
            objective_values[start_idx:end_idx] = complex_vals
        
        sorted_idx = np.argsort(objective_values)
        population = population[sorted_idx]
        objective_values = objective_values[sorted_idx]
        
        loop_count += 1
        
        if progress_callback:
            progress_callback(loop_count / max_loop)
    
    return best_x, best_f


def _initialize_population(n_points: int, bounds: np.ndarray) -> np.ndarray:
    """初始化种群"""
    n_params = len(bounds)
    population = np.zeros((n_points, n_params))
    for i in range(n_params):
        population[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], n_points)
    return population


def _select_three_indices(n: int, exclude: int) -> Tuple[int, int, int]:
    """选择三个不同的索引（排除指定的索引）"""
    available = [i for i in range(n) if i != exclude]
    if len(available) < 3:
        return 0, 1, 2 if n > 2 else (0, 0, 0)
    
    indices = np.random.choice(available, 3, replace=False)
    return indices[0], indices[1], indices[2]


def _random_point(bounds: np.ndarray) -> np.ndarray:
    """在给定边界内生成随机点"""
    n_params = len(bounds)
    return np.array([
        np.random.uniform(bounds[i, 0], bounds[i, 1])
        for i in range(n_params)
    ])
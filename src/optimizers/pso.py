# -*- coding: utf-8 -*-
"""
粒子群优化算法 (PSO)
"""
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict


def optimize_pso(
    objective: Callable,
    bounds: List[Tuple[float, float]],
    max_iter: int = 30,
    n_params: int = 15,
    algo_params: Optional[Dict] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, float]:
    """
    粒子群优化算法
    
    Args:
        objective: 目标函数
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        algo_params: 算法参数字典 {'n_particles': int, 'w': float, 'c1': float, 'c2': float}
        progress_callback: 进度回调函数
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    if algo_params is None:
        algo_params = {}
    
    n_particles = algo_params.get('n_particles', 100)
    w = algo_params.get('w', 0.7)
    c1 = algo_params.get('c1', 1.5)
    c2 = algo_params.get('c2', 1.5)
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    np.random.seed(42)
    particles = np.random.uniform(lb, ub, (n_particles, n_params))
    velocities = np.zeros((n_particles, n_params))
    
    fitness = np.array([objective(p) for p in particles])
    
    best_idx = np.argmin(fitness)
    best_position = particles[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    personal_best_pos = particles.copy()
    personal_best_fitness = fitness.copy()
    
    for i in range(max_iter):
        r1, r2 = np.random.random((n_particles, n_params)), np.random.random((n_particles, n_params))
        
        velocities = (w * velocities +
                      c1 * r1 * (personal_best_pos - particles) +
                      c2 * r2 * (best_position - particles))
        
        velocities = np.clip(velocities, lb - ub, ub - lb) * 0.5
        
        particles = particles + velocities
        particles = np.clip(particles, lb, ub)
        
        fitness = np.array([objective(p) for p in particles])
        
        improved = fitness < personal_best_fitness
        personal_best_fitness[improved] = fitness[improved]
        personal_best_pos[improved] = particles[improved]
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_position = particles[current_best_idx].copy()
        
        if progress_callback:
            progress_callback((i + 1) / max_iter)
    
    return best_position, best_fitness
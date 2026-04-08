# -*- coding: utf-8 -*-
"""
差分进化算法 (DE)
"""
import numpy as np
from scipy.optimize import differential_evolution


def optimize_de(objective, bounds, max_iter=30, n_params=15, algo_params=None):
    """
    差分进化优化算法
    
    Args:
        objective: 目标函数
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        algo_params: 算法参数字典 {'mutation_factor': float, 'crossover_prob': float, 'pop_size': int}
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    if algo_params is None:
        algo_params = {}
    
    mutation_factor = algo_params.get('mutation_factor', 0.8)
    crossover_prob = algo_params.get('crossover_prob', 0.7)
    pop_size = algo_params.get('pop_size', 20)
    
    result = differential_evolution(
        objective, bounds=bounds, maxiter=max_iter, tol=1e-6, polish=True,
        strategy='best1bin', mutation=mutation_factor, recombination=crossover_prob,
        popsize=pop_size,
    )
    return result.x, result.fun
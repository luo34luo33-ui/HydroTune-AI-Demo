# -*- coding: utf-8 -*-
"""
遗传算法 (GA)
"""
import numpy as np


def optimize_ga(objective, bounds, max_iter=30, n_params=15, algo_params=None):
    """
    遗传算法优化
    
    Args:
        objective: 目标函数
        bounds: 参数边界 list[(min, max), ...]
        max_iter: 最大迭代次数
        n_params: 参数数量
        algo_params: 算法参数字典 {'pop_size': int, 'crossover_rate': float, 'mutation_rate': float}
        
    Returns:
        (最优参数数组, 最优目标函数值)
    """
    if algo_params is None:
        algo_params = {}
    
    pop_size = algo_params.get('pop_size', 20)
    crossover_rate = algo_params.get('crossover_rate', 0.8)
    mutation_rate = algo_params.get('mutation_rate', 0.1)
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    np.random.seed(42)
    
    def create_individual():
        return np.random.uniform(lb, ub, n_params)
    
    pop = np.array([create_individual() for _ in range(pop_size)])
    fitness = np.array([objective(p) for p in pop])
    
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_position = pop[best_idx].copy()
    
    for _ in range(max_iter):
        new_pop = []
        for _ in range(pop_size):
            parent1_idx, parent2_idx = np.random.choice(pop_size, 2, replace=False)
            parent1, parent2 = pop[parent1_idx], pop[parent2_idx]
            
            child = parent1.copy()
            if np.random.random() < crossover_rate:
                crossover_point = np.random.randint(1, n_params)
                child[crossover_point:] = parent2[crossover_point:]
            
            if np.random.random() < mutation_rate:
                for i in range(n_params):
                    if np.random.random() < 0.5:
                        child[i] = np.random.uniform(lb[i], ub[i])
            
            child = np.clip(child, lb, ub)
            new_pop.append(child)
        
        pop = np.array(new_pop)
        fitness = np.array([objective(p) for p in pop])
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_position = pop[current_best_idx].copy()
    
    return best_position, best_fitness
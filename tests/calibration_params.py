# -*- coding: utf-8 -*-
"""
率定算法参数统一配置文件
通过修改此文件调整所有率定任务的算法参数
"""

TWO_STAGE_PARAMS = {
    'stage1_iter': 10,
    'stage2_iter': 30,
}

PSO_PARAMS = {
    'n_particles': 100,
    'w': 0.7,
    'c1': 1.5,
    'c2': 1.5,
}

SCE_PARAMS = {
    'maxiter': 50,
    'tol': 1e-6,
    'polish': True,
}

DE_PARAMS = {
    'mutation_factor': 0.8,
    'crossover_prob': 0.7,
    'pop_size': 50,
    'maxiter': 80,
    'tol': 1e-6,
}

GA_PARAMS = {
    'pop_size': 50,
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
}
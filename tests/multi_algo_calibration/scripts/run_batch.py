# -*- coding: utf-8 -*-
"""
批量运行所有模型和算法组合
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import load_events
from configs import CATCHMENT_AREA, WARMUP_STEPS, OUTPUT_PARAMS, MUSKINGUM_BOUNDS
from models import (
    run_tank_model, TANK_PARAM_BOUNDS,
    run_hbv_model, HBV_PARAM_BOUNDS,
    run_xaj_model, XAJ_PARAM_BOUNDS,
)
from algos import (
    optimize_ga, optimize_de, optimize_pso, optimize_sce, optimize_two_stage,
)
from configs import GA_PARAMS, DE_PARAMS, PSO_PARAMS, SCE_PARAMS
from utils import calc_nse, musk


MODELS = {
    'tank': {
        'run': run_tank_model,
        'bounds': TANK_PARAM_BOUNDS,
        'area': CATCHMENT_AREA,
        'del_t': 1.0,
    },
    'hbv': {
        'run': run_hbv_model,
        'bounds': HBV_PARAM_BOUNDS,
        'area': CATCHMENT_AREA,
    },
    'xaj': {
        'run': run_xaj_model,
        'bounds': XAJ_PARAM_BOUNDS,
        'area': CATCHMENT_AREA,
    },
}

ALGORITHMS = {
    'ga': (optimize_ga, GA_PARAMS),
    'de': (optimize_de, DE_PARAMS),
    'pso': (optimize_pso, PSO_PARAMS),
    'sce': (optimize_sce, SCE_PARAMS),
    'two_stage': (optimize_two_stage, {}),
}


def run_calibration(model_name: str, algo_name: str):
    """运行单个模型+算法组合"""
    from utils.plotter import plot_event
    
    print(f"\n{'='*50}")
    print(f"模型: {model_name.upper()}, 算法: {algo_name.upper()}")
    print(f"{'='*50}")
    
    events = load_events()
    model_config = MODELS[model_name]
    algo_func, algo_params = ALGORITHMS[algo_name]
    
    pnames = list(model_config['bounds'].keys()) + ['k_routing', 'x_routing']
    bounds = list(model_config['bounds'].values()) + [
        MUSKINGUM_BOUNDS['k_routing'], MUSKINGUM_BOUNDS['x_routing']
    ]
    
    def objective(p):
        mp = {k: v for k, v in zip(pnames[:-2], p[:-2])}
        try:
            nses = []
            for e in events:
                sim = model_config['run'](e['precip'], e['evap'], mp, model_config['area'])
                if e['upstream'] is not None:
                    sim = sim + musk(e['upstream'], p[-2], p[-1])
                n = calc_nse(e['flow'][WARMUP_STEPS:], sim[WARMUP_STEPS:])
                if not np.isnan(n):
                    nses.append(n)
            return -np.mean(nses) if nses else 1e10
        except Exception:
            return 1e10
    
    best_x, best_fun = algo_func(objective, bounds, 30, len(pnames), algo_params)
    best_params = {k: v for k, v in zip(pnames, best_x)}
    best_nse = -best_fun
    
    print(f"最优NSE: {best_nse:.4f}")
    print(f"参数: {best_params}")
    
    param_file = f"{model_name}_{algo_name}_params.csv"
    import pandas as pd
    pd.DataFrame([best_params]).to_csv(os.path.join(OUTPUT_PARAMS, param_file), index=False)
    
    for e in events:
        sim = model_config['run'](e['precip'], e['evap'], best_params, model_config['area'])
        if e['upstream'] is not None:
            sim = sim + musk(e['upstream'], best_params['k_routing'], best_params['x_routing'])
        
        plot_event(e, sim, best_params, model_name, algo_name, model_name)
    
    print(f"完成 {model_name}-{algo_name}")


def main():
    import numpy as np
    
    for model_name in MODELS.keys():
        for algo_name in ALGORITHMS.keys():
            try:
                run_calibration(model_name, algo_name)
            except Exception as e:
                print(f"错误 {model_name}-{algo_name}: {e}")
    
    print("\n所有实验完成!")


if __name__ == "__main__":
    main()
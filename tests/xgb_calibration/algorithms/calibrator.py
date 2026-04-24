# -*- coding: utf-8 -*-
"""
率定器
"""
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import calc_nse
from core.utils import musk
from config import WARMUP_STEPS, MAX_ITERATIONS
from .sce import optimize_sce


MUSKINGUM_BOUNDS = {
    'k_routing': (0.5, 5.0),
    'x_routing': (0.0, 0.5),
}


class Calibrator:
    """率定器"""
    
    MODELS = {
        'tank': {
            'run': None,
            'bounds': {
                't0_is': (0.0, 50.0),
                't0_boc': (0.15, 0.5),
                't0_soc_uo': (0.2, 0.6),
                't0_soc_lo': (0.15, 0.5),
                't0_soh_uo': (50.0, 120.0),
                't0_soh_lo': (10.0, 50.0),
                't1_is': (0.0, 50.0),
                't1_boc': (0.1, 0.4),
                't1_soc': (0.1, 0.4),
                't1_soh': (20.0, 80.0),
                't2_is': (0.0, 50.0),
                't2_boc': (0.05, 0.3),
                't2_soc': (0.05, 0.3),
                't2_soh': (10.0, 60.0),
                't3_is': (0.0, 50.0),
                't3_soc': (0.001, 0.05),
                'k_routing': MUSKINGUM_BOUNDS['k_routing'],
                'x_routing': MUSKINGUM_BOUNDS['x_routing'],
            },
            'area': 584.0,
            'del_t': 1.0,
        },
        'hbv': {
            'run': None,
            'bounds': {
                'fc': (100.0, 200.0),
                'beta': (1.0, 7.0),
                'c': (0.01, 0.07),
                'k0': (0.05, 0.2),
                'l': (2.0, 5.0),
                'k1': (0.01, 0.1),
                'k2': (0.01, 0.05),
                'kp': (0.01, 0.05),
                'lp': (0.3, 1.0),
                'k_routing': MUSKINGUM_BOUNDS['k_routing'],
                'x_routing': MUSKINGUM_BOUNDS['x_routing'],
            },
            'area': 584.0,
        },
        'xaj': {
            'run': None,
            'bounds': {
                'K': (0.7, 1.3),
                'B': (0.1, 0.5),
                'IM': (0.001, 0.1),
                'WUM': (10.0, 60.0),
                'WLM': (50.0, 150.0),
                'WM': (100.0, 330.0),
                'C': (0.1, 0.5),
                'SM': (10.0, 80.0),
                'EX': (1.0, 2.0),
                'KI': (0.1, 0.5),
                'KG': (0.1, 0.5),
                'CS': (0.5, 0.98),
                'L': (0.0, 20.0),
                'CI': (0.5, 0.98),
                'CG': (0.98, 0.999),
                'k_routing': MUSKINGUM_BOUNDS['k_routing'],
                'x_routing': MUSKINGUM_BOUNDS['x_routing'],
            },
            'area': 584.0,
        },
    }
    
    def __init__(self, model_name: str):
        """初始化率定器"""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        self.config = self.MODELS[model_name]
        
        from models import run_tank_model, run_hbv_model, run_xaj_model
        if model_name == 'tank':
            self.config['run'] = run_tank_model
        elif model_name == 'hbv':
            self.config['run'] = run_hbv_model
        else:
            self.config['run'] = run_xaj_model
    
    def calibrate(self, events: List[Dict], progress_callback=None) -> Tuple[Dict, float, float, List[str]]:
        """率定模型"""
        import time
        start_time = time.time()
        
        run_func = self.config['run']
        calib_event_names = [e['name'] for e in events]
        
        param_names = list(self.config['bounds'].keys())
        model_param_names = [p for p in param_names if p not in ['k_routing', 'x_routing']]
        routing_param_names = ['k_routing', 'x_routing']
        
        model_bounds = [self.config['bounds'][p] for p in model_param_names]
        routing_bounds = [self.config['bounds'][p] for p in routing_param_names]
        
        def objective(params_array):
            n_model_params = len(model_param_names)
            model_params = {k: v for k, v in zip(model_param_names, params_array[:n_model_params])}
            k_routing = params_array[n_model_params]
            x_routing = params_array[n_model_params + 1]
            
            try:
                nses = []
                for e in events:
                    precip = e['precip']
                    evap = e['evap']
                    flow = e['flow']
                    upstream = e.get('upstream')
                    
                    if self.model_name == 'tank':
                        sim = run_func(precip, evap, model_params, self.config['area'], self.config['del_t'])
                    else:
                        sim = run_func(precip, evap, model_params, self.config['area'])
                    
                    if upstream is not None:
                        sim = sim + musk(upstream, k_routing, x_routing)
                    
                    nse = calc_nse(flow[WARMUP_STEPS:], sim[WARMUP_STEPS:])
                    if not np.isnan(nse):
                        nses.append(nse)
                return -np.mean(nses) if nses else 1e10
            except Exception:
                return 1e10
        
        all_bounds = model_bounds + routing_bounds
        best_params_array, best_obj = optimize_sce(objective, all_bounds, MAX_ITERATIONS, len(param_names))
        
        elapsed_time = time.time() - start_time
        best_params = {k: v for k, v in zip(param_names, best_params_array)}
        best_nse = -best_obj
        
        return best_params, best_nse, elapsed_time, calib_event_names
    
    def run_all_events(self, params: Dict, events: List[Dict]) -> List[Dict]:
        """用率定参数运行所有场次"""
        run_func = self.config['run']
        
        model_params = {k: v for k, v in params.items() if k not in ['k_routing', 'x_routing']}
        k_routing = params.get('k_routing', 1.0)
        x_routing = params.get('x_routing', 0.2)
        
        results = []
        for e in events:
            precip = e['precip']
            evap = e['evap']
            flow = e['flow']
            upstream = e.get('upstream')
            
            if self.model_name == 'tank':
                sim = run_func(precip, evap, model_params, self.config['area'], self.config['del_t'])
            else:
                sim = run_func(precip, evap, model_params, self.config['area'])
            
            if upstream is not None:
                sim = sim + musk(upstream, k_routing, x_routing)
            
            error = sim - flow
            
            results.append({
                'name': e['name'],
                'precip': precip,
                'evap': evap,
                'flow': flow,
                'sim': sim,
                'error': error,
            })
        
        return results
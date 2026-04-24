# -*- coding: utf-8 -*-
"""
率定模块 - SCE-UA算法封装
"""
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from algos.sce import optimize_sce
from xgb_error_correction.data_loader import calc_nse
from xgb_error_correction.config import WARMUP_STEPS, MAX_ITERATIONS


MUSKINGUM_BOUNDS = {
    'k_routing': (0.5, 5.0),
    'x_routing': (0.0, 0.5),
}


def musk(u: np.ndarray, k: float, x: float, dt: float = 1.0) -> np.ndarray:
    """马斯京根汇流方法
    
    Args:
        u: 上游流量序列
        k: 传播时间
        x: 流量分配系数
        dt: 时间步长
        
    Returns:
        演算后的流量序列
    """
    n = len(u)
    if n == 0:
        return np.array([])
    
    d = k * (1 - x) + 0.5 * dt
    if abs(d) < 1e-12:
        return u
    
    C0 = (-k * x + 0.5 * dt) / d
    C1 = (k * x + 0.5 * dt) / d
    C2 = (k * (1 - x) - 0.5 * dt) / d
    
    r = np.zeros(n)
    r[0] = u[0]
    for t in range(1, n):
        r[t] = C0 * u[t] + C1 * u[t - 1] + C2 * r[t - 1]
    
    return np.maximum(r, 0)


def run_tank_model(precip: np.ndarray, evap: np.ndarray, params: Dict, area: float, del_t: float = 1.0) -> np.ndarray:
    """运行Tank模型"""
    from models.tank import run_tank_model as _run
    return _run(precip, evap, params, area, del_t)


def run_hbv_model(precip: np.ndarray, evap: np.ndarray, params: Dict, area: float) -> np.ndarray:
    """运行HBV模型"""
    from models.hbv import run_hbv_model as _run
    return _run(precip, evap, params, area)


def run_xaj_model(precip: np.ndarray, evap: np.ndarray, params: Dict, area: float) -> np.ndarray:
    """运行XAJ模型"""
    from models.xaj import run_xaj_model as _run
    return _run(precip, evap, params, area)


class Calibrator:
    """率定器"""
    
    MODELS = {
        'tank': {
            'run': run_tank_model,
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
            'default': {},
            'area': 584.0,
            'del_t': 1.0,
        },
        'hbv': {
            'run': run_hbv_model,
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
            'default': {},
            'area': 584.0,
        },
        'xaj': {
            'run': run_xaj_model,
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
            'default': {},
            'area': 584.0,
        },
    }
    
    def __init__(self, model_name: str):
        """初始化率定器
        
        Args:
            model_name: 模型名称 'tank' | 'hbv' | 'xaj'
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        self.config = self.MODELS[model_name]
    
    def calibrate(self, events: List[Dict], progress_callback=None) -> Tuple[Dict, float, float, List[str]]:
        """率定模型
        
        Args:
            events: 洪水场次列表（用于率定的场次）
            progress_callback: 进度回调
            
        Returns:
            (最优参数, NSE均值, 耗时秒, 选取的场次名称列表)
        """
        import time
        start_time = time.time()
        
        model_config = self.config
        run_func = model_config['run']
        
        # 记录用于率定的场次名称
        calib_event_names = [e['name'] for e in events]
        
        # 分离水文参数和马斯京根参数
        param_names = list(model_config['bounds'].keys())
        model_param_names = [p for p in param_names if p not in ['k_routing', 'x_routing']]
        routing_param_names = ['k_routing', 'x_routing']
        
        model_bounds = [model_config['bounds'][p] for p in model_param_names]
        routing_bounds = [model_config['bounds'][p] for p in routing_param_names]
        
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
                        sim = run_func(precip, evap, model_params, model_config['area'], model_config['del_t'])
                    elif self.model_name == 'hbv':
                        sim = run_func(precip, evap, model_params, model_config['area'])
                    else:
                        sim = run_func(precip, evap, model_params, model_config['area'])
                    
                    # 马斯京根汇流
                    if upstream is not None:
                        sim = sim + musk(upstream, k_routing, x_routing)
                    
                    nse = calc_nse(flow[WARMUP_STEPS:], sim[WARMUP_STEPS:])
                    if not np.isnan(nse):
                        nses.append(nse)
                return -np.mean(nses) if nses else 1e10
            except Exception:
                return 1e10
        
        all_bounds = model_bounds + routing_bounds
        best_params_array, best_obj = optimize_sce(
            objective, all_bounds, MAX_ITERATIONS, len(param_names)
        )
        
        elapsed_time = time.time() - start_time
        best_params = {k: v for k, v in zip(param_names, best_params_array)}
        best_nse = -best_obj
        
        return best_params, best_nse, elapsed_time, calib_event_names
    
    def run_all_events(self, params: Dict, events: List[Dict]) -> List[Dict]:
        """用率定参数运行所有场次
        
        Args:
            params: 率定参数
            events: 洪水场次列表
            
        Returns:
            结果列表，每个包含 name, precip, evap, flow, sim, error
        """
        model_config = self.config
        run_func = model_config['run']
        
        # 分离水文参数和马斯京根参数
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
                sim = run_func(precip, evap, model_params, model_config['area'], model_config['del_t'])
            elif self.model_name == 'hbv':
                sim = run_func(precip, evap, model_params, model_config['area'])
            else:
                sim = run_func(precip, evap, model_params, model_config['area'])
            
            # 马斯京根汇流
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
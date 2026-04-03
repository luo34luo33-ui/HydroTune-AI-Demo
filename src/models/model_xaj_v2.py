import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from .base_model import BaseModel


def _find_xaj_path():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    possible_roots = [
        os.path.dirname(os.path.dirname(current_dir)),
        os.path.dirname(current_dir),
        os.getcwd(),
    ]
    
    for root in possible_roots:
        xaj_path = os.path.join(root, "XAJ-model-structured")
        config_path = os.path.join(xaj_path, "config.py")
        if os.path.exists(xaj_path) and os.path.exists(config_path):
            return os.path.abspath(xaj_path)
    
    return None


_xaj_base_path = _find_xaj_path()
XAJ_AVAILABLE = False
run_new_xaj = None


def _import_xaj_module():
    global XAJ_AVAILABLE, run_new_xaj
    
    if _xaj_base_path is None:
        XAJ_AVAILABLE = False
        return
    
    saved_modules = {}
    for k in list(sys.modules.keys()):
        if ('config' in k.lower() or 'xaj' in k.lower()) and k != __name__:
            if 'src' not in (getattr(sys.modules.get(k), '__file__', '') or ''):
                saved_modules[k] = sys.modules.pop(k, None)
    
    xaj_str = str(_xaj_base_path)
    if xaj_str in sys.path:
        sys.path.remove(xaj_str)
    sys.path.insert(0, xaj_str)
    
    try:
        from config import PARAM_RANGES, DEFAULT_PARAMS, validate_params
        from main import run_new_xaj
        XAJ_AVAILABLE = True
    except ImportError as e:
        print(f"[WARN] XAJ 模块导入失败: {e}")
        XAJ_AVAILABLE = False


_import_xaj_module()


class XAJModelV2(BaseModel):
    """新安江模型版本2 - 包含上游出库Muskingum路由"""
    
    @property
    def name(self) -> str:
        return "新安江模型2"
    
    @property
    def model_type(self) -> str:
        return "lumped"
    
    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'B': (0.2, 0.4),
            'C': (0.1, 0.3),
            'WM': (100.0, 200.0),
            'WUM': (15.0, 35.0),
            'WLM': (40.0, 80.0),
            'IM': (0.0, 0.05),
            'SM': (20.0, 45.0),
            'EX': (0.8, 1.5),
            'K': (0.8, 1.5),
            'KG': (0.25, 0.5),
            'KI': (0.1, 0.3),
            'CG': (0.99, 0.999),
            'CI': (0.6, 0.95),
            'CS': (0.5, 0.9),
            'L': (0, 5),
            'X': (0.15, 0.4),
            'K_res': (3.0, 7.0),
            'X_res': (0.1, 0.2),
            'n': (1, 10)
        }
    
    @property
    def default_params(self) -> Dict[str, float]:
        return {
            'B': 0.3,
            'C': 0.2,
            'WM': 150.0,
            'WUM': 23.867202119765466,
            'WLM': 60.0,
            'IM': 0.02,
            'SM': 30.921720241298537,
            'EX': 1.1236984984316019,
            'K': 1.2,
            'KG': 0.3697662256735042,
            'KI': 0.2,
            'CG': 0.9977960537453832,
            'CI': 0.85,
            'CS': 0.7232327466183337,
            'L': 1,
            'X': 0.2696357296913911,
            'K_res': 4.877223153101952,
            'X_res': 0.14468991188350286,
            'n': 5
        }
    
    def get_timestep_hours(self, spatial_data):
        if spatial_data is None:
            return 1
        timestep = spatial_data.get('timestep', 'daily')
        return 1 if timestep == 'hourly' else 24
    
    def run(self, precip, evap, params, spatial_data=None, temperature=None, warmup_steps=0):
        global XAJ_AVAILABLE, run_new_xaj
        
        if not XAJ_AVAILABLE:
            _import_xaj_module()
        
        if not XAJ_AVAILABLE:
            raise RuntimeError("XAJ 模型不可用")
        
        area = 150.7944 if spatial_data is None else spatial_data.get('area', 150.7944)
        timestep_hours = self.get_timestep_hours(spatial_data)
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        n = len(precip)
        p_and_e = np.zeros((n, 1, 2))
        p_and_e[:, 0, 0] = precip
        p_and_e[:, 0, 1] = evap
        
        try:
            q_sim, es = run_new_xaj(p_and_e, full_params, warmup_length=warmup_steps, return_state=False)
            
            runoff_mm = q_sim[:, 0, 0]
            seconds_per_step = timestep_hours * 3600
            flow = runoff_mm * area * 1000 / seconds_per_step
            
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            
            return flow
        except Exception as e:
            print(f"[ERROR] XAJ模型运行失败: {e}")
            return np.zeros(n)
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        return True
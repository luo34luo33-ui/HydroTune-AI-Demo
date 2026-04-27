# -*- coding: utf-8 -*-
"""
新安江模型V2适配器
使用 tests/models/xaj.py 中的纯 NumPy 实现（高性能版本）
"""
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from .base_model import BaseModel

# 尝试导入 tests 中的纯 NumPy 版本
XAJ_NUMPY_AVAILABLE = False
run_xaj_model = None
XAJ_PARAM_BOUNDS = None

def _import_xaj_numpy():
    global XAJ_NUMPY_AVAILABLE, run_xaj_model, XAJ_PARAM_BOUNDS
    
    possible_paths = [
        os.path.join(os.getcwd(), "tests", "models", "xaj.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "tests", "models", "xaj.py"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("xaj_numpy", path)
                if spec and spec.loader:
                    xaj_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(xaj_module)
                    run_xaj_model = xaj_module.run_xaj_model
                    XAJ_PARAM_BOUNDS = xaj_module.XAJ_PARAM_BOUNDS
                    XAJ_NUMPY_AVAILABLE = True
                    print("[INFO] XAJ NumPy版本加载成功")
                    return
            except Exception as e:
                print(f"[WARN] XAJ NumPy版本导入失败: {e}")
    
    XAJ_NUMPY_AVAILABLE = False

import importlib.util
_import_xaj_numpy()


class XAJModelV2(BaseModel):
    """新安江模型V2 - 使用纯NumPy实现（高性能版本）"""
    
    @property
    def name(self) -> str:
        return "新安江模型2"
    
    @property
    def model_type(self) -> str:
        return "lumped"
    
    @property
    def supports_hourly(self) -> bool:
        return True
    
    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        if XAJ_PARAM_BOUNDS is not None:
            return XAJ_PARAM_BOUNDS
        return {
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
            'L': (0, 20),
            'CI': (0.5, 0.98),
            'CG': (0.98, 0.999),
        }
    
    @property
    def default_params(self) -> Dict[str, float]:
        return {
            'K': 1.1,
            'B': 0.3,
            'IM': 0.01,
            'WUM': 20.0,
            'WLM': 70.0,
            'WM': 120.0,
            'C': 0.15,
            'SM': 50.0,
            'EX': 1.5,
            'KI': 0.15,
            'KG': 0.3,
            'CS': 0.6,
            'L': 1,
            'CI': 0.8,
            'CG': 0.95,
        }
    
    @property
    def fixed_params(self) -> Dict[str, float]:
        return {}
    
    def get_param_descriptions(self) -> Dict[str, str]:
        return {
            'K': '蒸散发能力折减系数',
            'B': '张力水蓄水容量曲线指数',
            'IM': '不透水面积比例',
            'WUM': '上层张力水蓄水容量 (mm)',
            'WLM': '下层张力水蓄水容量 (mm)',
            'WM': '流域平均张力水蓄水容量 (mm)',
            'C': '深层蒸散发系数',
            'SM': '自由水蓄水容量 (mm)',
            'EX': '自由水蓄水容量曲线指数',
            'KI': '壤中流出流系数',
            'KG': '地下水出流系数',
            'CS': '河网蓄水消退系数',
            'L': '滞后时段数',
            'CI': '壤中流消退系数',
            'CG': '地下水消退系数',
        }
    
    def get_required_spatial_data(self) -> list:
        return ['area']
    
    def get_timestep_hours(self, spatial_data: Optional[Dict] = None):
        if spatial_data is None:
            return 24
        timestep = spatial_data.get('timestep', 'daily')
        return 1 if timestep == 'hourly' else 24
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float], spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None, warmup_steps: int = 0) -> np.ndarray:
        """运行新安江模型（纯NumPy版本，高性能）"""
        global run_xaj_model, XAJ_NUMPY_AVAILABLE
        
        if not XAJ_NUMPY_AVAILABLE:
            _import_xaj_numpy()
        
        if not XAJ_NUMPY_AVAILABLE or run_xaj_model is None:
            raise RuntimeError("XAJ NumPy 模型不可用")
        
        area = 584.0 if spatial_data is None else spatial_data.get('area', 584.0)
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        try:
            flow = run_xaj_model(
                precip=precip,
                evap=evap,
                params=full_params,
                area=area,
            )
            
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            flow = np.maximum(flow, 0)
            
            return flow
            
        except Exception as e:
            print(f"[ERROR] XAJ NumPy 模型运行失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(len(precip))
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        for name, value in params.items():
            if name in self.param_bounds:
                min_val, max_val = self.param_bounds[name]
                if not (min_val <= value <= max_val):
                    return False
        return True
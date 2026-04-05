# -*- coding: utf-8 -*-
"""
新安江模型V2适配器
使用新的xaj_core.py核心代码 (XinAnJiangModel类)
"""
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

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
        core_path = os.path.join(xaj_path, "xaj_core.py")
        if os.path.exists(xaj_path) and os.path.exists(core_path):
            return os.path.abspath(xaj_path)
    
    return None

_xaj_base_path = _find_xaj_path()
XAJ_V3_AVAILABLE = False
XinAnJiangModel = None

def _import_xaj_module():
    global XAJ_V3_AVAILABLE, XinAnJiangModel
    
    if _xaj_base_path is None:
        XAJ_V3_AVAILABLE = False
        return
    
    core_path = os.path.join(_xaj_base_path, "xaj_core.py")
    
    if not os.path.exists(core_path):
        XAJ_V3_AVAILABLE = False
        return
    
    try:
        spec = importlib.util.spec_from_file_location("xaj_core", core_path)
        if spec and spec.loader:
            xaj_core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(xaj_core)
            XinAnJiangModel = xaj_core.XinAnJiangModel
            XAJ_V3_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] XAJ V3 模块导入失败: {e}")
        XAJ_V3_AVAILABLE = False

_import_xaj_module()


class XAJModelV2(BaseModel):
    """新安江模型V2 - 使用新的xaj_core.py核心代码"""
    
    @property
    def name(self) -> str:
        return "新安江模型2"
    
    @property
    def model_type(self) -> str:
        return "lumped"
    
    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'B': (0.1, 0.5),
            'C': (0.1, 0.3),
            'WM': (100.0, 200.0),
            'WUM': (10.0, 40.0),
            'WLM': (40.0, 100.0),
            'IM': (0.0, 0.1),
            'SM': (10.0, 80.0),
            'EX': (1.0, 2.0),
            'K': (0.8, 1.5),
            'KG': (0.2, 0.5),
            'KI': (0.1, 0.3),
            'CG': (0.90, 0.999),
            'CI': (0.7, 0.95),
            'CS': (0.15, 0.85),
            'L': (0, 5),
            'X': (0.0, 0.5),
            'K_res': (1.0, 8.0),
            'X_res': (0.0, 0.5),
            'n': (0, 7),
            'Area': (100.0, 1500.0)
        }
    
    @property
    def default_params(self) -> Dict[str, float]:
        return {
            'B': 0.3,
            'C': 0.2,
            'WM': 150.0,
            'WUM': 23.87,
            'WLM': 60.0,
            'IM': 0.02,
            'SM': 30.92,
            'EX': 1.12,
            'K': 1.2,
            'KG': 0.37,
            'KI': 0.2,
            'CG': 0.998,
            'CI': 0.85,
            'CS': 0.72,
            'L': 1,
            'X': 0.27,
            'K_res': 4.88,
            'X_res': 0.14,
            'n': 5,
            'Area': 584.0
        }
    
    def get_param_descriptions(self) -> Dict[str, str]:
        return {
            'B': '张力水蓄水容量曲线指数',
            'C': '深层蒸散发系数',
            'WM': '流域平均张力水蓄水容量 (mm)',
            'WUM': '上层张力水蓄水容量 (mm)',
            'WLM': '下层张力水蓄水容量 (mm)',
            'IM': '不透水面积比例',
            'SM': '自由水蓄水容量 (mm)',
            'EX': '自由水蓄水容量曲线指数',
            'K': '蒸散发能力折算系数',
            'KG': '地下水出流系数',
            'KI': '壤中流出流系数',
            'CG': '地下水消退系数',
            'CI': '壤中流消退系数',
            'CS': '河网蓄水消退系数',
            'L': '滞后时段数',
            'X': '河道流量演算比重因子',
            'K_res': '水库出库演进传播时间 (h)',
            'X_res': '水库出库演进流量比重因子',
            'n': '河道单元数',
            'Area': '流域面积 (km²)'
        }
    
    def get_required_spatial_data(self) -> list:
        return ['area']
    
    def get_timestep_hours(self, spatial_data: Optional[Dict] = None):
        if spatial_data is None:
            return 1
        timestep = spatial_data.get('timestep', 'daily')
        return 1 if timestep == 'hourly' else 24
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float], spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None, warmup_steps: int = 0) -> np.ndarray:
        """运行新安江模型V3"""
        global XAJ_V3_AVAILABLE, XinAnJiangModel
        
        if not XAJ_V3_AVAILABLE:
            _import_xaj_module()
        
        if not XAJ_V3_AVAILABLE:
            raise RuntimeError("XAJ V3 模型不可用")
        
        area = 584.0 if spatial_data is None else spatial_data.get('area', 584.0)
        
        full_params = self.default_params.copy()
        full_params.update(params)
        full_params['Area'] = area
        
        n = len(precip)
        df = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=n, freq='h'),
            'P': precip,
            'E0': evap,
            'Qres_in': np.zeros(n)
        })
        
        try:
            model = XinAnJiangModel()
            model.set_params(**{k: v for k, v in full_params.items() 
                              if k not in ['Area']})
            model.Area = full_params.get('Area', 584.0)
            
            n_reaches = int(full_params.get('n', 1))
            result = model.run_model(df, n_reaches=n_reaches)
            
            flow = result['Q_total'].values
            
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            flow = np.maximum(flow, 0)
            
            return flow
            
        except Exception as e:
            print(f"[ERROR] XAJ V3 运行失败: {e}")
            return np.zeros(n)
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        for name, value in params.items():
            if name in self.param_bounds:
                min_val, max_val = self.param_bounds[name]
                if not (min_val <= value <= max_val):
                    return False
        return True
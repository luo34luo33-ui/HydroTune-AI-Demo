# -*- coding: utf-8 -*-
"""
Tank水箱模型适配器
使用 tests/models/tank.py 中的纯 NumPy 实现（高性能版本）
"""
import os
import sys
import importlib.util
from typing import Dict, Tuple, Optional
import numpy as np

from .base_model import BaseModel

# 尝试导入 tests 中的纯 NumPy 版本
TANK_NUMPY_AVAILABLE = False
run_tank_model = None
TANK_PARAM_BOUNDS = None
TANK_PARAM_ORDER = None

def _import_tank_numpy():
    global TANK_NUMPY_AVAILABLE, run_tank_model, TANK_PARAM_BOUNDS, TANK_PARAM_ORDER
    
    possible_paths = [
        os.path.join(os.getcwd(), "tests", "models", "tank.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "tests", "models", "tank.py"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("tank_numpy", path)
                if spec and spec.loader:
                    tank_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(tank_module)
                    run_tank_model = tank_module.run_tank_model
                    TANK_PARAM_BOUNDS = tank_module.TANK_PARAM_BOUNDS
                    TANK_PARAM_ORDER = tank_module.TANK_PARAM_ORDER
                    TANK_NUMPY_AVAILABLE = True
                    print("[INFO] Tank NumPy版本加载成功")
                    return
            except Exception as e:
                print(f"[WARN] Tank NumPy版本导入失败: {e}")
    
    TANK_NUMPY_AVAILABLE = False

_import_tank_numpy()


class TankModel(BaseModel):
    """
    Tank水箱模型 (Sugawara & Funiyuki, 1956)
    
    经典的概念性水文模型，通过多层水箱模拟水文过程。
    
    模型特点：
    - 四层串联水箱结构
    - 每层具有侧孔和底孔出流
    - 适用于湿润地区流域模拟
    """

    @property
    def name(self) -> str:
        return "tank水箱模型"

    @property
    def model_type(self) -> str:
        return "lumped"

    @property
    def supports_hourly(self) -> bool:
        return True

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        if TANK_PARAM_BOUNDS is not None:
            return TANK_PARAM_BOUNDS
        return {
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
        }

    @property
    def default_params(self) -> Dict[str, float]:
        return {
            't0_is': 10.0,
            't0_boc': 0.3,
            't0_soc_uo': 0.4,
            't0_soc_lo': 0.3,
            't0_soh_uo': 80.0,
            't0_soh_lo': 30.0,
            't1_is': 10.0,
            't1_boc': 0.25,
            't1_soc': 0.25,
            't1_soh': 50.0,
            't2_is': 10.0,
            't2_boc': 0.15,
            't2_soc': 0.15,
            't2_soh': 35.0,
            't3_is': 10.0,
            't3_soc': 0.02,
        }

    def get_param_descriptions(self) -> Dict[str, str]:
        return {
            't0_is': 'Tank-0 初始蓄水量 (mm)',
            't0_boc': 'Tank-0 底孔出流系数',
            't0_soc_uo': 'Tank-0 侧孔出流系数（上层）',
            't0_soc_lo': 'Tank-0 侧孔出流系数（下层）',
            't0_soh_uo': 'Tank-0 侧孔高度（上层）(mm)',
            't0_soh_lo': 'Tank-0 侧孔高度（下层）(mm)',
            't1_is': 'Tank-1 初始蓄水量 (mm)',
            't1_boc': 'Tank-1 底孔出流系数',
            't1_soc': 'Tank-1 侧孔出流系数',
            't1_soh': 'Tank-1 侧孔高度 (mm)',
            't2_is': 'Tank-2 初始蓄水量 (mm)',
            't2_boc': 'Tank-2 底孔出流系数',
            't2_soc': 'Tank-2 侧孔出流系数',
            't2_soh': 'Tank-2 侧孔高度 (mm)',
            't3_is': 'Tank-3 初始蓄水量 (mm，基流)',
            't3_soc': 'Tank-3 侧孔出流系数（基流）',
        }

    def get_required_spatial_data(self) -> list:
        return ['area']

    def get_timestep_hours(self, spatial_data: Optional[Dict] = None):
        if spatial_data is None:
            return 24
        timestep = spatial_data.get('timestep', 'daily')
        return 1 if timestep == 'hourly' else 24

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
        warmup_steps: int = 0,
    ) -> np.ndarray:
        """
        运行Tank水箱模型（纯NumPy版本，高性能）
        
        参数
        ----------
        precip : np.ndarray
            降水序列 (mm), shape: (n_timesteps,)
        evap : np.ndarray
            蒸发序列 (mm), shape: (n_timesteps,)
        params : Dict[str, float]
            模型参数字典
        spatial_data : Dict, optional
            空间数据，默认 {'area': 150.7944, 'del_t': 24.0}
        temperature : np.ndarray, optional
            温度序列，Tank模型不使用此参数
        warmup_steps : int
            预热期步数
        
        返回
        -------
        np.ndarray
            模拟流量序列 (m³/s), shape: (n_timesteps,)
        """
        global run_tank_model, TANK_NUMPY_AVAILABLE
        
        if not TANK_NUMPY_AVAILABLE:
            _import_tank_numpy()
        
        if not TANK_NUMPY_AVAILABLE or run_tank_model is None:
            raise RuntimeError("Tank NumPy 模型不可用")
        
        if spatial_data is None:
            spatial_data = {'area': 150.7944, 'del_t': 24.0}
        
        area = spatial_data.get('area', 150.7944)
        
        if 'del_t' in spatial_data:
            del_t = spatial_data['del_t']
        else:
            timestep = spatial_data.get('timestep', 'daily')
            del_t = 1.0 if timestep == 'hourly' else 24.0
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        if del_t == 1.0:
            precip_input = precip * 24.0
            evap_input = evap * 24.0
        else:
            precip_input = precip
            evap_input = evap
        
        try:
            flow = run_tank_model(
                precip=precip_input,
                evap=evap_input,
                params=full_params,
                area=area,
                del_t=del_t,
            )
            
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            flow = np.maximum(flow, 0)
            
            return flow
            
        except Exception as e:
            print(f"[ERROR] Tank NumPy 模型运行失败: {e}")
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
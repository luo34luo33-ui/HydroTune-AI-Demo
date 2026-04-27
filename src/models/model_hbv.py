# -*- coding: utf-8 -*-
"""
HBV水文模型适配器
使用 tests/models/hbv.py 中的纯 NumPy 实现（高性能版本）
"""
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from .base_model import BaseModel

# 尝试导入 tests 中的纯 NumPy 版本
HBV_NUMPY_AVAILABLE = False
run_hbv_model = None
HBV_PARAM_BOUNDS = None

def _import_hbv_numpy():
    global HBV_NUMPY_AVAILABLE, run_hbv_model, HBV_PARAM_BOUNDS
    
    possible_paths = [
        os.path.join(os.getcwd(), "tests", "models", "hbv.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "tests", "models", "hbv.py"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("hbv_numpy", path)
                if spec and spec.loader:
                    hbv_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(hbv_module)
                    run_hbv_model = hbv_module.run_hbv_model
                    HBV_PARAM_BOUNDS = hbv_module.HBV_PARAM_BOUNDS
                    HBV_NUMPY_AVAILABLE = True
                    print("[INFO] HBV NumPy版本加载成功")
                    return
            except Exception as e:
                print(f"[WARN] HBV NumPy版本导入失败: {e}")
    
    HBV_NUMPY_AVAILABLE = False

_import_hbv_numpy()


class HBVModelAdapter(BaseModel):
    """
    HBV (Hydrologiska Byråns Vattenbalans) 水文模型
    
    经典的概念性水文模型，由瑞典水文气象局开发。
    
    模型特点：
    - 土壤含水量模块
    - 蒸散发计算
    - 响应函数模块（三个线性水库）
    - 汇流模块
    
    参数说明：
    - fc: 田间持水量 (mm)
    - beta: 形状参数
    - c: 潜在蒸散发温度校正系数
    - k0, k1, k2: 各层出流系数
    - l: 表层储水阈值 (mm)
    - kp: 储水间交换系数
    - lp: 蒸散发限制系数
    """

    DEFAULT_MONTHLY_TEMP = np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3])
    DEFAULT_MONTHLY_PET = np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])

    @property
    def name(self) -> str:
        return "HBV模型"

    @property
    def model_type(self) -> str:
        return "lumped"

    @property
    def supports_hourly(self) -> bool:
        return False

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        if HBV_PARAM_BOUNDS is not None:
            return HBV_PARAM_BOUNDS
        return {
            'fc': (100.0, 200.0),
            'beta': (1.0, 7.0),
            'c': (0.01, 0.07),
            'k0': (0.05, 0.2),
            'l': (2.0, 5.0),
            'k1': (0.01, 0.1),
            'k2': (0.01, 0.05),
            'kp': (0.01, 0.05),
            'lp': (0.3, 1.0),
        }

    @property
    def default_params(self) -> Dict[str, float]:
        return {
            'fc': 195.0,
            'beta': 2.6143,
            'c': 0.07,
            'k0': 0.163,
            'l': 4.87,
            'k1': 0.029,
            'k2': 0.049,
            'kp': 0.050,
            'lp': 0.5,
        }

    def get_param_descriptions(self) -> Dict[str, str]:
        return {
            'fc': '田间持水量 - 土壤最大蓄水容量 (mm)',
            'beta': '形状参数 - 控制产流的非线性程度',
            'c': '潜在蒸散发温度校正系数',
            'k0': '表层储水出流系数 - 快速径流响应 (day⁻¹)',
            'l': '表层储水阈值 (mm)',
            'k1': '上层储水出流系数 - 慢速径流响应 (day⁻¹)',
            'k2': '下层储水出流系数 - 基流响应 (day⁻¹)',
            'kp': '储水间交换系数 - 上下层储水交换 (day⁻¹)',
            'lp': '蒸散发限制系数 - 限制蒸散发上限',
        }

    def get_required_spatial_data(self) -> list:
        return ['area']

    def get_timestep_hours(self, spatial_data: Optional[Dict] = None):
        return 24

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
        运行HBV水文模型（纯NumPy版本，高性能）
        
        参数
        ----------
        precip : np.ndarray
            降水序列 (mm/day), shape: (n_timesteps,)
        evap : np.ndarray
            潜在蒸散发序列 (mm/day), shape: (n_timesteps,)
        params : Dict[str, float]
            模型参数字典
        spatial_data : Dict, optional
            空间数据，默认 {'area': 150.7944}
        temperature : np.ndarray, optional
            温度序列 (°C)，本版本不需要
        warmup_steps : int
            预热期步数
        
        返回
        -------
        np.ndarray
            模拟流量序列 (m³/s), shape: (n_timesteps,)
        """
        global run_hbv_model, HBV_NUMPY_AVAILABLE
        
        if not HBV_NUMPY_AVAILABLE:
            _import_hbv_numpy()
        
        if not HBV_NUMPY_AVAILABLE or run_hbv_model is None:
            raise RuntimeError("HBV NumPy 模型不可用")
        
        if spatial_data is None:
            spatial_data = {'area': 150.7944}
        
        area = spatial_data.get('area', 150.7944)
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        try:
            flow = run_hbv_model(
                precip=precip,
                evap=evap,
                params=full_params,
                area=area,
            )
            
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            flow = np.maximum(flow, 0)
            
            return flow
            
        except Exception as e:
            print(f"[ERROR] HBV NumPy 模型运行失败: {e}")
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
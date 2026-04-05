# -*- coding: utf-8 -*-
"""
模型运行器基类 - 封装率定算法、马斯京根演算等核心功能
新模型接入后自动获得这些功能支持
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class CalibrationResult:
    """率定结果"""
    best_params: Dict[str, float]
    best_nse: float
    simulated: np.ndarray
    
@dataclass
class SimulationResult:
    """模拟结果"""
    flow: np.ndarray
    metadata: Dict[str, Any] = None


class BaseRunner(ABC):
    """模型运行器基类
    
    功能：
    1. 基础模拟 - run()
    2. 带上游演算的模拟 - simulate_with_routing()
    3. 参数率定 - calibrate()
    4. 指标计算 - evaluate()
    """
    
    def __init__(self, adapter=None):
        self.adapter = adapter
        self._name = adapter.name if adapter else "Unknown"
    
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None,
            warmup_steps: int = 0) -> np.ndarray:
        """运行模型 - 子类必须实现"""
        pass
    
    def simulate(self, precip: np.ndarray, evap: np.ndarray,
                 params: Dict[str, float],
                 spatial_data: Optional[Dict] = None,
                 temperature: Optional[np.ndarray] = None,
                 warmup_steps: int = 0) -> SimulationResult:
        """基础模拟 - 返回包含元数据的结果"""
        flow = self.run(precip, evap, params, spatial_data, temperature, warmup_steps)
        return SimulationResult(
            flow=flow,
            metadata={'model': self.name, 'n_timesteps': len(flow)}
        )
    
    def simulate_with_routing(self, precip: np.ndarray, evap: np.ndarray,
                              params: Dict[str, float],
                              spatial_data: Optional[Dict] = None,
                              temperature: Optional[np.ndarray] = None,
                              upstream_flow: Optional[np.ndarray] = None,
                              routing_params: Optional[Dict[str, float]] = None,
                              warmup_steps: int = 0) -> np.ndarray:
        """★ 模拟 + 马斯京根上游演算
        
        Args:
            precip: 降水数组
            evap: 蒸发数组
            params: 模型参数字典
            spatial_data: 空间数据
            temperature: 温度数组
            upstream_flow: 上游流量数组 (可选)
            routing_params: 马斯京根参数 {k_routing, x_routing} (可选)
            warmup_steps: 预热期步数
            
        Returns:
            模拟流量数组 (m³/s)
        """
        flow = self.run(precip, evap, params, spatial_data, temperature, warmup_steps)
        
        if upstream_flow is not None and routing_params is not None:
            from src.hydro_calc import muskingum_routing
            k = routing_params.get('k_routing', 2.5)
            x = routing_params.get('x_routing', 0.25)
            routed = muskingum_routing(upstream_flow, k, x)
            flow = flow + routed
        
        return flow
    
    def calibrate(self, precip: np.ndarray, evap: np.ndarray,
                  observed_flow: np.ndarray,
                  algorithm: str = 'two_stage',
                  max_iter: int = 30,
                  spatial_data: Optional[Dict] = None,
                  temperature: Optional[np.ndarray] = None,
                  upstream_flow: Optional[np.ndarray] = None,
                  enable_routing: bool = False,
                  warmup_steps: int = 0) -> CalibrationResult:
        """★ 率定模型参数
        
        Args:
            precip: 降水数组
            evap: 蒸发数组
            observed_flow: 实测流量数组
            algorithm: 优化算法 ('two_stage', 'pso', 'ga', 'sce', 'de')
            max_iter: 最大迭代次数
            spatial_data: 空间数据
            temperature: 温度数组
            upstream_flow: 上游流量数组 (可选)
            enable_routing: 是否启用上游演算
            warmup_steps: 预热期步数
            
        Returns:
            CalibrationResult: 包含最优参数、NSE、模拟流量
        """
        from src.hydro_calc import calibrate_model_fast
        
        kwargs = {
            'spatial_data': spatial_data,
            'temperature': temperature,
            'algorithm': algorithm,
            'max_iter': max_iter,
            'warmup_steps': warmup_steps,
        }
        
        if enable_routing and upstream_flow is not None:
            kwargs['upstream_flow'] = upstream_flow
            kwargs['enable_routing'] = True
        
        result = calibrate_model_fast(
            model_name=self.name,
            precip=precip,
            evap=evap,
            observed_flow=observed_flow,
            **kwargs
        )
        
        return CalibrationResult(
            best_params=result[0],
            best_nse=result[1],
            simulated=result[2]
        )
    
    def evaluate(self, observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
        """★ 计算评估指标
        
        Args:
            observed: 实测流量
            simulated: 模拟流量
            
        Returns:
            包含 NSE, RMSE, MAE, PBIAS, KGE 的字典
        """
        from src.hydro_calc import calc_nse, calc_rmse, calc_mae, calc_pbias, calc_kge
        
        return {
            'nse': calc_nse(observed, simulated),
            'rmse': calc_rmse(observed, simulated),
            'mae': calc_mae(observed, simulated),
            'pbias': calc_pbias(observed, simulated),
            'kge': calc_kge(observed, simulated)
        }
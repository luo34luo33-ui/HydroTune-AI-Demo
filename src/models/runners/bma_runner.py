# -*- coding: utf-8 -*-
"""
BMA集成运行器 - 支持多模型BMA集成预报
"""
import numpy as np
from typing import List, Dict, Tuple
from .base_runner import BaseRunner, CalibrationResult

class BMARunner:
    """BMA集成运行器
    
    用于多模型集成预报，自动计算BMA权重并集成各模型输出
    """
    
    def __init__(self, runners: List[BaseRunner]):
        """初始化BMA运行器
        
        Args:
            runners: BaseRunner列表
        """
        self.runners = runners
        self.names = [r.name for r in runners]
    
    def run_ensemble(self, 
                     precip: np.ndarray, 
                     evap: np.ndarray,
                     params_list: List[Dict[str, float]],
                     observed: np.ndarray,
                     temperature: np.ndarray = None,
                     spatial_data: Dict = None,
                     temperature_param: float = 2.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """★ 运行BMA集成预报
        
        Args:
            precip: 降水数组
            evap: 蒸发数组
            params_list: 各模型的参数字典列表
            observed: 实测流量（用于计算NSE）
            temperature: 温度数组（可选）
            spatial_data: 空间数据
            temperature_param: BMA温度参数
            
        Returns:
            (ensemble_flow, weights, metrics)
            - ensemble_flow: 集成后的流量
            - weights: 各模型权重
            - metrics: 评估指标
        """
        from src.bma_ensemble import (
            calc_bma_weights, 
            apply_bma_ensemble, 
            calc_bma_metrics
        )
        from src.hydro_calc import calc_nse
        
        simulated_list = []
        nse_list = []
        
        for runner, params in zip(self.runners, params_list):
            try:
                flow = runner.run(precip, evap, params, spatial_data, temperature)
                simulated_list.append(flow)
                
                nse = calc_nse(observed, flow)
                nse_list.append(nse)
            except Exception as e:
                print(f"[WARN] Model {runner.name} failed: {e}")
                simulated_list.append(np.zeros_like(observed))
                nse_list.append(-10.0)
        
        weights = calc_bma_weights(nse_list, temperature_param)
        ensemble = apply_bma_ensemble(simulated_list, weights)
        metrics = calc_bma_metrics(observed, simulated_list, weights)
        
        return ensemble, weights, metrics
    
    def get_weights_dict(self, weights: np.ndarray) -> Dict[str, float]:
        """获取模型权重字典"""
        return {name: float(w) for name, w in zip(self.names, weights)}
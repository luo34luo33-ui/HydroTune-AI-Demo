# -*- coding: utf-8 -*-
"""
Tank模型运行器
"""
import numpy as np
from typing import Dict, Optional
from .base_runner import BaseRunner
from ..loaders.tank_loader import TankLoader

class TankRunner(BaseRunner):
    """Tank模型运行器
    
    继承 BaseRunner 的所有功能：
    - run(): 基础模拟
    - simulate_with_routing(): 模拟 + 马斯京根演算
    - calibrate(): 参数率定
    - evaluate(): 计算评估指标
    """
    
    def __init__(self):
        super().__init__(None)
        self._name = "Tank水箱模型(完整版)"
        self.loader = TankLoader()
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None,
            warmup_steps: int = 0) -> np.ndarray:
        """运行Tank模型"""
        from src.models.registry import ModelRegistry
        
        model = ModelRegistry.get_model(self._name)
        
        full_params = model.default_params.copy()
        full_params.update(params)
        
        if spatial_data is None:
            spatial_data = {'area': 150.7944, 'del_t': 24.0}
        
        return model.run(precip, evap, full_params, spatial_data, temperature, warmup_steps)
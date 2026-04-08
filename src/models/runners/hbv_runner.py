# -*- coding: utf-8 -*-
"""
HBV模型运行器
"""
import numpy as np
from typing import Dict, Optional
from .base_runner import BaseRunner
from ..loaders.hbv_loader import HBVLoader

class HBVRunner(BaseRunner):
    """HBV模型运行器"""
    
    def __init__(self):
        super().__init__(None)
        self._name = "HBV模型"
        self.loader = HBVLoader()
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None,
            warmup_steps: int = 0) -> np.ndarray:
        """运行HBV模型"""
        from src.models.registry import ModelRegistry
        
        model = ModelRegistry.get_model(self._name)
        
        full_params = model.default_params.copy()
        full_params.update(params)
        
        if spatial_data is None:
            spatial_data = {
                'area': 150.7944,
                'monthly_temp': np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3]),
                'monthly_pet': np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])
            }
        
        return model.run(precip, evap, full_params, spatial_data, temperature, warmup_steps)
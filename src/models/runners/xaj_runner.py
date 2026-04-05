# -*- coding: utf-8 -*-
"""
XAJ模型运行器
"""
import numpy as np
from typing import Dict, Optional
from .base_runner import BaseRunner
from ..loaders.xaj_loader import XAJLoader

class XAJRunner(BaseRunner):
    """XAJ模型运行器"""
    
    def __init__(self):
        super().__init__(None)
        self._name = "新安江模型"
        self.loader = XAJLoader()
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None,
            warmup_steps: int = 0) -> np.ndarray:
        """运行XAJ模型"""
        from src.models.registry import ModelRegistry
        
        model = ModelRegistry.get_model(self._name)
        
        full_params = model.default_params.copy()
        full_params.update(params)
        
        if spatial_data is None:
            spatial_data = {'area': 150.7944}
        
        return model.run(precip, evap, full_params, spatial_data, temperature, warmup_steps)
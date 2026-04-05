# -*- coding: utf-8 -*-
"""
XAJ模型数据加载器
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_loader import BaseLoader

class XAJLoader(BaseLoader):
    """XAJ模型数据加载器
    
    将标准化DataFrame转换为XAJ模型所需的输入格式
    
    XAJ模型需要特殊的 p_and_e 格式：[n_timesteps, 1, 2]
    - 第一个通道 ([:,:,0]): 降水
    - 第二个通道 ([:,:,1]): 蒸发
    
    输出格式：
    - p_and_e: 降温和蒸发数组 [n, 1, 2]
    - spatial_data: 包含 area 的字典
    """
    
    name = "XAJLoader"
    
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        precip = self._ensure_array(df['precip'])
        evap = self._ensure_array(df.get('evap', np.zeros_like(precip)))
        
        n = len(precip)
        
        p_and_e = np.zeros((n, 1, 2))
        p_and_e[:, 0, 0] = precip
        p_and_e[:, 0, 1] = evap
        
        area = kwargs.get('area', 150.7944)
        
        spatial_data = {
            'area': area
        }
        
        return {
            'p_and_e': p_and_e,
            'spatial_data': spatial_data,
            'metadata': {
                'area': area,
                'n_timesteps': n
            }
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        required = ['p_and_e', 'spatial_data']
        for key in required:
            if key not in data or data[key] is None:
                return False
        
        p_and_e = data.get('p_and_e')
        if p_and_e is None or p_and_e.size == 0:
            return False
        
        return True
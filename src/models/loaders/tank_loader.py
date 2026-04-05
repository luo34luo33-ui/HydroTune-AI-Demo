# -*- coding: utf-8 -*-
"""
Tank模型数据加载器
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_loader import BaseLoader

class TankLoader(BaseLoader):
    """Tank模型数据加载器
    
    将标准化DataFrame转换为Tank模型所需的输入格式
    
    输出格式：
    - precip: 降水数组 (mm)
    - evap: 蒸发数组 (mm)
    - spatial_data: 包含 area 和 del_t 的字典
    """
    
    name = "TankLoader"
    
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        precip = self._ensure_array(df['precip'])
        evap = self._ensure_array(df.get('evap', np.zeros_like(precip)))
        
        # 获取空间数据
        area = kwargs.get('area', 150.7944)
        timestep = kwargs.get('timestep', 'daily')
        del_t = 1.0 if timestep == 'hourly' else 24.0
        
        spatial_data = {
            'area': area,
            'del_t': del_t,
            'timestep': timestep
        }
        
        # 检查是否有上游流量
        upstream = None
        if 'upstream' in df.columns:
            upstream = self._ensure_array(df['upstream'])
        
        return {
            'precip': precip,
            'evap': evap,
            'spatial_data': spatial_data,
            'upstream': upstream,
            'metadata': {
                'area': area,
                'timestep': timestep,
                'n_timesteps': len(precip)
            }
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        required = ['precip', 'evap', 'spatial_data']
        for key in required:
            if key not in data or data[key] is None:
                return False
        
        spatial = data.get('spatial_data', {})
        if 'area' not in spatial:
            return False
        
        return True
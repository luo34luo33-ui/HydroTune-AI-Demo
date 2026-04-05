# -*- coding: utf-8 -*-
"""
HBV模型数据加载器
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_loader import BaseLoader

class HBVLoader(BaseLoader):
    """HBV模型数据加载器
    
    将标准化DataFrame转换为HBV模型所需的输入格式
    
    输出格式：
    - precip: 降水数组 (mm)
    - evap: 蒸发数组 (mm)  
    - temperature: 温度数组 (°C)
    - spatial_data: 包含 area, monthly_temp, monthly_pet 的字典
    """
    
    name = "HBVLoader"
    
    DEFAULT_MONTHLY_TEMP = np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3])
    DEFAULT_MONTHLY_PET = np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])
    
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        precip = self._ensure_array(df['precip'])
        evap = self._ensure_array(df.get('evap', np.zeros_like(precip)))
        
        # 温度处理
        if 'temperature' in df.columns:
            temperature = self._ensure_array(df['temperature'])
        else:
            temperature = self._estimate_temperature(precip)
        
        # 获取空间数据
        area = kwargs.get('area', 150.7944)
        monthly_temp = kwargs.get('monthly_temp', self.DEFAULT_MONTHLY_TEMP)
        monthly_pet = kwargs.get('monthly_pet', self.DEFAULT_MONTHLY_PET)
        
        spatial_data = {
            'area': area,
            'monthly_temp': np.array(monthly_temp),
            'monthly_pet': np.array(monthly_pet)
        }
        
        return {
            'precip': precip,
            'evap': evap,
            'temperature': temperature,
            'spatial_data': spatial_data,
            'metadata': {
                'area': area,
                'n_timesteps': len(precip)
            }
        }
    
    def _estimate_temperature(self, precip: np.ndarray) -> np.ndarray:
        """从降水估算温度序列（当无实测温度时）"""
        n = len(precip)
        dates = np.arange(n)
        month_arr = (dates % 365) // 30
        month_arr = np.clip(month_arr, 0, 11)
        
        return np.array([
            self.DEFAULT_MONTHLY_TEMP[m] + np.random.randn() * 2 
            for m in month_arr
        ])
    
    def validate(self, data: Dict[str, Any]) -> bool:
        required = ['precip', 'evap', 'spatial_data']
        for key in required:
            if key not in data or data[key] is None:
                return False
        
        spatial = data.get('spatial_data', {})
        if 'area' not in spatial:
            return False
        
        return True
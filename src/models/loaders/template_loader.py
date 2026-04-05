# -*- coding: utf-8 -*-
"""
模板加载器 - 新模型接入的模板
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_loader import BaseLoader

class TemplateLoader(BaseLoader):
    """模板加载器 - 用于新模型接入
    
    使用方法：
    1. 继承此类
    2. 重写 load() 方法
    3. 实现自定义的数据转换逻辑
    
    示例：
        class MyModelLoader(TemplateLoader):
            name = "MyModelLoader"
            
            def load(self, df, **kwargs):
                # 自定义转换逻辑
                precip = df['precip'].values
                evap = df.get('evap', np.zeros_like(precip))
                
                return {
                    'precip': precip,
                    'evap': evap,
                    'spatial_data': {'area': kwargs.get('area', 150.0)},
                    'metadata': {}
                }
            
            def validate(self, data):
                return 'precip' in data and 'evap' in data
    """
    
    name = "TemplateLoader"
    
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """加载数据 - 子类需重写此方法
        
        Args:
            df: 标准化的DataFrame，必须包含 date, precip, evap 列
            **kwargs: 额外参数，如 area, timestep 等
            
        Returns:
            包含以下键的字典:
            - precip: np.ndarray
            - evap: np.ndarray  
            - spatial_data: dict
            - temperature: np.ndarray (可选)
            - metadata: dict
        """
        precip = self._ensure_array(df['precip'])
        evap = self._ensure_array(df.get('evap', np.zeros_like(precip)))
        
        return {
            'precip': precip,
            'evap': evap,
            'spatial_data': {},
            'metadata': {}
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证数据 - 子类需重写此方法"""
        return True
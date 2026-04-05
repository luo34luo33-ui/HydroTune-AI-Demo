# -*- coding: utf-8 -*-
"""
数据加载器基类 - 所有模型数据加载器的基类
新模型接入时，只需继承此类并实现 load() 方法
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

class BaseLoader(ABC):
    """数据加载器基类
    
    用途：将标准化的 DataFrame 转换为模型所需的输入格式
    
    标准输入格式 DataFrame 必须包含：
    - date: 日期列
    - precip: 降水 (mm)
    - evap: 蒸发 (mm)
    - flow: 流量 (m³/s)，可选（用于验证）
    - temperature: 温度 (°C)，可选
    - upstream: 上游流量 (m³/s)，可选
    """
    
    name: str = "BaseLoader"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """加载数据并转换为模型所需格式
        
        Args:
            df: 标准化的DataFrame
            **kwargs: 额外参数，如 area, timestep 等
            
        Returns:
            包含以下键的字典:
            - precip: np.ndarray
            - evap: np.ndarray  
            - spatial_data: dict
            - temperature: np.ndarray (可选)
            - metadata: dict (元数据)
        """
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证加载后的数据是否有效
        
        Args:
            data: load() 返回的数据字典
            
        Returns:
            True 表示有效，False 表示无效
        """
        pass
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return self.config.copy()
    
    def _ensure_array(self, arr) -> np.ndarray:
        """确保输入转换为numpy数组"""
        if arr is None:
            return np.array([])
        if isinstance(arr, pd.Series):
            return arr.values
        if isinstance(arr, np.ndarray):
            return arr
        return np.array(arr)
    
    def _check_columns(self, df: pd.DataFrame, required_cols: list) -> list:
        """检查必需的列是否存在
        
        Returns:
            缺失的列名列表
        """
        missing = []
        for col in required_cols:
            if col not in df.columns:
                missing.append(col)
        return missing


class SimpleLoader(BaseLoader):
    """简单加载器 - 仅需要 precip 和 evap"""
    
    name = "SimpleLoader"
    
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        precip = self._ensure_array(df['precip'])
        evap = self._ensure_array(df.get('evap', np.zeros_like(precip)))
        
        return {
            'precip': precip,
            'evap': evap,
            'spatial_data': {},
            'metadata': {}
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        required = ['precip', 'evap']
        for key in required:
            if key not in data or data[key] is None:
                return False
            if len(data[key]) == 0:
                return False
        return True
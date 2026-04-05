# -*- coding: utf-8 -*-
"""
数据加载器模块 - 统一管理数据输入转换
"""
from .base_loader import BaseLoader
from .tank_loader import TankLoader
from .hbv_loader import HBVLoader
from .xaj_loader import XAJLoader
from .template_loader import TemplateLoader

__all__ = [
    'BaseLoader',
    'TankLoader',
    'HBVLoader', 
    'XAJLoader',
    'TemplateLoader',
    'get_loader',
]

def get_loader(model_name: str) -> BaseLoader:
    """根据模型名称获取对应的Loader"""
    if 'Tank' in model_name:
        return TankLoader()
    elif 'HBV' in model_name:
        return HBVLoader()
    elif '新安江' in model_name or 'XAJ' in model_name:
        return XAJLoader()
    else:
        return TemplateLoader()
# -*- coding: utf-8 -*-
"""
模型运行器模块 - 封装率定算法、马斯京根演算、BMA集成等核心功能
"""
from .base_runner import BaseRunner, CalibrationResult
from .tank_runner import TankRunner
from .hbv_runner import HBVRunner
from .xaj_runner import XAJRunner
from .xaj_v3_runner import XAJV3Runner
from .bma_runner import BMARunner
from .template_runner import TemplateRunner

__all__ = [
    'BaseRunner',
    'CalibrationResult',
    'TankRunner',
    'HBVRunner',
    'XAJRunner',
    'XAJV3Runner',
    'BMARunner',
    'TemplateRunner',
    'get_runner',
]

def get_runner(model_name: str) -> 'BaseRunner':
    """根据模型名称获取对应的Runner"""
    if 'Tank' in model_name:
        return TankRunner()
    elif 'HBV' in model_name:
        return HBVRunner()
    elif 'V3' in model_name:
        return XAJV3Runner()
    elif '新安江' in model_name or 'XAJ' in model_name:
        return XAJRunner()
    else:
        return TemplateRunner()
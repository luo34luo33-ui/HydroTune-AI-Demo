# -*- coding: utf-8 -*-
"""
Tank水箱模型核心计算模块
代理到 tank-model-structured 子模块
"""
try:
    from tank_model_structured.src import tank_discharge, TankModel, MODEL_PARAMS, TANK_PARAMETER_ORDER
except ImportError:
    from .tank_generation import tank_discharge, TankModel
    from .tank_config import MODEL_PARAMS, TANK_PARAMETER_ORDER

__all__ = [
    'tank_discharge',
    'TankModel',
    'MODEL_PARAMS',
    'TANK_PARAMETER_ORDER',
]
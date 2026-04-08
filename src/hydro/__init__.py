# -*- coding: utf-8 -*-
"""
水文模型简化实现模块
从tests/models迁移，独立于tests目录
"""
from .xaj_simple import run_xaj_model, XAJ_PARAM_BOUNDS
from .tank_simple import run_tank_model, TANK_PARAM_BOUNDS
from .hbv_simple import run_hbv_model, HBV_PARAM_BOUNDS

try:
    from tank_model_structured.src import tank_discharge, TankModel, MODEL_PARAMS, TANK_PARAMETER_ORDER
except ImportError:
    from .tank_generation import tank_discharge, TankModel
    from .tank_config import MODEL_PARAMS, TANK_PARAMETER_ORDER

__all__ = [
    'run_xaj_model',
    'XAJ_PARAM_BOUNDS',
    'run_tank_model',
    'TANK_PARAM_BOUNDS',
    'run_hbv_model',
    'HBV_PARAM_BOUNDS',
    'tank_discharge',
    'TankModel',
    'MODEL_PARAMS',
    'TANK_PARAMETER_ORDER',
]
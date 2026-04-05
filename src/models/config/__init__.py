# -*- coding: utf-8 -*-
"""
模型配置管理模块
统一管理所有水文模型的参数边界、默认值和Schema定义
"""
from .model_schemas import (
    ModelSchema,
    ModelCategory,
    MODEL_SCHEMAS,
    register_schema,
    get_schema,
    list_schemas,
)
from .param_bounds import (
    TANK_PARAM_BOUNDS,
    HBV_PARAM_BOUNDS,
    XAJ_PARAM_BOUNDS,
    XAJ_V3_PARAM_BOUNDS,
    get_param_bounds,
)
from .default_params import (
    TANK_DEFAULT_PARAMS,
    HBV_DEFAULT_PARAMS,
    XAJ_DEFAULT_PARAMS,
    XAJ_V3_DEFAULT_PARAMS,
    get_default_params,
)

__all__ = [
    'ModelSchema',
    'ModelCategory',
    'MODEL_SCHEMAS',
    'register_schema',
    'get_schema',
    'list_schemas',
    'TANK_PARAM_BOUNDS',
    'HBV_PARAM_BOUNDS',
    'XAJ_PARAM_BOUNDS',
    'XAJ_V3_PARAM_BOUNDS',
    'get_param_bounds',
    'TANK_DEFAULT_PARAMS',
    'HBV_DEFAULT_PARAMS',
    'XAJ_DEFAULT_PARAMS',
    'XAJ_V3_DEFAULT_PARAMS',
    'get_default_params',
]
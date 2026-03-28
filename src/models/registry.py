"""
模型注册表
支持动态注册和调用水文模型
"""
from typing import Dict, List, Optional
from .base_model import BaseModel


class ModelRegistry:
    """模型注册表 - 支持动态注册和调用"""

    _models: Dict[str, BaseModel] = {}

    @classmethod
    def register(cls, model: BaseModel):
        """注册模型"""
        cls._models[model.name] = model

    @classmethod
    def get_model(cls, name: str) -> BaseModel:
        """获取模型实例"""
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"模型 '{name}' 未注册。可用模型: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有已注册模型名称"""
        return list(cls._models.keys())

    @classmethod
    def get_all_models(cls) -> Dict[str, BaseModel]:
        """获取所有模型实例"""
        return cls._models.copy()

    @classmethod
    def get_all_bounds(cls) -> Dict[str, Dict]:
        """获取所有模型的参数范围"""
        return {name: model.param_bounds for name, model in cls._models.items()}

    @classmethod
    def clear(cls):
        """清空所有注册的模型"""
        cls._models.clear()

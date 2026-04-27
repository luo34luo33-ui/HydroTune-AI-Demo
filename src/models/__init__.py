# HydroTune-AI Models Package
from .base_model import BaseModel
from .registry import ModelRegistry

# 简化版模型已废弃，请使用完整模型

# 注册Tank水箱模型 (完整版)
try:
    from .model_tank import TankModel
    ModelRegistry.register(TankModel())
    print("[HydroTune-AI] tank水箱模型已注册")
except ImportError as e:
    print(f"[HydroTune-AI] Tank水箱模型注册失败: {e}")

# 注册HBV模型 (完整版，需日尺度)
try:
    from .model_hbv import HBVModelAdapter
    ModelRegistry.register(HBVModelAdapter())
    print("[HydroTune-AI] HBV模型已注册")
except ImportError as e:
    print(f"[HydroTune-AI] HBV模型注册失败: {e}")

# 注册新安江模型V2
try:
    from .model_xaj_v2 import XAJModelV2
    ModelRegistry.register(XAJModelV2())
    print("[HydroTune-AI] 新安江模型V2已注册")
except ImportError as e:
    print(f"[HydroTune-AI] 新安江模型V2注册失败: {e}")

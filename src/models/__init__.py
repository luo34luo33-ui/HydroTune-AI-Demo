# HydroTune-AI Models Package
from .base_model import BaseModel
from .registry import ModelRegistry

# 目前使用示例模型
from .example_model import SimpleTankModel, LinearReservoirModel, HBVLikeModel

# 注册示例模型
ModelRegistry.register(SimpleTankModel())
ModelRegistry.register(LinearReservoirModel())
ModelRegistry.register(HBVLikeModel())

# 注册Tank水箱模型 (需要先于XAJ导入以避免sys.path冲突)
try:
    from .model_tank import TankModel
    ModelRegistry.register(TankModel())
    print("[HydroTune-AI] Tank水箱模型已注册")
except ImportError as e:
    print(f"[HydroTune-AI] Tank水箱模型注册失败: {e}")

# 注册HBV模型 (完整版，需日尺度)
try:
    from .model_hbv import HBVModelAdapter
    ModelRegistry.register(HBVModelAdapter())
    print("[HydroTune-AI] HBV模型(完整版)已注册")
except ImportError as e:
    print(f"[HydroTune-AI] HBV模型(完整版)注册失败: {e}")

# 注册新安江模型
try:
    from .model_xaj import XAJModel
    ModelRegistry.register(XAJModel())
    print("[HydroTune-AI] 新安江模型已注册")
except ImportError as e:
    print(f"[HydroTune-AI] 新安江模型注册失败: {e}")

# 注册新安江模型V2
try:
    from .model_xaj_v2 import XAJModelV2
    ModelRegistry.register(XAJModelV2())
    print("[HydroTune-AI] 新安江模型V2已注册")
except ImportError as e:
    print(f"[HydroTune-AI] 新安江模型V2注册失败: {e}")

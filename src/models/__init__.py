# Hydromind Models Package
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
    print("[Hydromind] Tank水箱模型已注册")
except ImportError as e:
    print(f"[Hydromind] Tank水箱模型注册失败: {e}")

# 注册HBV模型 (完整版，需日尺度)
try:
    from .model_hbv import HBVModelAdapter
    ModelRegistry.register(HBVModelAdapter())
    print("[Hydromind] HBV模型(完整版)已注册")
except ImportError as e:
    print(f"[Hydromind] HBV模型(完整版)注册失败: {e}")

# 注册新安江模型
try:
    from .model_xaj import XAJModel
    ModelRegistry.register(XAJModel())
    print("[Hydromind] 新安江模型已注册")
except ImportError as e:
    print(f"[Hydromind] 新安江模型注册失败: {e}")

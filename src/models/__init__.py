# HydroTune-AI Models Package
from .base_model import BaseModel
from .registry import ModelRegistry

# 保留既有模型名称及参数；五参数 HBV 作为独立模型提供。

import traceback

# 注册Tank水箱模型 (完整版)
try:
    from .model_tank import TankModel
    ModelRegistry.register(TankModel())
    print("[HydroTune-AI] Tank Model registered")
except Exception as e:
    print(f"[HydroTune-AI] Tank Model registration failed: {e}")
    traceback.print_exc()

# 注册HBV模型 (完整版，需日尺度)
try:
    from .model_hbv import HBVModelAdapter
    ModelRegistry.register(HBVModelAdapter())
    print("[HydroTune-AI] HBV Model registered")
except Exception as e:
    print(f"[HydroTune-AI] HBV Model registration failed: {e}")
    traceback.print_exc()

# 注册新安江模型V2
try:
    from .model_xaj_v2 import XAJModelV2
    ModelRegistry.register(XAJModelV2())
    print("[HydroTune-AI] XAJ Model registered")
except Exception as e:
    print(f"[HydroTune-AI] XAJ Model registration failed: {e}")
    traceback.print_exc()

# 接入已有五参数 HBV，日/小时尺度均按实际计算步换算流量。
from .model_hbv_simple import HBVSimpleModel
ModelRegistry.register(HBVSimpleModel())

# HydroTune-AI Models Package
from .base_model import BaseModel
from .registry import ModelRegistry

# 平台统一使用支持日与小时尺度的五参数 HBV。

import traceback

# 注册Tank水箱模型 (完整版)
try:
    from .model_tank import TankModel
    ModelRegistry.register(TankModel())
    print("[HydroTune-AI] Tank Model registered")
except Exception as e:
    print(f"[HydroTune-AI] Tank Model registration failed: {e}")
    traceback.print_exc()

# 注册平台唯一的 HBV 模型（日与小时尺度）
try:
    from .model_hbv_simple import HBVSimpleModel
    ModelRegistry.register(HBVSimpleModel())
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

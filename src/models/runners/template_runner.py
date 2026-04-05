# -*- coding: utf-8 -*-
"""
模板运行器 - 新模型接入的模板
"""
import numpy as np
from typing import Dict, Optional
from .base_runner import BaseRunner

class TemplateRunner(BaseRunner):
    """模板运行器 - 用于新模型接入
    
    使用方法：
    1. 继承此类
    2. 重写 run() 方法
    3. 自动获得以下功能：
       - simulate_with_routing(): 模拟 + 马斯京根演算
       - calibrate(): 参数率定
       - evaluate(): 计算评估指标
    
    示例：
        class MyModelRunner(TemplateRunner):
            def __init__(self):
                super().__init__(None)
                self._name = "我的模型"
            
            def run(self, precip, evap, params, spatial_data, temperature, warmup_steps):
                # 调用你的模型核心函数
                result = my_model_core(precip, evap, params)
                return result
    """
    
    def __init__(self):
        super().__init__(None)
        self._name = "TemplateRunner"
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None,
            warmup_steps: int = 0) -> np.ndarray:
        """运行模型 - 子类需重写此方法"""
        raise NotImplementedError("子类必须实现 run() 方法")
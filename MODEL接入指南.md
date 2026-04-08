# -*- coding: utf-8 -*-
"""
流域水文模型接入指南
==================

本文档说明如何将新的结构化水文模型接入到 HydroTune-AI 系统中。

## 一、框架概述

```
┌─────────────────────────────────────────────────────────────┐
│                    插件式模型架构                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  Loaders    │───►│  Runners    │───►│   BMA集成   │    │
│  │  (数据转换) │    │  (运行器)   │    │  (集成预报) │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  模型适配器 (Adapters) + 统一接口 (BaseModel)           ││
│  └─────────────────────────────────────────────────────────┘│
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  结构化模型子模块 (Tank/HBV/XAJ/新模型)                  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 二、接入流程（仅4步）

### Step 1: 创建模型Schema配置

在 `src/models/config/model_schemas.py` 中添加新模型配置：

```python
from .model_schemas import ModelSchema, ModelCategory, register_schema

# 定义模型Schema
new_model_schema = ModelSchema(
    id='new_model_v1',
    name='新模型名称',
    category=ModelCategory.CUSTOM,
    description='模型描述',
    
    # 模型能力
    supports_hourly=True/False,
    requires_temperature=True/False,
    requires_spatial=True/False,
    supports_upstream_routing=True/False,
    
    # 参数配置
    param_bounds={
        'param1': (min, max),
        'param2': (min, max),
        # ...
    },
    param_defaults={
        'param1': default_value,
        # ...
    },
    param_descriptions={
        'param1': '参数描述',
        # ...
    },
    
    # 子模块信息
    module_path='new-model-structured',
    core_function='core_function_name',
)

# 注册
register_schema(new_model_schema)
```

### Step 2: 创建数据加载器（Loader）

在 `src/models/loaders/` 目录下创建加载器：

```python
# src/models/loaders/new_model_loader.py
from .base_loader import BaseLoader
import numpy as np
import pandas as pd
from typing import Dict, Any

class NewModelLoader(BaseLoader):
    """新模型数据加载器"""
    
    name = "NewModelLoader"
    
    def load(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """加载并转换数据"""
        precip = df['precip'].values
        evap = df.get('evap', np.zeros_like(precip))
        
        # 根据需要处理空间数据
        area = kwargs.get('area', 150.0)
        
        return {
            'precip': precip,
            'evap': evap,
            'spatial_data': {'area': area},
            'metadata': {}
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        return 'precip' in data and 'evap' in data
```

### Step 3: 创建模型运行器（Runner）

在 `src/models/runners/` 目录下创建运行器：

```python
# src/models/runners/new_model_runner.py
from .base_runner import BaseRunner
import numpy as np
from typing import Dict, Optional

class NewModelRunner(BaseRunner):
    """新模型运行器"""
    
    def __init__(self):
        super().__init__(None)
        self._name = "新模型名称"
    
    def run(self, precip: np.ndarray, evap: np.ndarray,
            params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None,
            warmup_steps: int = 0) -> np.ndarray:
        """运行模型核心逻辑"""
        from src.models.registry import ModelRegistry
        
        # 获取模型
        model = ModelRegistry.get_model(self._name)
        
        # 合并默认参数
        full_params = model.default_params.copy()
        full_params.update(params)
        
        # 运行模型
        return model.run(precip, evap, full_params, spatial_data, temperature, warmup_steps)
```

### Step 4: 注册模型

在 `src/models/__init__.py` 中注册：

```python
# 注册新模型
from .model_new import NewModelAdapter
ModelRegistry.register(NewModelAdapter())
```

## 三、自动获得的功能

接入后，新模型自动获得以下功能支持：

### 1. 率定算法支持

```python
runner = NewModelRunner()

# 调用 calibrate() 自动使用所有优化算法
result = runner.calibrate(
    precip=precip,
    evap=evap,
    observed_flow=observed_flow,
    algorithm='two_stage',  # 或 'pso', 'ga', 'sce', 'de'
    max_iter=30
)

print(f"最优参数: {result.best_params}")
print(f"NSE: {result.best_nse}")
```

### 2. 马斯京根上游演算

```python
# 模拟 + 上游流量演算
flow = runner.simulate_with_routing(
    precip=precip,
    evap=evap,
    params=params,
    upstream_flow=upstream_flow,      # 上游流量
    routing_params={'k_routing': 2.5, 'x_routing': 0.25}  # 马斯京根参数
)
```

### 3. BMA集成预报

```python
from src.models.runners.bma_runner import BMARunner

# 创建多个模型的runner列表
runners = [TankRunner(), HBVRunner(), XAJRunner(), NewRunner()]

# 创建BMA运行器
bma_runner = BMARunner(runners)

# 运行BMA集成
ensemble, weights, metrics = bma_runner.run_ensemble(
    precip=precip,
    evap=evap,
    params_list=[tank_params, hbv_params, xaj_params, new_params],
    observed=observed_flow
)

print(f"集成流量: {ensemble}")
print(f"模型权重: {bma_runner.get_weights_dict(weights)}")
print(f"BMA NSE: {metrics['nse']}")
```

### 4. 评估指标计算

```python
metrics = runner.evaluate(observed, simulated)

print(f"NSE: {metrics['nse']}")
print(f"RMSE: {metrics['rmse']}")
print(f"MAE: {metrics['mae']}")
print(f"PBIAS: {metrics['pbias']}")
print(f"KGE: {metrics['kge']}")
```

## 四、接口规范

### 4.1 模型输入格式

标准化DataFrame必须包含以下列：
- `date`: 日期
- `precip`: 降水 (mm)
- `evap`: 蒸发 (mm)

可选列：
- `flow`: 流量 (m³/s)
- `temperature`: 温度 (°C)
- `upstream`: 上游流量 (m³/s)

### 4.2 模型输出要求

`run()` 方法必须返回：
- `np.ndarray`: 流量数组 (m³/s)
- Shape: (n_timesteps,)

### 4.3 参数约束

通过Schema配置参数边界后，系统自动验证：
```python
if model.validate_params(params):
    # 参数有效
else:
    # 参数无效
```

## 五、示例：接入SimpleRain模型

假设你有一个简单的rain-runoff模型 `simple-rain-model/`:

### Step 1: 添加Schema

```python
# src/models/config/model_schemas.py
register_schema(ModelSchema(
    id='simple_rain_v1',
    name='SimpleRain模型',
    category=ModelCategory.CUSTOM,
    supports_hourly=True,
    requires_temperature=False,
    requires_spatial=False,
    supports_upstream_routing=False,
    param_bounds={
        'alpha': (0.1, 0.9),
        'beta': (0.01, 0.3),
    },
    param_defaults={
        'alpha': 0.5,
        'beta': 0.1,
    },
    module_path='simple-rain-model',
    core_function='run_model',
))
```

### Step 2: 创建Loader

```python
# src/models/loaders/simple_rain_loader.py
class SimpleRainLoader(BaseLoader):
    name = "SimpleRainLoader"
    
    def load(self, df, **kwargs):
        return {
            'precip': df['precip'].values,
            'evap': df.get('evap', np.zeros(len(df['precip']))).values,
            'spatial_data': {},
            'metadata': {}
        }
    
    def validate(self, data):
        return 'precip' in data
```

### Step 3: 创建Runner

```python
# src/models/runners/simple_rain_runner.py
class SimpleRainRunner(BaseRunner):
    def __init__(self):
        super().__init__(None)
        self._name = "SimpleRain模型"
    
    def run(self, precip, evap, params, **kwargs):
        # 调用你的模型核心函数
        return your_model_core(precip, evap, params)
```

完成！系统自动支持率定、BMA集成等功能。

## 六、文件结构参考

```
src/models/
├── config/                    # 配置层
│   ├── model_schemas.py       # 模型Schema定义
│   ├── param_bounds.py        # 参数边界
│   └── default_params.py      # 默认参数
│
├── loaders/                   # 数据加载层
│   ├── base_loader.py         # 加载器基类
│   ├── tank_loader.py         # Tank加载器
│   ├── hbv_loader.py          # HBV加载器
│   ├── xaj_loader.py          # XAJ加载器
│   └── template_loader.py     # 模板加载器
│
├── runners/                   # 模型运行层
│   ├── base_runner.py         # 运行器基类
│   ├── tank_runner.py         # Tank运行器
│   ├── hbv_runner.py          # HBV运行器
│   ├── xaj_runner.py          # XAJ运行器
│   ├── bma_runner.py          # BMA集成运行器
│   └── template_runner.py     # 模板运行器
│
├── adapters/                  # 模型适配器层
├── base_model.py              # 抽象基类
└── registry.py                # 模型注册表
```

---

**注意事项**：
1. 所有参数必须通过 `param_bounds` 配置边界
2. 模型输出单位必须是 m³/s
3. 如有特殊约束条件，在Schema中设置 `param_constraints`
4. 如需额外空间数据，在Loader中处理

如有问题，请查看现有模型（Tank/HBV/XAJ）的实现作为参考。
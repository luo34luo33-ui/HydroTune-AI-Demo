# Hydromind-Demo

水文模型集成演示项目，支持多种概念性水文模型的统一调用。

## 模型推荐

### 生产推荐使用

| 模型 | 类名 | 描述 | 参数数量 | 适用场景 |
|------|------|------|---------|---------|
| **Tank水箱模型** | `TankModel` | 日本学者Sugawara提出的多层水箱模型 | 16个 | 湿润地区流域模拟 |
| **HBV模型** | `HBVModelAdapter` | 瑞典水文气象局开发的概念性水文模型 | 10个 | 寒温带/有积雪流域 |
| **新安江模型** | `XAJModel` | 中国学者提出的三水源模型 | 15个 | 中国湿润地区 |

### 演示用模型 (deprecated)

| 模型 | 类名 | 说明 |
|------|------|------|
| ~~水箱模型~~ | `SimpleTankModel` | 已废弃，请使用Tank水箱模型 |
| 线性水库模型 | `LinearReservoirModel` | 独立汇流模型，仍可使用 |
| ~~HBV简化模型~~ | `HBVLikeModel` | 已废弃，请使用HBV模型 | |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from src.models import ModelRegistry
import numpy as np

# 查看所有可用模型
print(ModelRegistry.list_models())

# 准备输入数据
precip = np.random.rand(365) * 10  # 日降水 (mm)
evap = np.random.rand(365) * 3     # 日蒸散发 (mm)

# 获取模型并运行（完整调用）
model = ModelRegistry.get_model("Tank水箱模型")
flow = model.run(precip, evap, {}, {'area': 410.0, 'del_t': 24.0})
```

### 简化调用 (推荐)

所有推荐模型支持简化调用，自动使用默认参数：

```python
from src.models import ModelRegistry
import numpy as np

precip = np.random.rand(365) * 10
evap = np.random.rand(365) * 3

# Tank水箱模型 - 仅需precip和evap
model = ModelRegistry.get_model("Tank水箱模型")
flow = model.run(precip, evap, {})  # 自动使用默认spatial_data

# HBV模型 - 仅需precip和evap
model = ModelRegistry.get_model("HBV模型")
flow = model.run(precip, evap, {})  # 自动估算温度和月数据

# 新安江模型 - 仅需precip和evap
model = ModelRegistry.get_model("新安江模型")
flow = model.run(precip, evap, {})
```

## 模型接口规范

所有模型继承自 `BaseModel` 抽象基类，遵循统一接口：

```python
from src.models.base_model import BaseModel

class BaseModel(ABC):
    @property
    def name(self) -> str:
        """模型名称"""
        pass

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """参数取值范围"""
        pass

    def run(
        self,
        precip: np.ndarray,           # 降水序列 (mm)
        evap: np.ndarray,            # 蒸发序列 (mm)
        params: Dict[str, float],     # 模型参数
        spatial_data: Dict = None,    # 空间数据
        temperature: np.ndarray = None # 温度序列 (°C)
    ) -> np.ndarray:
        """运行模型，返回流量序列 (m³/s)"""
        pass
```

### 各模型参数说明

#### Tank水箱模型
```python
# 完整调用
spatial_data = {
    'area': 410.0,    # 流域面积 (km²)
    'del_t': 24.0     # 时间步长 (小时)，默认24
}
flow = model.run(precip, evap, params, spatial_data)

# 简化调用 (自动使用默认值)
flow = model.run(precip, evap, params)
```

#### HBV模型
```python
# 完整调用
spatial_data = {
    'area': 410.0,                          # 流域面积 (km²)
    'monthly_temp': np.array([...]),       # 月平均温度 (°C)，长度12
    'monthly_pet': np.array([...]),        # 月平均PET (mm/day)，长度12
}
temperature = np.array([...])  # 日温度序列 (°C)
flow = model.run(precip, evap, params, spatial_data, temperature)

# 简化调用 (自动估算)
flow = model.run(precip, evap, params)
```

#### 新安江模型
```python
# 只需基本输入
flow = model.run(precip, evap, params)
```

## 项目结构

```
Hydromind-Demo/
├── src/
│   ├── models/                 # 模型接口层
│   │   ├── base_model.py       # 抽象基类
│   │   ├── registry.py         # 模型注册表
│   │   ├── model_tank.py       # Tank模型适配器
│   │   ├── model_hbv.py        # HBV模型适配器
│   │   └── model_xaj.py        # 新安江模型适配器
│   ├── hydro_calc.py           # 水文计算
│   ├── agent/                  # 智能体
│   └── data/                   # 数据处理
├── tank-model-structured/      # Tank模型源码
├── HBV_model_structured/       # HBV模型源码
├── XAJ-model-structured/        # 新安江模型源码
└── demo_data/                  # 示例数据
```

## 添加新模型

参考 `src/models/model_xaj.py` 的实现方式：

1. 创建模型适配器类，继承 `BaseModel`
2. 实现 `name` 和 `param_bounds` 属性
3. 实现 `run()` 方法
4. 在 `src/models/__init__.py` 中注册

```python
from .base_model import BaseModel

class MyModel(BaseModel):
    @property
    def name(self) -> str:
        return "我的模型"
    
    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {'param1': (0, 10), 'param2': (0, 1)}
    
    def run(self, precip, evap, params, spatial_data=None, temperature=None):
        # 实现模型逻辑
        return flow
```

## 参考资料

- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - 架构设计规范
- 各子项目的 README 文档

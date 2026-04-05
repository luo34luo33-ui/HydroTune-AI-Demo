# HydroTune-AI 开发指南

本文档为 Agentic 编码代理提供开发规范和操作指南。

---

## 一、项目概述

HydroTune-AI 是一个水文模型智能率定系统，支持多模型集成（Tank、HBV、新安江）、多场次洪水率定、数据预分析、参数优化与智能报告。

### 项目结构

```
HydroTune-AI/
├── src/                        # 主应用代码
│   ├── models/                 # 模型接口层
│   │   ├── base_model.py       # 抽象基类
│   │   ├── registry.py         # 模型注册表
│   │   └── model_*.py          # 各模型适配器
│   ├── data/                   # 数据处理模块
│   ├── agent/                  # 工作流代理
│   ├── hydro_calc.py           # 水文计算与率定
│   ├── data_agent.py           # 数据清洗与场次识别
│   └── data_preanalysis.py     # 数据预分析
├── tank-model-structured/      # Tank模型子模块
├── HBV_model_structured/       # HBV模型子模块
└── XAJ-model-structured/       # 新安江模型子模块
```

---

## 二、构建与测试命令

### 2.1 安装依赖

```bash
# 主项目依赖
pip install -r requirements.txt

# 子模块依赖
cd tank-model-structured && pip install -r requirements.txt
```

### 2.2 运行应用

```bash
streamlit run app.py
```

### 2.3 测试命令

```bash
# 运行所有测试（根目录和子模块）
pytest

# 运行单个子模块测试
cd tank-model-structured && pytest
cd HBV_model_structured && pytest

# 运行单个测试文件
pytest tests/test_core_generation.py

# 运行单个测试函数
pytest tests/test_core_generation.py::test_tank_discharge_shape -v
```

### 2.4 代码检查

项目使用 flake8 进行代码检查，配置如下：

```bash
# 严格模式：检查语法错误和未定义名称
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 警告模式：最大行长度127，最大复杂度10
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### 2.5 子模块初始化

```bash
git submodule update --init --recursive
```

---

## 三、代码风格规范

### 3.1 通用规范

- **Python版本**: >= 3.6
- **编码**: UTF-8 (`# -*- coding: utf-8 -*-`)
- **语言**: 中文注释优先，保持代码可读性

### 3.2 导入规范

```python
# 标准库
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from abc import ABC, abstractmethod

# 第三方库
from scipy.optimize import differential_evolution, dual_annealing
import matplotlib.pyplot as plt

# 本地模块（相对导入）
from .models.registry import ModelRegistry
from ..data.loader import DataLoader
```

### 3.3 命名规范

| 类型 | 命名方式 | 示例 |
|------|----------|------|
| 模块/包 | 小写字母+下划线 | `data_preanalysis.py`, `hydro_calc.py` |
| 类 | 大驼峰 | `BaseModel`, `TankModelAdapter` |
| 函数/方法 | 小写下划线 | `calc_nse()`, `run_model()` |
| 常量 | 全大写+下划线 | `MAX_ITER`, `DEFAULT_AREA` |
| 私有变量 | 单下划线前缀 | `_internal_state` |
| 类型注解 | 大驼峰（泛型除外） | `Dict[str, Tuple[float, float]]` |

### 3.4 函数文档规范

```python
def function_name(param1: str, param2: int, **kwargs) -> Dict[str, float]:
    """简短描述。
    
    详细描述（可选）。
    
    Args:
        param1: 参数1的描述
        param2: 参数2的描述
        **kwargs: 其他参数
        
    Returns:
        返回值的描述
        
    Raises:
        ValueError: 异常情况描述
        
    Example:
        >>> result = function_name("test", 10)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

### 3.5 类型注解要求

- 所有公共函数必须有类型注解
- 数组类型使用 `np.ndarray`
- 字典使用 `Dict[KeyType, ValueType]`
- 可选参数使用 `Optional[Type]`

```python
# 正确示例
def calc_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    ...

def run_model(
    precip: np.ndarray,
    evap: np.ndarray,
    params: Dict[str, float],
    spatial_data: Optional[Dict] = None,
) -> np.ndarray:
    ...
```

### 3.6 格式化规范

- **缩进**: 4空格
- **最大行长度**: 127字符
- **import顺序**: 标准库 > 第三方 > 本地模块（用空行分隔）
- **运算符前后**: 加空格 `a + b`，但 `a*b` 不加
- **函数参数**: 换行对齐

```python
# 函数参数换行
def long_function_name(
    param1: str,
    param2: int,
    param3: float,
    param4: Dict,
) -> Dict[str, float]:
    ...
```

### 3.7 错误处理规范

```python
# 使用具体异常类型
try:
    result = model.run(precip, evap, params, spatial_data)
except ValueError as e:
    logger.error(f"参数错误: {e}")
    raise
except Exception as e:
    logger.error(f"模型运行失败: {e}")
    return np.nan  # 或返回默认值

# 避免 bare except
# 错误: except:
# 正确: except Exception:
```

### 3.8 数值计算规范

```python
# 避免除零，添加小量 epsilon
denominator = np.maximum(np.sum((obs - obs_mean) ** 2), 1e-10)

# 确保非负
flow = np.maximum(flow, 0)

# 限制范围
w = np.clip(w, 0, wm)

# 处理 NaN
mask = ~(np.isnan(observed) | np.isnan(simulated))
obs, sim = observed[mask], simulated[mask]
```

---

## 四、模型开发规范

### 4.1 新增模型接口

所有模型必须继承 `BaseModel` 并实现以下接口：

```python
from src.models.base_model import BaseModel
from typing import Dict, Tuple
import numpy as np

class MyModel(BaseModel):
    @property
    def name(self) -> str:
        return "我的模型"
    
    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'k1': (0.01, 0.3),
            'k2': (0.001, 0.05),
            'c': (0.01, 0.3),
        }
    
    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 实现模型逻辑
        return flow
```

### 4.2 注册模型

在 `src/models/__init__.py` 中注册：

```python
from .my_model import MyModel
ModelRegistry.register(MyModel())
```

### 4.3 参数约束

- 物理意义明确的参数名称
- 取值范围基于流域物理特性
- 提供默认值（范围中值）

---

## 五、测试规范

### 5.1 单元测试要求

- 每个核心模块必须有独立测试
- 测试文件放在 `tests/` 目录
- 使用 `pytest` 框架

```python
def test_function_basic():
    """基本功能测试"""
    result = function_name(input1, input2)
    assert result == expected_value

def test_function_edge_cases():
    """边界情况测试"""
    # 测试零值、负值、空数组等
    result = function_name(zero_input)
    assert result >= 0
```

### 5.2 集成测试

- 测试完整模型运行流程
- 验证输出形状正确
- 验证无 NaN 值
- 验证结果非负

### 5.3 对比测试

新增算法需与原有实现对比验证：

```python
def test_compare_with_original():
    result_original = original_func(data, params)
    result_new = new_func(data, params)
    diff = np.abs(result_original - result_new)
    assert np.max(diff) < 1e-10
```

---

## 六、Git 提交规范

### 分支策略

- `main`: 稳定版本
- `develop`: 开发版本
- `feature/*`: 新功能
- `bugfix/*`: Bug修复

### 提交信息格式

```
<类型>: <简短描述>

<详细描述>

类型：feat, fix, docs, refactor, test
```

---

## 七、注意事项

1. **子模块处理**: 提交前确认子模块已正确初始化
2. **数据验证**: 所有输入数据必须经过验证
3. **单位一致**: 注意单位转换（mm ↔ m³/s）
4. **预热期**: 模型运行需考虑预热期初始状态
5. **NSE异常值**: NSE返回-9999表示计算无效

---

## 八、参考文档

- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - 架构设计规范
- 各子项目 README 文档
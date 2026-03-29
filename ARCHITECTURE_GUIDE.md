# 概念水文模型项目架构规范

## 一、项目架构设计原则

### 1.1 模块化原则
- **单一职责**：每个模块只负责一个明确的功能
- **低耦合**：模块间通过明确的接口交互，减少相互依赖
- **高内聚**：相关功能集中在同一模块内

### 1.2 可扩展性原则
- **开闭原则**：对扩展开放，对修改关闭
- **接口标准化**：统一的数据格式和函数接口
- **插件化设计**：便于添加新的算法或方法

### 1.3 可验证性原则
- **基准测试**：保留原始代码作为验证基准
- **对比验证**：新旧实现结果必须一致
- **单元测试**：每个模块应有独立的测试

---

## 二、模型集成层设计 (HydroTune-AI统一接口)

### 2.1 设计目标

通过统一的抽象基类 `BaseModel` 和模型注册机制 `ModelRegistry`，实现多种水文模型的标准化接入。

### 2.2 核心组件

#### BaseModel 抽象基类
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional

class BaseModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass

    @property
    def model_type(self) -> str:
        """模型类型: 'lumped' 或 'distributed'"""
        return "lumped"

    @property
    @abstractmethod
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """参数取值范围"""
        pass

    @property
    def default_params(self) -> Dict[str, float]:
        """默认参数值"""
        return {k: (v[0] + v[1]) / 2 for k, v in self.param_bounds.items()}

    @abstractmethod
    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        运行模型

        Args:
            precip: 降水序列 (mm)
            evap: 蒸发序列 (mm)
            params: 模型参数字典
            spatial_data: 空间数据（如流域面积等）
            temperature: 温度序列 (°C)，部分模型需要

        Returns:
            模拟流量序列 (m³/s)
        """
        pass
```

#### ModelRegistry 模型注册表
```python
class ModelRegistry:
    _models: Dict[str, BaseModel] = {}

    @classmethod
    def register(cls, model: BaseModel):
        """注册模型"""
        cls._models[model.name] = model

    @classmethod
    def get_model(cls, name: str) -> BaseModel:
        """获取模型实例"""
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有已注册模型"""
        return list(cls._models.keys())
```

### 2.3 模型适配器模式

对于已有的独立模型项目，通过创建适配器类将其接入统一接口：

```python
class TankModelAdapter(BaseModel):
    """Tank水箱模型适配器"""

    @property
    def name(self) -> str:
        return "Tank水箱模型"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        # 从原生配置转换为统一格式
        return {...}

    def run(self, precip, evap, params, spatial_data=None, temperature=None):
        # 调用原生模型，处理数据格式转换
        area = spatial_data.get('area')
        # ... 调用 tank_discharge()
        return flow
```

### 2.4 统一调用示例

```python
from src.models import ModelRegistry

# 列出所有模型
models = ModelRegistry.list_models()
print(models)

# 获取并运行模型
model = ModelRegistry.get_model("Tank水箱模型")
flow = model.run(precip, evap, params, spatial_data={'area': 410.0})
```

---

## 三、推荐项目结构

```
hydrological_model/
│
├── README.md                   # 项目说明文档
├── LICENSE                     # 许可证
├── requirements.txt            # 依赖包列表
├── setup.py                    # 安装脚本（可选）
│
├── config/                     # 配置文件目录
│   ├── __init__.py
│   ├── model_config.py         # 模型参数配置
│   ├── default_params.json     # 默认参数值
│   └── param_ranges.json       # 参数取值范围
│
├── core/                       # 核心算法模块
│   ├── __init__.py
│   ├── generation.py           # 产流模块
│   ├── routing.py              # 汇流模块
│   ├── evapotranspiration.py   # 蒸散发模块（可选单独列出）
│   └── snow.py                 # 融雪模块（寒冷地区需要）
│
├── data/                       # 数据处理模块
│   ├── __init__.py
│   ├── loader.py               # 数据加载
│   ├── preprocessor.py         # 数据预处理
│   ├── validator.py            # 数据验证
│   └── generator.py            # 合成数据生成
│
├── calibration/                # 参数率定模块
│   ├── __init__.py
│   ├── optimizer.py            # 优化算法
│   ├── objective.py            # 目标函数
│   └── sensitivity.py          # 敏感性分析
│
├── evaluation/                 # 模型评估模块
│   ├── __init__.py
│   ├── metrics.py              # 性能指标计算
│   └── statistics.py           # 统计分析
│
├── visualization/              # 可视化模块
│   ├── __init__.py
│   ├── plot_hydrograph.py      # 水文过程线
│   ├── plot_comparison.py      # 对比图
│   └── plot_analysis.py        # 分析图
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── logger.py               # 日志记录
│   ├── io_utils.py             # 文件操作
│   └── math_utils.py           # 数学工具
│
├── main.py                     # 主程序入口
├── run_calibration.py          # 参数率定脚本
├── run_validation.py           # 模型验证脚本
│
├── example_data/               # 示例数据
│   ├── daily_data.csv          # 日尺度数据
│   ├── hourly_data.csv         # 小时尺度数据
│   └── parameters.json         # 参数配置示例
│
├── tests/                      # 测试目录
│   ├── __init__.py
│   ├── test_generation.py      # 产流模块测试
│   ├── test_routing.py         # 汇流模块测试
│   └── test_integration.py     # 集成测试
│
└── docs/                       # 文档目录
    ├── theory.md               # 理论说明
    ├── manual.md               # 使用手册
    └── api.md                  # API文档
```

---

## 四、核心模块设计要点

### 3.1 配置模块 (config/)

**必须包含的内容**：
1. **参数定义**：所有模型参数的名称、符号、单位、物理意义
2. **默认值**：经过验证的参数默认值
3. **取值范围**：用于参数率定的上下限
4. **约束规则**：参数间的约束关系（如 ki + kg < 1）

**设计示例**：
```python
MODEL_PARAMS = {
    'k': {
        'default': 0.8,
        'min': 0.5,
        'max': 1.5,
        'unit': '-',
        'description': '蒸散发系数'
    },
    # ... 其他参数
}
```

### 4.2 产流模块 (core/generation.py)

**核心函数**：
```python
def calculate_evaporation(precip, pet, wu, wl, wd, params):
    """
    计算三层蒸散发
    
    Args:
        precip: 降水量
        pet: 潜在蒸散发
        wu, wl, wd: 上/下/深层土壤含水量
        params: 模型参数
    
    Returns:
        eu, el, ed: 各层实际蒸散发
    """
    pass

def calculate_runoff(precip, evap, w, params):
    """
    计算产流量
    
    Args:
        precip: 降水量
        evap: 蒸散发量
        w: 土壤含水量
        params: 模型参数
    
    Returns:
        r: 产流量
        w_new: 更新后的土壤含水量
    """
    pass
```

**注意事项**：
- 输入输出使用NumPy数组，便于向量化计算
- 处理边界情况（如负值、零值）
- 保持数值稳定性（避免除零、溢出）

### 4.3 汇流模块 (core/routing.py)

**核心函数**：
```python
def source_separation(pe, r, sm, ex, ki, kg, s0, fr0):
    """
    水源划分：将总径流划分为地表径流、壤中流、地下径流
    """
    pass

def linear_reservoir(x, weight, last_y):
    """
    线性水库调蓄
    """
    pass

def muskingum_routing(inflow, k, x, dt):
    """
    Muskingum河道演算
    """
    pass
```

### 4.4 数据模块 (data/)

**数据格式标准**：
```python
# 标准输入格式 [time, basin, feature]
# feature: [0] = precipitation, [1] = evapotranspiration
input_shape = (n_timesteps, n_basins, 2)

# 标准输出格式 [time, basin, feature]
# feature: [0] = streamflow
output_shape = (n_timesteps, n_basins, 1)
```

**数据验证清单**：
- [ ] 无缺失值
- [ ] 无负值（降水、蒸散发）
- [ ] 时间连续性
- [ ] 数值范围合理

### 4.5 率定模块 (calibration/)

**优化算法接口**：
```python
def optimize(
    objective_func,      # 目标函数
    param_ranges,        # 参数范围
    algorithm='SCE-UA',  # 优化算法
    max_iter=1000,       # 最大迭代次数
    **kwargs
):
    """
    统一的优化接口
    
    Returns:
        best_params: 最优参数
        best_value: 最优目标函数值
    """
    pass
```

**目标函数示例**：
```python
def nse_objective(params):
    """纳什效率系数（最大化）"""
    simulated = run_model(params)
    return -calculate_nse(observed, simulated)  # 负号因为是最小化

def rmse_objective(params):
    """均方根误差（最小化）"""
    simulated = run_model(params)
    return calculate_rmse(observed, simulated)
```

---

## 五、关键注意事项

### 5.1 数值计算

| 问题 | 解决方案 |
|------|----------|
| 除零错误 | 添加小量 epsilon (如 1e-10) |
| 数值溢出 | 使用 np.clip() 限制范围 |
| 浮点精度 | 设置容差阈值 (如 1e-6) |
| 负值处理 | 使用 np.maximum() 确保非负 |

**示例**：
```python
# 避免除零
fr = r / np.maximum(pe, 1e-10)

# 限制范围
w = np.clip(w, 0, wm)

# 确保非负
r = np.maximum(r, 0)
```

### 5.2 时间步长处理

**不同时段长的参数转换**：
```python
def convert_parameters(params, time_interval_hours):
    """
    将日参数转换为其他时段长的参数
    """
    # ki, kg 需要根据时段长转换
    n_per_day = 24 / time_interval_hours
    ki_new = 1 - (1 - ki) ** (1 / n_per_day)
    kg_new = 1 - (1 - kg) ** (1 / n_per_day)
    return ki_new, kg_new
```

### 5.3 初始条件

**预热期处理**：
```python
def run_model_with_warmup(data, params, warmup_length):
    """
    带预热期的模型运行
    """
    # 1. 运行预热期
    warmup_data = data[:warmup_length]
    _, final_states = run_model(warmup_data, params, return_states=True)
    
    # 2. 使用预热期最终状态运行正式期
    main_data = data[warmup_length:]
    result = run_model(main_data, params, initial_states=final_states)
    
    return result
```

### 5.4 单位转换

**常见单位转换**：
```python
# 降水: mm -> m³/s
# Q = P * A / (3600 * 1000)  [m³/s]
# 其中 P [mm/h], A [km²]

# 流量: m³/s -> mm
# Q_mm = Q * 3600 * 1000 / A / 1e6  [mm/h]
```

### 5.5 结果验证

**必须验证的指标**：
1. **水量平衡**：∑P - ∑E - ∑Q ≈ ΔS
2. **纳什效率系数**：NSE > 0.5 (可接受)，NSE > 0.7 (良好)
3. **峰值误差**：峰值流量误差 < 20%
4. **径流系数**：在合理范围内 (0.2-0.8)

---

## 六、测试规范

### 6.1 单元测试

```python
def test_calculate_evaporation():
    """测试蒸散发计算"""
    # 准备输入
    precip = np.array([10.0])
    pet = np.array([5.0])
    wu, wl, wd = 10.0, 50.0, 40.0
    params = {'um': 20, 'lm': 70, 'dm': 60, 'c': 0.15}
    
    # 运行函数
    eu, el, ed = calculate_evaporation(precip, pet, wu, wl, wd, params)
    
    # 验证结果
    assert eu >= 0 and eu <= pet
    assert el >= 0
    assert ed >= 0
    assert eu + el + ed <= precip + wu + wl
```

### 6.2 集成测试

```python
def test_full_model_run():
    """测试完整模型运行"""
    # 准备测试数据
    data = generate_synthetic_data(365)
    params = get_default_params()
    
    # 运行模型
    result = run_model(data, params)
    
    # 验证输出
    assert result.shape[0] == data.shape[0]
    assert np.all(result >= 0)  # 流量非负
    assert not np.any(np.isnan(result))  # 无NaN值
```

### 6.3 对比测试

```python
def test_compare_with_original():
    """与原始代码对比"""
    # 运行原始代码
    result_original = run_original_model(data, params)
    
    # 运行新代码
    result_new = run_new_model(data, params)
    
    # 计算差异
    diff = np.abs(result_original - result_new)
    max_diff = np.max(diff)
    
    # 验证一致性
    assert max_diff < 1e-10, f"最大差异 {max_diff} 超过容差"
```

---

## 七、文档规范

### 7.1 代码文档

**函数文档模板**：
```python
def function_name(param1, param2, **kwargs):
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
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### 7.2 项目文档

**README必须包含**：
1. 项目简介
2. 安装说明
3. 快速开始
4. 模块说明
5. 示例数据
6. 常见问题
7. 参考文献

---

## 八、版本控制

### 8.1 Git规范

**分支策略**：
- `main`: 稳定版本
- `develop`: 开发版本
- `feature/*`: 新功能
- `bugfix/*`: Bug修复

**提交信息格式**：
```
<类型>: <简短描述>

<详细描述（可选）>

类型包括：
- feat: 新功能
- fix: Bug修复
- docs: 文档更新
- refactor: 重构
- test: 测试
```

### 8.2 版本号

采用语义化版本：`MAJOR.MINOR.PATCH`
- MAJOR: 不兼容的API变更
- MINOR: 向后兼容的功能新增
- PATCH: 向后兼容的问题修正

---

## 九、总结清单

### 项目必备要素

- [ ] 模块化结构（产流、汇流、数据、率定、可视化）
- [ ] 参数外部化配置（JSON/TXT）
- [ ] 示例数据（日尺度、小时尺度）
- [ ] 对比验证功能
- [ ] 中文README文档
- [ ] 单元测试
- [ ] 可视化功能

### 代码质量要求

- [ ] 函数接口清晰，有完整文档
- [ ] 处理边界情况和异常
- [ ] 数值计算稳定
- [ ] 支持向量化运算
- [ ] 有日志记录

### 数据要求

- [ ] 标准化的输入输出格式
- [ ] 数据验证功能
- [ ] 支持多种数据格式（CSV、JSON）
- [ ] 提供合成数据生成器

---

**文档版本**: 1.0
**最后更新**: 2024年
**适用范围**: 概念性水文模型（如新安江模型、Sacramento模型、HBV模型等）

# HydroTune-AI 水文模型智能率定系统

水文模型智能率定系统，支持多模型集成、多场次洪水率定、数据预分析、参数优化与智能报告。

## 1. 项目概述

**HydroTune-AI** 是一个水文模型智能率定系统，支持：

- 📊 **多模型集成**: HBV模型、新安江模型、水箱模型
- 🔄 **智能率定**: 自动率定模型参数，支持多种优化算法
- 🎯 **XGBoost误差校正**: 对模型模拟结果进行后处理校正，提升精度
- 📈 **数据预分析**: 自动识别场次洪水，分析数据质量
- 🤖 **AI智能报告**: 自动生成分析报告

## 2. 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
streamlit run app.py
```

### 测试命令

```bash
# 运行所有测试
pytest

# 运行单个子模块测试
cd tank-model-structured && pytest

# 运行单个测试文件
pytest tests/test_core_generation.py
```

### 代码检查

```bash
# 严格模式：检查语法错误和未定义名称
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 警告模式
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### 子模块初始化

```bash
git submodule update --init --recursive
```

## 3. 数据格式

上传的 CSV/Excel 文件需包含以下列（列名支持中英文）：

| 列名 | 说明 | 单位 |
|------|------|------|
| date | 时间 | - |
| precip | 降水 | mm |
| evap | 蒸发（可选） | mm |
| flow | 流量 | m³/s |

## 4. 支持的模型

### 当前可用模型

| 模型 | 描述 | 参数数量 | 适用场景 |
|------|------|---------|----------|
| **tank水箱模型** | 多层水箱结构模型 | 6个 | 通用水文模拟 |
| **HBV模型** | 瑞典水文模型，含土壤模块 | 10个 | 日尺度水文模拟 |
| **新安江模型2** | 三水源概念性水文模型 | 21个 | 中国湿润地区 |

### 参数说明

#### 水箱模型

| 参数名 | 物理意义 | 单位 | 取值范围 |
|--------|----------|------|----------|
| k1 | 快速流调蓄系数 | - | 0.01 ~ 0.3 |
| k2 | 慢速流调蓄系数（基流） | - | 0.001 ~ 0.05 |
| c | 产流系数 | - | 0.01 ~ 0.3 |

#### HBV模型

| 参数名 | 物理意义 | 单位 | 取值范围 |
|--------|----------|------|----------|
| fc | 田间持水量 | mm | 50 ~ 500 |
| beta | 形状参数 | - | 1.0 ~ 5.0 |
| k0 | 快速出流系数 | - | 0.01 ~ 0.5 |
| k1 | 慢速出流系数 | - | 0.001 ~ 0.1 |
| lp | 蒸散发限制系数 | - | 0.3 ~ 1.0 |

#### 新安江模型2

| 参数名 | 物理意义 | 单位 | 取值范围 |
|--------|----------|------|----------|
| B | 蓄水容量曲线指数 | - | 0.1 ~ 0.5 |
| C | 深层蒸散发系数 | - | 0.1 ~ 0.3 |
| WM | 流域平均蓄水容量 | mm | 100 ~ 200 |
| WUM | 上层土壤蓄水容量 | mm | 10 ~ 50 |
| WLM | 下层土壤蓄水容量 | mm | 50 ~ 150 |
| IM | 不透水面积比例 | - | 0.01 ~ 0.1 |
| SM | 自由水蓄水容量 | mm | 10 ~ 80 |
| EX | 自由水容量曲线指数 | - | 1.0 ~ 2.0 |
| K | 蒸散发系数 | - | 0.5 ~ 1.5 |
| KI | 壤中流出流系数 | - | 0.3 ~ 0.7 |
| KG | 地下水出流系数 | - | 0.01 ~ 0.2 |
| CG | 地下水消退系数 | - | 0.9 ~ 0.999 |
| CI | 壤中流消退系数 | - | 0.1 ~ 0.9 |
| CS | 流域汇流系数 | - | 0.1 ~ 0.5 |
| L | 滞后时间 | h | 0 ~ 24 |
| X | 马斯京根流量比重因子 | - | 0.1 ~ 0.5 |
| K_routing | 马斯京根汇流时间 | h | 1 ~ 24 |
| X_routing | 马斯京根流量比重因子 | - | 0.1 ~ 0.5 |

## 5. 率定算法

系统支持5种优化算法进行水文模型参数率定：

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **两阶段算法(推荐)** | 全局搜索+局部精细优化，速度快精度高 | 日常率定，推荐首选 |
| **PSO** | 粒子群优化，收敛速度快 | 大规模参数优化 |
| **SCE-UA** | 洗牌复形进化，全局搜索能力强 | 复杂非线性问题 |
| **差分进化(DE)** | 简单高效，参数少 | 快速测试 |
| **遗传算法(GA)** | 遗传多样性好，稳健性强 | 高维参数空间 |

## 6. XGBoost误差校正

### 功能说明

XGBoost误差校正是对水文模型模拟结果进行后处理校正的功能，通过机器学习方法预测模型误差并修正模拟结果，提升模拟精度。

### 使用方法

1. 在侧边栏选择 **"多文件（每文件一场洪水）"** 上传模式
2. 在侧边栏 **"XGBoost误差校正"** 部分配置参数：
   - `max_depth`: 树深度 (3-15，默认10)
   - `learning_rate`: 学习率 (0.01-0.5，默认0.05)
   - `n_estimators`: 树数量 (50-500，默认300)
   - `随机种子`: seed值 (1-9999，默认2025)
3. 点击 **"🚀 开始分析"** 按钮

### 技术细节

**输入特征**（共8个）：
- e(t-1) ~ e(t-5): 前5个时间步的真实误差
- p(t-1) ~ p(t-3): 前3个时间步的降水量

**训练集划分**：
- 训练集：率定场次（75%）
- 测试集：验证场次（25%）
- 固定随机种子：seed=2025

**校正公式**：
```
error(t) = flow(t) - simulated(t)
corrected(t) = simulated(t) - predicted_error(t)
```

### 代码位置

| 文件 | 说明 |
|------|------|
| `app.py` | XGB误差校正主流程 |
| `src/app/error_correction.py` | ErrorCorrector类定义 |

## 7. 项目结构

```
HydroTune-AI/
├── src/                           # 主应用代码
│   ├── models/                    # 模型接口层
│   │   ├── base_model.py         # 抽象基类
│   │   ├── registry.py            # 模型注册表
│   │   ├── model_tank.py         # Tank模型适配器
│   │   ├── model_hbv.py          # HBV模型适配器
│   │   └── model_xaj_v2.py       # 新安江模型适配器
│   ├── hydro/                     # 水文计算核心
│   ├── optimizers/                # 优化算法
│   │   ├── de.py                 # 差分进化
│   │   ├── ga.py                 # 遗传算法
│   │   ├── pso.py                # 粒子群
│   │   ├── sce.py                # SCE-UA
│   │   └── two_stage.py          # 两阶段率定
│   ├── data/                      # 数据处理
│   ├── hydro_calc.py             # 水文计算与率定
│   ├── data_agent.py             # 数据清洗与场次识别
│   ├── data_preanalysis.py       # 数据预分析模块
│   ├── bma_ensemble.py           # BMA集合预报
│   ├── llm_reporter.py           # LLM报告生成
│   └── llm_api.py                # LLM接口
├── tank-model-structured/        # Tank模型子模块
├── HBV_model_structured/          # HBV模型子模块
├── XAJ-model-structured/          # 新安江模型子模块
├── demo_data/                     # 示例数据
├── example_data/                  # 参数模板
└── tests/                         # 测试文件
```

## 8. 添加新模型

参考 `src/models/example_model.py` 的实现方式：

```python
from .base_model import BaseModel

class MyModel(BaseModel):
    @property
    def name(self) -> str:
        return "我的模型"
    
    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {'k1': (0.01, 0.3), 'k2': (0.001, 0.05), 'c': (0.01, 0.3)}
    
    def run(self, precip, evap, params, spatial_data=None, temperature=None):
        # 实现模型逻辑
        return flow
```

然后在 `src/models/__init__.py` 中注册：

```python
from .my_model import MyModel
ModelRegistry.register(MyModel())
```

## 9. 云端部署 (Streamlit Cloud)

### 部署步骤

1. **Fork 本仓库** 到你的 GitHub 账号
2. **初始化子模块**：在部署前需要正确初始化子模块
   ```bash
   git submodule update --init --recursive
   ```
3. **在 Streamlit Cloud 部署**：
   - 连接你的 GitHub 仓库
   - 在 Advanced settings 中勾选 **"Include submodules"** 选项
   - 部署主文件：`app.py`

### 常见问题

**Q: 新安江模型加载失败？**
A: 确保在 Streamlit Cloud 部署时勾选了 "Include submodules" 选项。

**Q: 中文字体显示为方块？**
A: 系统会自动检测可用的中文字体。Streamlit Cloud 环境中可能需要配置自定义字体。

---

### 贡献指南

欢迎提交 Pull Request 或 Issue！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
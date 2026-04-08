# HydroTune-AI

水文模型智能率定系统，支持多模型集成、多场次洪水率定、数据预分析、参数优化与智能报告。

## 主要特性

- **新安江模型**：中国湿润地区流域模拟专用
- **多场次率定**：自动识别洪水场次，支持多文件批量处理
- **导入参数**：支持直接导入CSV参数文件，绕过率定环节
- **上游汇流叠加**：支持上游出库流量马斯京根演算叠加
- **时间尺度适配**：支持日尺度和小时尺度数据
- **数据预分析**：自动数据质量评估、洪水事件识别、Pearson III频率分析

## 工作流程

```
上传数据 → 列名映射 → 时间尺度检测 → 数据预分析 → 模型率定 → 智能报告
```

### 数据预分析

1. **数据质量评估**：完整率、连续性、降水-径流相关性、极值检测
2. **洪水事件识别**：基于斜率变化的自动洪水场次识别
3. **频率分析**：Pearson III型曲线拟合，设计洪水计算
4. **代表性选取**：多准则智能选取5场代表性洪水用于率定

## 模型配置

### 当前可用模型

| 模型 | 描述 | 参数数量 | 适用场景 |
|------|------|---------|----------|
| **tank水箱模型** | 多层水箱结构模型 | 6个 | 通用水文模拟 |
| **HBV模型** | 瑞典水文模型，含土壤模块 | 10个 | 日尺度水文模拟 |
| **新安江模型2** | 三水源概念性水文模型 | 21个 | 中国湿润地区 |

### 参数文件模板

参数文件位于 `example_data/` 目录：

- `params_xaj_with_routing_template.csv` - 新安江模型参数（含k_routing, x_routing马斯京根参数）

### 导入参数格式

CSV文件格式：模型名 + 参数名 + 参数值

```csv
模型,B,C,WM,WUM,WLM,IM,SM,EX,K,KI,KG,CG,CI,CS,L,X,K_res,X_res,n,k_routing,x_routing
新安江模型2,0.3,0.2,150.0,23.87,60.0,0.02,30.92,1.12,1.2,0.2,0.37,0.998,0.85,0.72,1,0.27,4.88,0.14,5,2.5,0.25
```

## 快速开始

### 安装依赖

```bash
# 主项目依赖
pip install -r requirements.txt

# 子模块依赖（如需要）
cd tank-model-structured && pip install -r requirements.txt
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

## 云端部署 (Streamlit Cloud)

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

### 子模块说明

本项目包含三个子模块：
- `XAJ-model-structured` - 新安江模型源码
- `HBV_model_structured` - HBV模型源码  
- `tank-model-structured` - 水箱模型源码

### 常见问题

**Q: 新安江模型加载失败？**
A: 确保在 Streamlit Cloud 部署时勾选了 "Include submodules" 选项。部署日志中会显示模型加载状态。

**Q: 中文字体显示为方块？**
A: 系统会自动检测可用的中文字体。Streamlit Cloud 环境中可能需要配置自定义字体。

### 数据格式

上传的 CSV/Excel 文件需包含以下列（列名支持中英文）：

| 列名 | 说明 | 单位 |
|------|------|------|
| date | 时间 | - |
| precip | 降水 | mm |
| evap | 蒸发（可选） | mm |
| flow | 流量 | m³/s |

## 参数说明

### 水箱模型

| 参数名 | 物理意义 | 单位 | 取值范围 |
|--------|----------|------|----------|
| k1 | 快速流调蓄系数 | - | 0.01 ~ 0.3 |
| k2 | 慢速流调蓄系数（基流） | - | 0.001 ~ 0.05 |
| c | 产流系数 | - | 0.01 ~ 0.3 |

### HBV模型

| 参数名 | 物理意义 | 单位 | 取值范围 |
|--------|----------|------|----------|
| fc | 田间持水量 | mm | 50 ~ 500 |
| beta | 形状参数 | - | 1.0 ~ 5.0 |
| k0 | 快速出流系数 | - | 0.01 ~ 0.5 |
| k1 | 慢速出流系数 | - | 0.001 ~ 0.1 |
| lp | 蒸散发限制系数 | - | 0.3 ~ 1.0 |

### 新安江模型2

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

## 项目结构

```
HydroTune-AI/
├── src/                           # 主应用代码
│   ├── models/                    # 模型接口层
│   │   ├── base_model.py         # 抽象基类
│   │   ├── registry.py            # 模型注册表
│   │   ├── example_model.py      # 简化模型示例
│   │   ├── model_tank.py         # Tank模型适配器
│   │   ├── model_hbv.py          # HBV模型适配器
│   │   ├── model_xaj.py          # 新安江模型适配器
│   │   ├── model_xaj_v2.py       # 新安江模型v2
│   │   ├── config/               # 模型配置
│   │   ├── loaders/              # 模型加载器
│   │   └── runners/              # 模型运行器
│   ├── hydro/                     # 水文计算核心
│   │   ├── tank_simple.py        # Tank模型核心
│   │   ├── xaj_simple.py        # 新安江核心
│   │   ├── hbv_simple.py         # HBV核心
│   │   ├── tank_generation.py    # Tank代码生成
│   │   └── tank_config.py        # Tank配置
│   ├── optimizers/                # 优化算法
│   │   ├── de.py                 # 差分进化
│   │   ├── ga.py                 # 遗传算法
│   │   ├── pso.py                # 粒子群
│   │   ├── sce.py                # SCE-UA
│   │   └── two_stage.py          # 两阶段率定
│   ├── agent/                     # 工作流代理
│   │   └── workflow.py           # 工作流编排
│   ├── data/                      # 数据处理
│   │   ├── parser.py             # 数据解析
│   │   └── spatial_handler.py    # 空间数据处理
│   ├── hydro_calc.py             # 水文计算与率定
│   ├── data_agent.py             # 数据清洗与场次识别
│   ├── data_preanalysis.py       # 数据预分析模块
│   ├── bma_ensemble.py           # BMA集合预报
│   ├── muskingum_routing_v2.py   # 马斯京根演算
│   ├── llm_reporter.py           # LLM报告生成模块
│   └── llm_api.py                # LLM接口
├── tank-model-structured/        # Tank模型子模块
├── HBV_model_structured/         # HBV模型子模块
├── XAJ-model-structured/          # 新安江模型子模块
├── demo_data/                     # 示例数据
├── example_data/                  # 参数模板
└── tests/                         # 测试文件
```

## 率定算法

采用两阶段快速率定算法：

1. **全局搜索**：使用 `dual_annealing` 进行全局优化
2. **局部优化**：使用 `L-BFGS-B` 进行精细调整

## 数据流域面积

默认流域面积：**150.7944 km²**

可在侧边栏根据实际情况调整。

## 添加新模型

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

## 参考资料

- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - 架构设计规范
- 各子项目的 README 文档

---

## 后续开发计划

### Phase 1: 核心功能完善 ⏱️ 2024 Q2-Q3

#### 1.1 模型增强
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 完整版Tank模型 | P1 | 支持多层水箱结构，改进小时尺度模拟 |
| 完整版HBV模型 | P1 | 集成积雪模块，支持冻土产流 |
| SAC-SMA模型 | P2 | 美国工程兵团标准版土壤蓄水量模型 |
| SMAP模型 | P2 | 简化版水量平衡模型 |

#### 1.2 率定算法优化
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 多目标率定 | P1 | 支持NSE+RMSE+水量平衡多目标优化 |
| 不确定性分析 | P2 | GLUE方法，参数不确定性估计 |
| 敏感性分析 | P2 | Morris筛选法、SOBOL指数 |
| 分布式率定 | P3 | 并行化率定加速 |

### Phase 2: AI能力增强 ⏱️ 2024 Q3-Q4

#### 2.1 智能分析
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 参数物理约束检查 | P1 | LLM自动识别不合理参数组合 |
| 模拟结果异常诊断 | P1 | 自动识别尖峰、断流等异常并分析原因 |
| 流域特征智能提取 | P2 | 从数据中自动识别流域特征参数 |

#### 2.2 报告增强
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 对比分析报告 | P1 | 多模型、多场次自动对比分析 |
| 率定质量评估 | P2 | 自动评估率定结果的可靠性和代表性 |
| 历史报告管理 | P2 | 报告版本管理与检索 |

### Phase 3: 分布式水文模型 ⏱️ 2024 Q4 - 2025 Q1

#### 3.1 空间数据支持
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 流域离散化 | P1 | 支持子流域划分与拓扑关系构建 |
| 气象场插值 | P1 | 雷达降水、卫星降水网格化 |
| DEM处理 | P1 | 河网提取、流域边界识别 |

#### 3.2 分布式模型
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 分布式Tank模型 | P2 | 网格/子流域分布式模拟 |
| TOPMODEL | P2 | 基于地形的水文模型 |
| SWAT模型集成 | P3 | 集成SWAT模型 |

### Phase 4: 产品化与部署 ⏱️ 2025 Q1-Q2

#### 4.1 系统功能
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 用户权限管理 | P1 | 多用户、项目组权限控制 |
| 数据版本管理 | P1 | 历史数据与结果追溯 |
| API接口开放 | P2 | RESTful API，支持第三方集成 |

#### 4.2 性能优化
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 批量计算优化 | P1 | 异步任务队列，大规模计算支持 |
| 结果缓存 | P2 | 中间结果缓存，避免重复计算 |
| 前端性能优化 | P2 | 大数据量图表优化 |

### Phase 5: 生态扩展 ⏱️ 2025 Q2-Q4

#### 5.1 插件系统
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 插件市场 | P2 | 第三方模型与算法插件 |
| 插件SDK | P2 | Python插件开发规范与文档 |
| 模型校验工具 | P2 | 第三方模型接入标准化校验 |

#### 5.2 知识库
| 功能 | 优先级 | 描述 |
|------|--------|------|
| 流域特征库 | P3 | 典型流域参数参考库 |
| 专家规则库 | P3 | 专家经验参数建议系统 |
| 案例库 | P3 | 典型洪水案例分析 |

### 技术债务与维护

| 任务 | 优先级 | 描述 |
|------|--------|------|
| 单元测试覆盖 | P1 | 核心模块测试覆盖率 > 80% |
| 文档完善 | P1 | API文档、用户手册 |
| CI/CD流水线 | P2 | 自动测试与部署 |
| 代码重构 | P2 | 模块解耦与接口标准化 |

### 贡献指南

欢迎提交 Pull Request 或 Issue！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

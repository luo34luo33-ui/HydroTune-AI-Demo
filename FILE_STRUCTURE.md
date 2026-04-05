# HydroTune-AI 文件结构详细说明

本文档详细说明 HydroTune-AI 项目中每个文件的作用、内置函数、函数输入参数以及它们被哪些功能所调用。

---

## 一、项目主目录文件

### 1.1 `app.py` - Streamlit 主入口应用

**作用**：HydroTune-AI 的 Web 界面主程序，提供完整的水文模型率定交互界面。

**主要函数**：

| 函数名 | 功能 | 输入参数 | 被调用位置 |
|--------|------|----------|------------|
| `show_models_page()` | 显示水文模型介绍页面 | 无 | 页面路由 |
| `show_main_page()` | 显示主页（已内联在主流程中） | 无 | 页面路由 |
| 文件读取与数据处理 | 上传文件、列名映射、数据清洗 | `uploaded_files` | 侧边栏配置 |
| 洪水场次识别 | 调用 `detect_flood_events()` 识别洪水场次 | 日期、降水、流量 | 按钮点击后 |
| 数据预分析 | 调用 `DataPreAnalyzer` 进行预分析 | 降水、流量、日期序列 | 率定前 |
| 模型率定 | 调用 `calibrate_model_fast()` 率定模型 | 模型名、降水、蒸发、流量 | 率定流程 |
| 绘图展示 | 使用 matplotlib 绘制流量过程线 | 模拟结果、实测数据 | 结果展示 |
| BMA集成 | 调用 `calc_bma_weights()` 和 `apply_bma_ensemble()` | NSE列表 | 结果展示 |
| AI报告生成 | 调用 `generate_multifile_report()` | 率定结果 | 报告生成 |

**导入的核心模块**：
```python
from src.llm_api import call_minimax
from src.data_agent import clean_data_with_sandbox, infer_timestep, infer_timestep_by_llm, get_timestep_info, detect_flood_events, FloodEvent
from src.hydro_calc import calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias, calc_kge, get_model_param_info, generate_param_table
from src.data_preanalysis import DataPreAnalyzer, PreAnalysisResult, FloodEvent as PAFloodEvent, DataQualityResult, FrequencyAnalysisResult
from src.llm_reporter import generate_preanalysis_report, generate_calibration_report, generate_comprehensive_report, generate_multifile_report
from src.models.registry import ModelRegistry
from src.bma_ensemble import calc_bma_weights, apply_bma_ensemble, calc_bma_metrics, format_weights_string
```

---

## 二、核心模块 `src/`

### 2.1 `src/__init__.py`

**作用**：初始化 src 包，注册所有模型到 ModelRegistry。

**主要功能**：无直接函数，被 `import src` 时自动执行，用于触发 `src/models/__init__.py` 中的模型注册。

---

### 2.2 `src/hydro_calc.py` - 水文计算与率定模块

**作用**：提供所有水文计算核心功能，包括 NSE/RMSE/KGE/PBIAS 评估指标计算、多种优化算法率定模型、马斯京根河道演算。

**内置函数**：

#### 评估指标计算

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `calc_nse(observed, simulated)` | 计算纳什效率系数 | `observed`: 观测值数组, `simulated`: 模拟值数组 | float: NSE值，-9999表示无效 |
| `calc_rmse(observed, simulated)` | 计算均方根误差 | 同上 | float: RMSE |
| `calc_mae(observed, simulated)` | 计算平均绝对误差 | 同上 | float: MAE |
| `calc_pbias(observed, simulated)` | 计算百分比偏差 | 同上 | float: PBIAS |
| `calc_kge(observed, simulated)` | 计算Kling-Gupta效率系数 | 同上 | float: KGE |

#### 优化算法

| 函数名 | 功能 | 输入 | 被调用 |
|--------|------|------|--------|
| `calibrate_model(model_name, precip, evap, observed_flow, max_iter, spatial_data, temperature, timestep, callback)` | 差分进化算法率定 | 模型名、降水、蒸发、观测流量、迭代次数等 | 已废弃，使用 calibrate_model_fast |
| `calibrate_model_fast(model_name, precip, evap, observed_flow, max_iter, spatial_data, temperature, timestep, algorithm, algo_params, upstream_flow, enable_routing, calib_events, warmup_steps)` | **多算法快速率定** | 模型名、数据、迭代次数、算法选择('two_stage'/'pso'/'ga'/'sce'/'de')、算法参数等 | app.py 中的率定流程 |
| `_two_stage_optimize(objective, bounds, max_iter, n_params)` | 两阶段优化（退火+L-BFGS-B） | 目标函数、参数边界、迭代次数 | calibrate_model_fast |
| `_pso_optimize(objective, bounds, max_iter, n_params, algo_params)` | 粒子群优化 | 同上 | calibrate_model_fast |
| `_ga_optimize(objective, bounds, max_iter, n_params, algo_params)` | 遗传算法 | 同上 | calibrate_model_fast |
| `_sce_optimize(objective, bounds, max_iter, n_params)` | SCE-UA优化 | 同上 | calibrate_model_fast |
| `_de_optimize(objective, bounds, max_iter, n_params, algo_params)` | 差分进化优化 | 同上 | calibrate_model_fast |

#### 河道演算

| 函数名 | 功能 | 输入 | 被调用 |
|--------|------|------|--------|
| `muskingum_routing(upstream_flow, k, x)` | 马斯京根河道汇流演算 | 上游流量数组, k(传播时间), x(权重因子) | calibrate_model_fast (当 enable_routing=True) |

#### 多模型比较

| 函数名 | 功能 | 输入 | 被调用 |
|--------|------|------|--------|
| `compare_all_models(precip, evap, observed_flow, max_iter, spatial_data, temperature)` | 运行所有已注册模型并比较 | 降水、蒸发、流量、迭代次数等 | 较少使用 |

#### 辅助函数

| 函数名 | 功能 | 输入 | 被调用 |
|--------|------|------|--------|
| `get_model_param_info(model_name)` | 获取模型参数描述、单位、范围 | 模型名称 | app.py 中生成参数表格 |
| `generate_param_table(model_name, params)` | 生成参数表格DataFrame | 模型名、参数字典 | app.py 中生成参数表格 |

**被调用关系**：
- `calc_nse()`: 被 `calibrate_model_fast()`、`bma_ensemble.py` 中的评估函数调用
- `calibrate_model_fast()`: 被 `app.py` 的多模型率定流程调用
- `muskingum_routing()`: 被 `calibrate_model_fast()` 在启用上游汇流时调用

---

### 2.3 `src/data_agent.py` - 数据沙盒执行器

**作用**：提供数据指纹提取、LLM智能数据清洗、时间尺度推断、洪水场次识别等数据预处理功能。

**内置函数**：

#### 时间尺度检测

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `infer_timestep(dates)` | 从日期列推断时间尺度 | `dates`: pandas Series | 'hourly' 或 'daily' |
| `infer_timestep_by_llm(dates, llm_caller)` | 使用LLM智能推断时间尺度 | 日期序列, LLM调用函数 | 'hourly' 或 'daily' |
| `get_timestep_info(timestep)` | 获取时间尺度相关信息 | 'hourly' 或 'daily' | dict: 包含hours、seconds等 |

#### 洪水场次识别

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `detect_flood_events(dates, precip, flow, evap)` | 基于降水和流量峰值自动识别洪水场次 | 日期、降水、流量、蒸发(可选) | List[FloodEvent] |
| `split_into_events(df, event_col, n_events)` | 将数据分割成多个洪水场次 | DataFrame、分割列名、场次数 | List[DataFrame] |

#### 数据指纹与清洗

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `extract_fingerprint(df)` | 提取数据指纹发送给LLM分析 | 原始DataFrame | dict: columns、dtypes、head等 |
| `fallback_rename(df)` | 保底方案：基于关键词强行重命名列 | 原始DataFrame | 重命名后的DataFrame |
| `clean_data_with_sandbox(df, llm_caller)` | 核心功能：向LLM发送数据指纹→生成清洗代码→exec()执行 | 原始DataFrame, LLM调用函数 | (清洗后DataFrame, 时间尺度) |

#### 数据类

| 类名 | 作用 | 属性 |
|------|------|------|
| `FloodEvent` | 洪水场次数据类 | name, start_idx, end_idx, start_date, end_date, precip, evap, observed_flow |

**被调用关系**：
- `infer_timestep()`: 被 `clean_data_with_sandbox()` 调用
- `infer_timestep_by_llm()`: 被 `app.py` 中的时间尺度确认流程调用
- `detect_flood_events()`: 被 `app.py` 中的洪水场次识别调用
- `clean_data_with_sandbox()`: 被用于数据自动清洗（当前 Demo 版本主要用 fallback_rename）

---

### 2.4 `src/data_preanalysis.py` - 数据预分析模块

**作用**：提供数据质量评估、洪水事件自动识别、洪水特征统计分析、皮尔逊III型频率分析、代表性洪水智能选取等功能。

**内置函数**：

#### 数据类

| 类名 | 作用 |
|------|------|
| `FloodEvent` | 洪水事件数据类，包含event_id、start_idx、end_idx、peak_flow、flood_volume等 |
| `DataQualityResult` | 数据质量评估结果，包含completeness、continuity、correlation、quality_level等 |
| `FrequencyAnalysisResult` | 频率分析结果，包含n_samples、mean、std、cv、cs、design_values等 |
| `PreAnalysisResult` | 预分析完整结果，整合quality、events、frequency、selected_events |

#### 主类

| 类名 | 方法 | 功能 | 输入 | 返回值 |
|------|------|------|------|--------|
| `DataPreAnalyzer` | `__init__(area)` | 初始化预分析器 | 流域面积 | - |
| | `evaluate_quality(precip, flow, dates)` | 评估数据质量 | 降水、流量、日期序列 | DataQualityResult |
| | `detect_flood_events(dates, precip, flow, threshold_ratio)` | 基于斜率变化识别洪水事件 | 日期、降水、流量 | List[FloodEvent] |
| | `detect_flood_events_by_slope(flow, dates, precip_threshold, flow_threshold, min_duration, min_peak_ratio)` | 基于斜率变化的洪水事件识别 | 流量、日期 | List[Dict] |
| | `frequency_analysis_pearson(peaks)` | 皮尔逊III型频率分析 | 洪峰流量数组 | FrequencyAnalysisResult |
| | `estimate_baseflow(flow, window)` | 估计基流（滚动最小值法） | 流量序列 | 基流数组 |
| | `select_representative_floods(events, freq_result, n_select)` | 智能选取代表性洪水 | 洪水事件列表、频率分析结果 | List[FloodEvent] |
| | `analyze(dates, precip, flow, timestep, n_select)` | 执行完整数据预分析 | 日期、降水、流量、时间尺度 | PreAnalysisResult |

#### 便捷函数

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `analyze_flood_data(dates, precip, flow, area, timestep, n_select)` | 便捷函数：执行完整数据预分析 | 日期、降水、流量、流域面积、时间尺度 | PreAnalysisResult |

**被调用关系**：
- `DataPreAnalyzer`: 被 `app.py` 中的数据预分析流程调用
- `analyze_flood_data()`: 便捷入口函数，被较少使用

---

### 2.5 `src/llm_api.py` - Minimax LLM API 调用封装

**作用**：封装 Minimax API 调用，提供 AI 驱动的数据分析和报告生成能力。

**内置函数**：

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `call_minimax(prompt, system_prompt)` | 调用 Minimax API 返回生成的文本 | 用户输入prompt, 系统提示词system_prompt | str: LLM响应文本 |

**被调用关系**：
- `call_minimax()`: 被 `data_preanalysis.py` 中的 LLM 智能分析调用，被 `llm_reporter.py` 中所有报告生成函数调用，被 `app.py` 中的时间尺度智能识别调用

---

### 2.6 `src/llm_reporter.py` - LLM 水文报告生成模块

**作用**：基于 LLM 生成专业的水文分析报告，包括数据预分析报告、模型率定分析报告、综合分析报告、多文件模式报告。

**内置函数**：

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `generate_preanalysis_report(result, call_llm)` | 生成数据预分析报告 | PreAnalysisResult, LLM调用函数 | str: Markdown格式报告 |
| `generate_calibration_report(calibration_results, all_results, catchment_area, call_llm)` | 生成模型率定分析报告 | 率定结果、模拟结果、流域面积 | str: Markdown格式报告 |
| `generate_comprehensive_report(preanalysis_result, calibration_results, all_results, call_llm)` | 生成综合分析报告 | 预分析结果、率定结果 | str: Markdown格式报告 |
| `generate_multifile_report(file_data_list, calibration_results, file_simulation_results, call_llm, warmup_hours)` | 生成多文件模式分析报告（率定-验证分开） | 文件数据列表、率定结果、模拟结果 | str: Markdown格式报告 |
| `_build_preanalysis_llm_prompt(quality, events, freq, selected)` | 构建预分析的LLM提示词 | 质量评估、洪水事件、频率分析、选取事件 | str: prompt |

**被调用关系**：
- `generate_preanalysis_report()`: 被较少使用
- `generate_calibration_report()`: 较少使用
- `generate_comprehensive_report()`: 较少使用
- `generate_multifile_report()`: 被 `app.py` 在多文件模式下调用，用于生成率定验证分析报告

---

### 2.7 `src/bma_ensemble.py` - BMA (贝叶斯模型平均) 集成模块

**作用**：通过加权平均多个模型的预测结果来提高预测精度，权重基于各模型在验证集上的 NSE 表现计算。

**内置函数**：

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `calc_bma_weights(nse_list, temperature)` | 基于NSE计算BMA权重 | NSE值列表, 温度参数(默认2.0) | np.ndarray: 归一化权重数组 |
| `apply_bma_ensemble(simulated_list, weights)` | 应用BMA集成 | 模拟结果列表, 权重 | np.ndarray: 集成结果 |
| `calc_bma_metrics(observed, simulated_list, weights)` | 计算BMA集成的评估指标 | 实测流量, 模拟结果列表, 权重 | dict: NSE/RMSE/MAE/PBIAS |
| `get_model_weights_dict(model_names, weights)` | 获取模型权重字典 | 模型名列表, 权重数组 | dict: 模型名→权重 |
| `format_weights_string(model_names, weights)` | 格式化权重字符串用于图例 | 模型名列表, 权重数组 | str: 格式化的权重字符串 |

**内部辅助函数**（与 hydro_calc.py 重复）：

| 函数名 | 功能 |
|--------|------|
| `calc_nse(observed, simulated)` | 计算NSE |
| `calc_rmse(observed, simulated)` | 计算RMSE |
| `calc_mae(observed, simulated)` | 计算MAE |
| `calc_pbias(observed, simulated)` | 计算PBIAS |

**被调用关系**：
- `calc_bma_weights()`: 被 `app.py` 在绘制多模型结果时调用
- `apply_bma_ensemble()`: 被 `app.py` 在绘制 BMA 集成结果时调用
- `calc_bma_metrics()`: 较少使用
- `format_weights_string()`: 被 `app.py` 在图例中显示权重

---

## 三、模型模块 `src/models/`

### 3.1 `src/models/__init__.py` - 模型注册初始化

**作用**：初始化模型包，自动注册所有可用模型到 ModelRegistry。

**注册流程**：
```python
# 注册Tank水箱模型 (完整版)
from .model_tank import TankModel
ModelRegistry.register(TankModel())

# 注册HBV模型 (完整版)
from .model_hbv import HBVModelAdapter
ModelRegistry.register(HBVModelAdapter())

# 注册新安江模型
from .model_xaj import XAJModel
ModelRegistry.register(XAJModel())

# 注册新安江模型V2
from .model_xaj_v2 import XAJModelV2
ModelRegistry.register(XAJModelV2())
```

**被调用关系**：
- 被 `app.py` 导入时自动执行：`from src.models.registry import ModelRegistry`

---

### 3.2 `src/models/base_model.py` - 水文模型基类

**作用**：定义所有水文模型的统一抽象接口，符合 ARCHITECTURE_GUIDE 规范。

**内置函数/属性**：

| 属性/方法 | 功能 | 输入 | 返回值 |
|-----------|------|------|--------|
| `name` (property) | 模型名称（抽象属性） | - | str |
| `model_type` (property) | 模型类型 | - | 'lumped' 或 'distributed' |
| `supports_hourly` (property) | 模型是否支持小时尺度 | - | bool |
| `get_timestep_hours(spatial_data)` | 获取时间步长(小时) | spatial_data字典 | float: 1.0或24.0 |
| `param_bounds` (property) | 参数取值范围（抽象属性） | - | Dict[str, Tuple[float, float]] |
| `default_params` (property) | 默认参数值（取范围中值） | - | Dict[str, float] |
| `run(precip, evap, params, spatial_data, temperature, warmup_steps)` | 运行模型（抽象方法） | 降水、蒸发、参数、空间数据、温度、预热期 | np.ndarray: 模拟流量 |
| `validate_params(params)` | 验证参数是否在有效范围内 | params字典 | bool |
| `get_required_spatial_data()` | 获取模型需要的空间数据类型 | - | list |

**被调用关系**：
- `TankModel`、`HBVModelAdapter`、`XAJModel`、`XAJModelV2` 继承此基类
- `ModelRegistry` 调用 `run()` 方法执行模型

---

### 3.3 `src/models/registry.py` - 模型注册表

**作用**：支持动态注册和调用水文模型。

**内置函数**：

| 函数名 | 功能 | 输入 | 返回值 |
|--------|------|------|--------|
| `register(model)` | 注册模型 | BaseModel实例 | - |
| `get_model(name)` | 获取模型实例 | 模型名称str | BaseModel |
| `list_models()` | 列出所有已注册模型名称 | - | List[str] |
| `get_all_models()` | 获取所有模型实例 | - | Dict[str, BaseModel] |
| `get_all_bounds()` | 获取所有模型的参数范围 | - | Dict[str, Dict] |
| `clear()` | 清空所有注册的模型 | - | - |

**被调用关系**：
- 被 `app.py` 导入并调用 `list_models()` 获取可用模型列表
- 被 `hydro_calc.py` 中的率定函数调用 `get_model()` 获取模型实例

---

### 3.4 `src/models/model_tank.py` - Tank水箱模型适配器

**作用**：将 tank-model-structured 项目适配到 HydroTune-AI 的统一接口。

**内置函数/属性**：

| 属性/方法 | 功能 |
|-----------|------|
| `name` | "Tank水箱模型(完整版)" |
| `model_type` | "lumped" |
| `param_bounds` | 返回18个Tank模型参数的取值范围 |
| `default_params` | 返回默认参数值 |
| `run(precip, evap, params, spatial_data, temperature, warmup_steps)` | 运行Tank水箱模型 |
| `get_param_descriptions()` | 获取参数描述 |

**参数列表**：t0_is, t0_boc, t0_soc_uo, t0_soc_lo, t0_soh_uo, t0_soh_lo, t1_is, t1_boc, t1_soc, t1_soh, t2_is, t2_boc, t2_soc, t2_soh, t3_is, t3_soc

**被调用关系**：
- 通过 ModelRegistry 注册
- 被 `calibrate_model_fast()` 调用 `run()` 方法

---

### 3.5 `src/models/model_hbv.py` - HBV模型适配器

**作用**：将 HBV_model_structured 项目适配到 HydroTune-AI 的统一接口。

**内置函数/属性**：

| 属性/方法 | 功能 |
|-----------|------|
| `name` | "HBV模型(完整版)" |
| `model_type` | "lumped" |
| `supports_hourly` | False（日尺度） |
| `param_bounds` | 返回HBV模型参数的取值范围 |
| `default_params` | 返回默认参数值 |
| `run(precip, evap, params, spatial_data, temperature, warmup_steps)` | 运行HBV模型 |
| `get_param_descriptions()` | 获取参数描述 |
| `_estimate_temperature(precip, evap)` | 从降水和蒸发估算温度序列 |

**参数列表**：dd, fc, beta, c, k0, l, k1, k2, kp, pwp

**被调用关系**：同 Tank 模型

---

### 3.6 `src/models/model_xaj.py` - 新安江模型适配器

**作用**：将 XAJ-model-structured 项目适配到 HydroTune-AI 的统一接口。

**内置函数/属性**：

| 属性/方法 | 功能 |
|-----------|------|
| `name` | "新安江模型" |
| `model_type` | "lumped" |
| `param_bounds` | 返回15个XAJ模型参数的取值范围 |
| `default_params` | 返回默认参数值 |
| `run(precip, evap, params, spatial_data, temperature, warmup_steps)` | 运行新安江模型 |
| `validate_params(params)` | 验证参数有效性（特殊约束：ki + kg < 1） |
| `get_param_descriptions()` | 获取参数描述 |

**参数列表**：k, b, im, um, lm, dm, c, sm, ex, ki, kg, cs, l, ci, cg

**被调用关系**：同 Tank 模型

---

### 3.7 `src/models/model_xaj_v2.py` - 新安江模型V2适配器

**作用**：新安江模型的另一个版本，与 model_xaj.py 类似。

---

### 3.8 `src/models/example_model.py` - 示例模型

**作用**：作为新增模型的示例代码，展示如何创建符合 HydroTune-AI 接口的新模型。

---

## 四、工作流代理模块 `src/agent/`

### 4.1 `src/agent/workflow.py` - 工作流代理

**作用**：定义自动化工作流程，协调数据处理、模型率定、报告生成等步骤。

**内置函数**（需要进一步查看文件内容）

---

## 五、数据处理模块 `src/data/`

### 5.1 `src/data/spatial_handler.py` - 空间数据处理

**作用**：处理流域空间数据（如 DEM、土地利用、土壤类型等）。

### 5.2 `src/data/parser.py` - 数据解析器

**作用**：解析各种格式的水文数据文件。

---

## 六、子模块

### 6.1 `tank-model-structured/` - Tank模型子模块

**目录结构**：
```
tank-model-structured/
├── core/
│   ├── generation.py    # 产流计算核心
│   ├── routing.py       # 汇流计算
│   └── evapotranspiration.py  # 蒸散发计算
├── config/
│   └── model_config.py  # 模型参数配置
├── data/
│   ├── loader.py        # 数据加载
│   ├── generator.py     # 数据生成
│   └── validator.py     # 数据验证
├── calibration/
│   └── optimizer.py     # 率定优化
└── tests/
    ├── test_core_generation.py
    ├── test_core_routing.py
    └── test_integration.py
```

### 6.2 `HBV_model_structured/` - HBV模型子模块

**目录结构**：
```
HBV_model_structured/
├── core/
│   ├── generation.py    # 产流计算
│   ├── routing.py       # 汇流计算
│   └── snow.py          # 积雪模块
├── config/
│   └── model_config.py
├── data/
│   └── loader.py
├── calibration/
│   └── optimizer.py
├── evaluation/
│   └── metrics.py
└── tests/
```

### 6.3 `XAJ-model-structured/` - 新安江模型子模块

**目录结构**：
```
XAJ-model-structured/
├── core/
│   ├── generation.py    # 产流计算
│   ├── routing.py       # 汇流计算
├── config/
│   └── model_config.py
├── calibration/
│   └── optimizer.py
├── preprocessing.py
├── visualization.py
└── tests/
```

---

## 七、调用关系总结

### 7.1 核心调用链

```
app.py (主界面)
    │
    ├──→ ModelRegistry.list_models()           # 获取可用模型
    │
    ├──→ data_agent.infer_timestep_by_llm()    # 智能识别时间尺度
    │
    ├──→ data_agent.detect_flood_events()      # 识别洪水场次
    │
    ├──→ data_preanalysis.DataPreAnalyzer      # 数据预分析
    │       │
    │       ├──→ evaluate_quality()            # 数据质量评估
    │       ├──→ detect_flood_events_by_slope() # 洪水识别
    │       ├──→ frequency_analysis_pearson()  # 频率分析
    │       └──→ select_representative_floods() # 选取代表性洪水
    │
    ├──→ hydro_calc.calibrate_model_fast()     # 模型率定
    │       │
    │       ├──→ ModelRegistry.get_model()     # 获取模型实例
    │       ├──→ model.run()                   # 运行模型
    │       ├──→ calc_nse()                    # 计算NSE
    │       └──→ muskingum_routing()           # (可选)上游汇流演算
    │
    ├──→ bma_ensemble.calc_bma_weights()       # BMA权重计算
    ├──→ bma_ensemble.apply_bma_ensemble()     # BMA集成
    │
    ├──→ llm_reporter.generate_multifile_report() # 生成AI报告
    │       │
    │       └──→ llm_api.call_minimax()        # 调用LLM
    │
    └──→ hydro_calc.generate_param_table()     # 生成参数表格
```

### 7.2 数据流向

```
用户上传文件
    │
    ↓
app.py: 文件读取 → 列名映射
    │
    ↓
data_agent: 数据清洗（可选LLM）
    │
    ↓
data_preanalysis: 数据质量评估、洪水识别、频率分析
    │
    ↓
hydro_calc: 模型率定（多算法优化）
    │
    ↓
models: Tank/HBV/XAJ 模型运行
    │
    ↓
bma_ensemble: 多模型集成
    │
    ↓
llm_reporter: 生成分析报告
    │
    ↓
展示与下载
```

---

## 八、关键配置

### 8.1 时间步长设置

- **日尺度 (daily)**: `del_t = 24.0` 小时
- **小时尺度 (hourly)**: `del_t = 1.0` 小时

### 8.2 流域面积默认值

- 默认值: `150.7944 km²`
- 可通过 `spatial_data['area']` 自定义

### 8.3 优化算法选择

- `two_stage`（推荐）: 退火 + L-BFGS-B
- `pso`: 粒子群优化
- `ga`: 遗传算法
- `sce`: SCE-UA
- `de`: 差分进化

---

本文档涵盖了 HydroTune-AI 项目中所有核心文件的作用、函数定义、输入参数以及调用关系。如有需要补充或修正，请随时提出。
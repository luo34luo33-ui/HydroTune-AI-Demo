# XGBoost误差修正模型效果对比实验

## 项目概述

本项目用于验证**少样本率定 + XGBoost误差校正**方法与**大样本纯水文模型率定**方法的效果对比与时间效率对比。

## 实验设计

### 目标
- 验证少量洪水场次（5/10/15场）率定水文模型参数，配合XGBoost误差修正模型，能否达到75%大样本率定的效果
- 对比两种方法的时间效率

### 数据
- 输入：60个洪水场次（`tests/inputs/*.csv`）
- 字段：降水(precip)、流量(flow)、蒸发(evap)、上游流量(upstream)
- 预热期：72小时

### 模型
| 模型 | 说明 |
|------|------|
| tank | Tank水箱模型 |
| hbv | HBV水文模型 |
| xaj | 新安江(XAJ)模型 |

### 率定算法
- SCE-UA (Shuffle Complex Evolution)
- 迭代次数：30（MAX_ITERATIONS）
- 参数：水文模型参数 + 马斯京根汇流参数(k_routing, x_routing)

### XGBoost误差模型特征
- 误差滞后：`e[t], e[t-1], ..., e[t-7]` (8维)
- 降水滞后：`p[t-1], p[t-2], p[t-3]` (3维)
- 共11维特征

## 目录结构

```
tests/xgb_error_correction/
├── __init__.py           # 项目初始化
├── config.py             # 配置文件
├── data_loader.py       # 数据加载模块
├── calibrator.py        # SCE-UA率定封装
├── xgb_model.py         # XGBoost误差修正模型
├── benchmark.py         # 75%基准测试
├── plotter.py           # 绘图模块
└── runner.py            # 统一运行脚本
```

## 运行方式

```bash
# 完整运行
python tests/xgb_error_correction/runner.py

# 调试模式（减少迭代次数和运行次数）
# 修改 config.py 中 DEBUG_MODE = True
```

## 输出结果

### CSV文件
位置：`tests/outputs/xgb_correction/xgb_correction_results.csv`

| 字段 | 说明 | 计算方法 |
|------|------|----------|
| model | 模型名称 | 直接赋值：tank/hbv/xaj |
| calibration_ratio | 率定比例 | 直接赋值："5场"/"10场"/"15场"/"75%基准" |
| nse_mean | 复合模拟NSE均值 | 所有场次的误差校正后NSE（all_nse_corrected）|
| nse_std | NSE标准差 | 同一率定比例下多次运行的NSE标准差（当前版本为0，因为每次运行结果单独存储） |
| time_seconds | 耗时(秒) | 单次率定+运行的总耗时 |
| calib_nse | 率定NSE | 率定目标函数值，**未校正**的率定场次平均NSE |
| calib_events | 率定选取的场次 | 逗号分隔的场次名称，如 "20191031,20241016,20180105" |
| run_iter | 运行编号 | 基准为0，实验组为1-10 |
| calib_nse_raw | 率定期的纯水文模型NSE | 率定场次的纯水文模型NSE（未校正），同calib_nse |
| non_calib_nse_raw | 非率定场次的纯水文模型NSE | 非率定场次的纯水文模型NSE（未校正） |
| all_nse_raw | 所有场次的纯水文模型NSE | 全部60场的纯水文模型NSE（未校正） |
| calib_nse_corrected | 率定期的误差校正后NSE | 率定场次经过XGBoost误差校正后的NSE |
| non_calib_nse_corrected | 非率定场次的误差校正后NSE | 非率定场次经过XGBoost误差校正后的NSE |
| all_nse_corrected | 所有场次的误差校正后NSE | 全部60场经过XGBoost误差校正后的NSE |

### 字段详细计算方法

#### 1. model（模型名称）
```
直接赋值：tank / hbv / xaj
```

#### 2. calibration_ratio（率定比例）
```
基准组："75%基准"
实验组：根据CALIBRATION_RATIOS配置，"5场"/"10场"/"15场"
```

#### 3. nse_mean（复合模拟NSE均值）
```
计算流程：
1. 使用率定好的参数（水文参数 + 马斯京根参数）运行全部60场洪水
2. 计算每场的模拟误差：error = sim - obs
3. 构建XGBoost特征：
   - 误差滞后特征: e[t], e[t-1], ..., e[t-7]
   - 降水滞后特征: p[t-1], p[t-2], p[t-3]
4. 用误差数据训练XGBoost模型（70%训练，30%测试）
5. 对全部60场应用XGBoost误差修正：
   corrected_sim = sim + XGB_predicted_error
6. 计算修正后每场的NSE：
   NSE = 1 - Σ(obs - sim)² / Σ(obs - mean(obs))²
7. 计算60场NSE的均值
```

#### 4. nse_std（NSE标准差）
```
计算方法：
- 同一calibration_ratio下多次运行的NSE标准差
- 当前版本为0（因为每次运行结果单独存储为一行）
- 如需计算标准差，需在同一行内聚合多次结果
```

#### 5. time_seconds（耗时秒）
```
计算方法：
- 从进入calibrate()开始计时，到XGBoost评估完成
- 包含：率定时间 + 模型运行时间 + XGBoost训练和预测时间
```

#### 6. calib_nse / calib_nse_raw（率定期的纯水文模型NSE）
```
计算方法：
1. 从60场中随机选取n_calib场（如5场）作为率定场次
2. 使用SCE-UA算法率定参数，最大化率定场次的平均NSE
3. 目标函数：
   obj = -mean(NSE(each_calib_event))
4. 返回率定完成时的NSE均值

重要：此NSE是【未校正】的纯水文模型NSE，
      即直接用 sim vs obs 计算，未经过XGBoost误差修正
```

#### 7. non_calib_nse_raw（非率定场次的纯水文模型NSE）
```
计算方法：
1. 率定参数运行全部60场
2. 筛选出不在率定集合中的场次
3. 计算这些场次的纯水文模型NSE均值

公式：mean(NSE(non_calib_events, sim))
```

#### 8. all_nse_raw（所有场次的纯水文模型NSE）
```
计算方法：
1. 率定参数运行全部60场
2. 计算60场的纯水文模型NSE均值

公式：mean(NSE(all_events, sim))
```

#### 9. calib_nse_corrected（率定期的误差校正后NSE）
```
计算方法：
1. 用率定参数运行全部60场，计算误差
2. 构建XGBoost特征：误差滞后 + 降水滞后
3. 用误差数据训练XGBoost模型
4. 对率定场次应用误差修正：
   corrected_sim = sim + XGB_predicted_error
5. 计算校正后NSE均值

公式：mean(NSE(calib_events, corrected_sim))
```

#### 10. non_calib_nse_corrected（非率定场次的误差校正后NSE）
```
计算方法：
1. 同上，对非率定场次应用误差修正
2. 计算校正后NSE均值

公式：mean(NSE(non_calib_events, corrected_sim))
```

#### 11. all_nse_corrected（所有场次的误差校正后NSE）
```
计算方法：
1. 同上，对全部60场应用误差修正
2. 计算校正后NSE均值

公式：mean(NSE(all_events, corrected_sim))
```

#### 7. calib_events（率定选取的场次）
```
计算方法：
1. 对60场随机打乱（np.random.permutation）
2. 取前n_calib个作为率定场次
3. 格式：逗号分隔的场次名称
   例如："20191031,20241016,20180105,20250514,20171107"
```

#### 8. run_iter（运行编号）
```
基准组：run_iter = 0
实验组：run_iter = 1, 2, 3, ..., N_RUNS
```

### NSE计算公式

```
NSE = 1 - Σ(obs[t] - sim[t])² / Σ(obs[t] - mean(obs))²

其中：
- obs: 实测流量
- sim: 模拟流量
- t: 时间步
- 预热期（WARMUP_STEPS=72）不参与计算
```

### 图表
位置：`tests/outputs/xgb_correction/plots/`

| 文件 | 说明 |
|------|------|
| nse_comparison.png | NSE对比柱状图 |
| time_efficiency.png | 时间效率柱状图 |
| convergence.png | 收敛曲线 |

## 配置参数

文件：`config.py`

```python
# 率定参数
WARMUP_STEPS = 72       # 预热期步数(h)
MAX_ITERATIONS = 30     # SCE-UA迭代次数
CALIBRATION_RATIOS = [5, 10, 15]  # 率定场次数
BENCHMARK_RATIO = 0.75  # 75%基准
N_RUNS = 10             # 随机次数

# 随机种子
RANDOM_SEED = 42
```

## 实验流程

```
加载60场数据
    │
    ├── 基准组（75% = 45场）
    │   固定随机种子(42)，选取45场
    │   SCE-UA率定 → 运行全部60场 → 记录NSE
    │   run_iter = 0
    │
    └── 实验组（5/10/15场）
        ├── 随机选5场率定（种子42+run_i）
        │   SCE-UA率定 → 运行60场
        │   计算误差 → 训练XGBoost → 复合模拟 → NSE
        │   run_iter = 1~10
        ├── 随机选10场率定（种子42+run_i）
        │   ...
        └── 随机选15场率定（种子42+run_i）
            ...

    每种比例重复10次 → 结果追加到CSV
```

## 核心算法

### XGBoost误差修正
```
复合模拟流量 = 水文模型模拟值 + XGBoost预测误差

其中：
- XGBoost输入特征：11维 [e(t), e(t-1), ..., e(t-7), p(t-1), p(t-2), p(t-3)]
- 训练数据：率定场次的误差序列
- 训练/测试划分：70% / 30%（随机划分）
```

### SCE-UA率定
```
目标：最大化率定场次的平均NSE
参数：水文模型参数 + 马斯京根汇流参数(k_routing, x_routing)
优化器：scipy.optimize.differential_evolution
```

## 依赖

- numpy
- pandas
- xgboost
- matplotlib

## 注意事项

1. 确保子模块已初始化：`git submodule update --init --recursive`
2. DEBUG_MODE可减少运行时间用于测试流程
3. 结果CSV支持增量追加，意外中断后可直接重新运行
4. CSV中每行代表一次独立的运行结果，多次运行的均值需自行计算
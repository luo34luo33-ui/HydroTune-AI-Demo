# XGB少样本率定误差修正实验

## 项目概述

本项目用于研究**少样本率定 + XGBoost误差校正**的效果，对比纯水文模型大样本率定的性能。

### 核心问题

在水文模型率定中，通常需要较多洪水场次才能获得良好的参数。但实际应用中：
- 历史洪水场次有限
- 人工率定耗时耗力

本项目探索：是否可以用少量场次率定 + 机器学习误差校正来达到大样本率定的效果？

---

## 功能特性

### 1. 支持多种水文模型

- **Tank模型**：水箱模型
- **HBV模型**：水文模型
- **XAJ模型**：新安江模型

### 2. 支持多种率定算法

- **SCE-UA**： shuffler complex evolution
- 可扩展其他算法

### 3. XGBoost误差修正

#### 3.1 误差校正原理

XGBoost模型学习水文模型的**系统误差模式**，用历史误差序列预测当前误差：

```
特征输入：
- 误差滞后: e[t-1], e[t-2], ..., e[t-N]
- 降水滞后: p[t-1], p[t-2], ..., p[t-M]

预测输出：
- e[t]: 当前时刻的预测误差
```

#### 3.2 递归预测（避免数据泄露）

为确保预测的真实性，采用**递归预测**方式：

```
t=0: 初始化，用真实误差
t=1: 用预测误差 e[0] 作为输入
t=2: 用预测误差 e[0], e[1] 作为输入
...
```

这样避免了使用真实未来误差的"作弊"行为。

### 4. 场次选择策略

- **随机选择**：随机选取率定场次（当前实现）
- **扩展接口**：预留策略模式，可扩展基于特征/历史NSE的选择算法

---

## 目录结构

```
xgb_calibration/
├── config.py                 # 全局配置
├── run_single.py            # 单次运行脚本
│
├── core/                     # 核心模块
│   ├── data_loader.py        # 数据加载
│   ├── evaluator.py          # 评估指标
│   └── utils.py              # 工具函数
│
├── models/                   # 水文模型
│   ├── tank.py, hbv.py, xaj.py
│
├── algorithms/               # 率定算法
│   ├── sce.py                # SCE-UA
│   └── calibrator.py         # 率定器
│
├── selection/                # 场次选择（预留）
│   └── __init__.py
│
├── correction/               # 误差修正
│   ├── xgb_model.py          # XGBoost模型
│   └── corrector.py          # 修正器接口
│
├── experiment/               # 实验运行
│   ├── runner.py             # 运行器
│   └── benchmark.py          # 基准测试
│
├── visualization/             # 可视化
│   ├── plotter.py            # 绘图工具
│   ├── hydrograph.py         # 过程线对比图
│   └── heatmap.py            # 热力图
│
└── outputs/                  # 输出结果
    └── {run_name}/
        └── hydrographs/
            └── {model}/
```

---

## 快速开始

### 环境要求

```bash
pip install numpy pandas scipy xgboost matplotlib
```

### 运行实验

```bash
cd tests/xgb_calibration

# 基本用法
python run_single.py --model tank --n-calib 10

# 指定输出名称（自动创建结果文件夹）
python run_single.py --model tank --n-calib 10 --name tank_experiment

# 其他模型
python run_single.py --model hbv --n-calib 15
python run_single.py --model xaj --n-calib 5
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 水文模型 | tank |
| `--n-calib` | 率定场次数 | 10 |
| `--name` | 结果文件夹名称 | 时间戳 |
| `--seed` | 随机种子 | 42 |

---

## 实验流程

### 流程图

```
1. 加载数据
   └── 60个洪水场次

2. 随机选择率定场次
   └── 5场（假设 n_calib=5）

3. 率定水文模型参数
   └── 用5场数据率定Tank模型

4. 运行全部场次
   └── 得到每个场次的模拟流量 sim

5. 计算误差
   └── error = sim - observed

6. 训练XGBoost
   ├── 非率定期（55场）中划分训练集/测试集
   ├── 训练: 39场
   └── 测试: 16场

7. 误差校正
   └── 对测试集场次进行递归预测校正

8. 评估效果
   └── 比较原始NSE vs 校正后NSE
```

### 输出示例

```
============================================================
Model: TANK
Selector: random
Calibration: 5/60 events
Selected events: 20100806, 20100923, 20230820, 20240621, 20140428
============================================================
[CALIBRATION] Done in 18.5s, NSE=0.7536
[CORRECTION] XGB trained, test NSE=0.7617
[PLOTTING] Generating hydrographs...

[SUMMARY]
  Raw NSE: 0.3751 ± 0.7985
  Corrected NSE: 0.1426 ± 0.8806
  Total time: 13.5s

============================================================
Run completed successfully!
Results saved to: outputs/tank_experiment/
```

---

## 输出说明

### 1. 过程线对比图

每个场次生成一张对比图，包含：
- **蓝色**：实测流量 (Observed)
- **红色虚线**：原始模拟 (Simulated Raw)
- **绿色实线**：XGB校正后 (Simulated Corrected)
- **降水柱状图**
- **NSE指标**

保存位置：`outputs/{run}/hydrographs/{model}/`

### 2. CSV数据

每个场次同时保存为CSV：
- time, observed, simulated_raw, simulated_corrected

### 3. 结果统计

结果自动保存到 `outputs/xgb_correction_results.csv`

---

## 配置说明

### 主要配置项（config.py）

```python
# 数据配置
DATA_DIR = "../inputs"           # 洪水场次数据目录
WARMUP_STEPS = 72               # 预热期步数

# 率定配置
MAX_ITERATIONS = 30              # SCE迭代次数
CALIBRATION_RATIOS = [5, 10, 15] # 率定场次配置

# XGBoost配置
XGB_PARAMS = {
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 300,
}

# 特征配置
N_ERROR_LAGS = 5                # 误差滞后阶数
N_PRECIP_LAGS = 3               # 降水滞后阶数
```

---

## 扩展开发

### 新增场次选择策略

在 `selection/` 目录下实现策略类：

```python
from selection import EventSelector

class MySelector(EventSelector):
    @property
    def name(self) -> str:
        return "my_selector"
    
    def select(self, events: List[Dict], n_calib: int) -> List[Dict]:
        # 实现选择逻辑
        pass
```

### 新增误差修正方法

在 `correction/` 目录下实现修正器：

```python
from correction import Corrector

class MyCorrector(Corrector):
    @property
    def name(self) -> str:
        return "my_corrector"
    
    def train(self, events: List[Dict]) -> 'Corrector':
        # 训练模型
        pass
    
    def correct(self, event: Dict) -> np.ndarray:
        # 校正误差
        pass
```

---

## 参考文献

1. Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. Water Resources Research.

2. Chen, Y., & Zhang, D. (2006). Data assimilation for transient flood modeling. Journal of Hydrology.

---

## 联系方式

项目维护：Hydromodel Team
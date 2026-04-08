# 水文模型率定测试代码

独立运行的率定测试代码，不依赖项目其他模块。

## 目录结构

```
tests/
├── calibration_base.py       # 共享配置常量
├── algos/                    # 优化算法
│   ├── __init__.py
│   ├── two_stage.py          # 两阶段算法
│   ├── pso.py                # 粒子群优化
│   ├── sce.py                # SCE-UA算法
│   ├── de.py                 # 差分进化
│   └── ga.py                 # 遗传算法
├── models/                   # 水文模型
│   ├── __init__.py
│   ├── tank.py               # Tank水箱模型
│   ├── hbv.py                # HBV模型
│   └── xaj.py                # 新安江模型
├── outputs/                  # 输出目录
│   ├── params/               # 参数表
│   ├── plots/                # 洪水过程线图
│   └── data/                 # 模拟结果CSV
└── [model]_[algo]_calibration.py  # 15个率定文件
```

## 率定任务列表 (3模型 × 5算法 = 15个)

| 模型 | 算法 | 文件 |
|------|------|------|
| Tank | 两阶段 | `tank_two_stage_calibration.py` |
| Tank | PSO | `tank_pso_calibration.py` |
| Tank | SCE | `tank_sce_calibration.py` |
| Tank | DE | `tank_de_calibration.py` |
| Tank | GA | `tank_ga_calibration.py` |
| HBV | 两阶段 | `hbv_two_stage_calibration.py` |
| HBV | PSO | `hbv_pso_calibration.py` |
| HBV | SCE | `hbv_sce_calibration.py` |
| HBV | DE | `hbv_de_calibration.py` |
| HBV | GA | `hbv_ga_calibration.py` |
| XAJ | 两阶段 | `xaj_two_stage_calibration.py` |
| XAJ | PSO | `xaj_pso_calibration.py` |
| XAJ | SCE | `xaj_sce_calibration.py` |
| XAJ | DE | `xaj_de_calibration.py` |
| XAJ | GA | `xaj_ga_calibration.py` |

## 运行方式

### 单个率定任务

```bash
# 进入tests目录
cd tests

# 运行Tank模型 + 两阶段算法率定
python tank_two_stage_calibration.py

# 运行HBV模型 + PSO算法率定
python hbv_pso_calibration.py

# 运行新安江模型 + DE算法率定
python xaj_de_calibration.py
```

### 批量运行

```bash
# 运行所有率定任务
for f in *_calibration.py; do python "$f"; done
```

## 配置参数

在 `calibration_base.py` 中配置：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| DATA_DIR | 测试数据目录 | `example_data/60场+提前72h换列名` |
| CATCHMENT_AREA | 流域面积(km²) | 584.0 |
| WARMUP_STEPS | 预热期步数(h) | 72 |
| MAX_ITERATIONS | 迭代次数 | 30 |
| COL_MAPPING | 列名映射 | `avg_rain`→降水, `E0`→蒸发, `GZ_in`→流量, `GB_out`→上游 |

## 输出文件

每个率定任务输出：

```
outputs/
├── params/
│   └── [model]_[algo]_params.csv    # 最优参数表
├── plots/
│   └── [model]/
│       └── [model]_[algo]_flood_[场次名].png  # 洪水过程线图
└── data/
    └── [model]/
        └── [model]_[algo]_simulation_[场次名].csv  # 模拟结果
```

## 依赖

```
numpy
pandas
matplotlib
scipy
```

## 注意事项

1. 代码完全独立，不依赖项目其他模块
2. 上游出库流量(GB_out)通过马斯京根演算后叠加到出口断面
3. 马斯京根参数(k_routing, x_routing)与模型参数一并率定
4. 率定目标：多场次洪水NSE平均值最大化

---

执行示例：

```bash
cd tests
python tank_two_stage_calibration.py
```

将自动：
1. 加载60场洪水数据
2. 使用两阶段算法率定Tank模型+马斯京根参数
3. 输出参数表到 `outputs/params/tank_two_stage_params.csv`
4. 为每场洪水生成过程线图到 `outputs/plots/tank/`
5. 导出模拟结果CSV到 `outputs/data/tank/`
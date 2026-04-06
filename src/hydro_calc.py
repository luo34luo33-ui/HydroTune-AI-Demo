"""
水文计算模块
包含：率定算法、NSE计算、多模型比较
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing, minimize
from typing import Dict, Callable, Tuple, List

from .models.registry import ModelRegistry


# ============================================================
# NSE 计算
# ============================================================
def calc_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    计算纳什效率系数 (Nash-Sutcliffe Efficiency)

    Args:
        observed: 观测值数组
        simulated: 模拟值数组

    Returns:
        NSE值，范围(-inf, 1]，越接近1越好
    """
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]

    if len(obs) == 0:
        return -9999

    obs_mean = np.mean(obs)
    denominator = np.sum((obs - obs_mean) ** 2)

    if denominator == 0:
        return -9999

    return 1 - np.sum((obs - sim) ** 2) / denominator


def calc_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算均方根误差"""
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    if len(obs) == 0:
        return np.nan
    return np.sqrt(np.mean((obs - sim) ** 2))


def calc_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算平均绝对误差"""
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    if len(obs) == 0:
        return np.nan
    return np.mean(np.abs(obs - sim))


def calc_pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算百分比偏差"""
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    if len(obs) == 0:
        return np.nan
    obs_sum = np.sum(obs)
    if obs_sum == 0:
        return np.nan
    return 100 * np.sum(obs - sim) / obs_sum


def calc_kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算Kling-Gupta效率系数"""
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    
    if len(obs) < 2:
        return np.nan
    
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    
    if obs_std == 0 or sim_std == 0:
        return np.nan
    
    r = np.corrcoef(obs, sim)[0, 1]
    if np.isnan(r):
        r = 0
    
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge


# ============================================================
# 差分进化率定
# ============================================================
def calibrate_model(
    model_name: str,
    precip: np.ndarray,
    evap: np.ndarray,
    observed_flow: np.ndarray,
    max_iter: int = 30,
    spatial_data: Dict = None,
    temperature: np.ndarray = None,
    timestep: str = 'daily',
    callback: Callable = None,
) -> Tuple[dict, float, np.ndarray]:
    """
    使用差分进化算法率定模型参数

    Args:
        model_name: 模型名称
        precip: 降水序列
        evap: 蒸发序列
        observed_flow: 观测流量序列
        max_iter: 最大迭代次数
        spatial_data: 空间数据（如流域面积等）
        temperature: 温度序列（HBV模型需要）
        timestep: 时间尺度 'hourly' 或 'daily'
        callback: 进度回调函数

    Returns:
        (最优参数字典, 最优NSE值, 模拟流量数组)
    """
    model = ModelRegistry.get_model(model_name)
    bounds = list(model.param_bounds.values())
    param_names = list(model.param_bounds.keys())

    if spatial_data is None:
        spatial_data = {}
    spatial_data['timestep'] = timestep

    def objective(params_array):
        params = {k: v for k, v in zip(param_names, params_array)}
        try:
            simulated = model.run(precip, evap, params, spatial_data, temperature)
            nse = calc_nse(observed_flow, simulated)
            if np.isnan(nse) or np.isinf(nse):
                return 1e10
            return -nse
        except Exception as e:
            return 1e10

    result = differential_evolution(
        objective, bounds=bounds, maxiter=max_iter, seed=42, tol=1e-6, polish=True
    )

    best_params = {k: v for k, v in zip(param_names, result.x)}
    best_nse = -result.fun
    simulated = model.run(precip, evap, best_params, spatial_data, temperature)

    return best_params, best_nse, simulated


# ============================================================
# 马斯京根河道演算
# ============================================================
def muskingum_routing(upstream_flow: np.ndarray, k: float, x: float) -> np.ndarray:
    """
    马斯京根(Muskingum)河道汇流演算
    
    Args:
        upstream_flow: 上游来水流量序列
        k: 传播时间参数
        x: 权重因子
        
    Returns:
        演算后的流量序列
    """
    n = len(upstream_flow)
    if n == 0:
        return np.array([])
    
    dt = 1.0
    
    denom = k * (1 - x) + 0.5 * dt
    if denom == 0:
        return upstream_flow.copy()
    
    C0 = (-k * x + 0.5 * dt) / denom
    C1 = (k * x + 0.5 * dt) / denom
    C2 = (k * (1 - x) - 0.5 * dt) / denom
    
    routed = np.zeros(n)
    routed[0] = upstream_flow[0]
    
    for t in range(1, n):
        I = upstream_flow[t]
        I_prev = upstream_flow[t - 1]
        Q_prev = routed[t - 1]
        routed[t] = C0 * I + C1 * I_prev + C2 * Q_prev
    
    return np.maximum(routed, 0)


# 两阶段快速率定 (推荐)
# ============================================================
def calibrate_model_fast(
    model_name: str,
    precip: np.ndarray,
    evap: np.ndarray,
    observed_flow: np.ndarray,
    max_iter: int = 10,
    spatial_data: Dict = None,
    temperature: np.ndarray = None,
    timestep: str = 'daily',
    algorithm: str = 'two_stage',
    algo_params: Dict = None,
    upstream_flow: np.ndarray = None,
    enable_routing: bool = False,
    calib_events: list = None,
    warmup_steps: int = 0,
    progress_callback: callable = None,
) -> Tuple[dict, float, np.ndarray]:
    """
    多算法率定模型参数
    
    支持的算法:
    - 'two_stage': 两阶段算法 (dual_annealing + L-BFGS-B)
    - 'pso': 粒子群优化算法
    - 'ga': 遗传算法
    - 'sce': SCE-UA算法
    - 'de': 差分进化算法
    
    Args:
        model_name: 模型名称
        precip: 降水序列
        evap: 蒸发序列
        observed_flow: 观测流量序列
        max_iter: 最大迭代次数
        spatial_data: 空间数据
        temperature: 温度序列
        timestep: 时间尺度 'hourly' 或 'daily'
        algorithm: 算法选择
        algo_params: 算法参数字典
        upstream_flow: 上游出库流量序列（可选）
        enable_routing: 是否启用上游汇流演算
        calib_events: 率定场次数据列表（每个场次包含precip/evap/flow/upstream）
        warmup_steps: 预热期步数
        
    Returns:
        (最优参数字典, 最优NSE值, 模拟流量数组)
    """
    if spatial_data is None:
        spatial_data = {}
    if 'area' not in spatial_data:
        spatial_data['area'] = 150.7944
    spatial_data['timestep'] = timestep
    
    if algo_params is None:
        algo_params = {}
    
    model = ModelRegistry.get_model(model_name)
    param_names = list(model.param_bounds.keys())
    bounds = list(model.param_bounds.values())
    
    routing_params_added = False
    if enable_routing and upstream_flow is not None and len(upstream_flow) > 0:
        param_names.extend(['k_routing', 'x_routing'])
        bounds.extend([(0.5, 5.0), (0.0, 0.5)])
        routing_params_added = True
    
    n_params = len(param_names)
    
    # 多场次模式：遍历所有率定场次计算平均NSE
    use_multi_events = calib_events is not None and len(calib_events) > 0
    
    def objective(params_array):
        if routing_params_added:
            model_params = {k: v for k, v in zip(param_names[:-2], params_array[:-2])}
            k_rout = params_array[-2]
            x_rout = params_array[-1]
        else:
            model_params = {k: v for k, v in zip(param_names, params_array)}
            k_rout, x_rout = 2.5, 0.25
        
        if hasattr(model, 'validate_params') and not model.validate_params(model_params):
            return 1e10
        
        try:
            if use_multi_events:
                nse_list = []
                for event in calib_events:
                    event_precip = event['precip']
                    event_evap = event['evap']
                    event_flow = event['flow']
                    event_upstream = event.get('upstream')
                    
                    simulated = model.run(event_precip, event_evap, model_params, spatial_data, temperature, warmup_steps)
                    
                    if routing_params_added and event_upstream is not None and len(event_upstream) > 0:
                        routed_upstream = muskingum_routing(event_upstream, k_rout, x_rout)
                        simulated = simulated + routed_upstream
                    
                    # 抹除预热期
                    obs = event_flow[warmup_steps:] if warmup_steps > 0 and len(event_flow) > warmup_steps else event_flow
                    sim = simulated[warmup_steps:] if warmup_steps > 0 and len(simulated) > warmup_steps else simulated
                    
                    nse = calc_nse(obs, sim)
                    if not np.isnan(nse) and not np.isinf(nse):
                        nse_list.append(nse)
                
                avg_nse = np.mean(nse_list) if nse_list else -1e10
                return -avg_nse
            else:
                # 单场次模式（原逻辑）
                simulated = model.run(precip, evap, model_params, spatial_data, temperature, warmup_steps)
                
                if routing_params_added and upstream_flow is not None and len(upstream_flow) > 0:
                    routed_upstream = muskingum_routing(upstream_flow, k_rout, x_rout)
                    simulated = simulated + routed_upstream
                
                # 抹除预热期
                obs = observed_flow[warmup_steps:] if warmup_steps > 0 and len(observed_flow) > warmup_steps else observed_flow
                sim = simulated[warmup_steps:] if warmup_steps > 0 and len(simulated) > warmup_steps else simulated
                
                nse = calc_nse(obs, sim)
                if np.isnan(nse) or np.isinf(nse):
                    return 1e10
                return -nse
        except Exception:
            return 1e10
    
    if algorithm == 'two_stage' or algorithm == '两阶段算法(推荐)':
        best_x, best_nse = _two_stage_optimize(objective, bounds, max_iter, n_params, progress_callback)
    elif algorithm == 'pso' or algorithm == 'PSO':
        best_x, best_nse = _pso_optimize(objective, bounds, max_iter, n_params, algo_params, progress_callback)
    elif algorithm == 'ga' or algorithm == '遗传算法(GA)':
        best_x, best_nse = _ga_optimize(objective, bounds, max_iter, n_params, algo_params, progress_callback)
    elif algorithm == 'sce' or algorithm == 'SCE-UA':
        best_x, best_nse = _sce_optimize(objective, bounds, max_iter, n_params, progress_callback)
    elif algorithm == 'de' or algorithm == '差分进化(DE)':
        best_x, best_nse = _de_optimize(objective, bounds, max_iter, n_params, algo_params, progress_callback)
    else:
        best_x, best_nse = _two_stage_optimize(objective, bounds, max_iter, n_params, progress_callback)
    
    if routing_params_added:
        best_params = {k: v for k, v in zip(param_names[:-2], best_x[:-2])}
        best_params['k_routing'] = best_x[-2]
        best_params['x_routing'] = best_x[-1]
    else:
        best_params = {k: v for k, v in zip(param_names, best_x)}
    
    try:
        if use_multi_events:
            first_event = calib_events[0]
            simulated = model.run(first_event['precip'], first_event['evap'], best_params, spatial_data, temperature, warmup_steps)
            if routing_params_added and first_event.get('upstream') is not None:
                routed_upstream = muskingum_routing(first_event['upstream'], 
                                                    best_params.get('k_routing', 2.5), 
                                                    best_params.get('x_routing', 0.25))
                simulated = simulated + routed_upstream
        else:
            simulated = model.run(precip, evap, best_params, spatial_data, temperature, warmup_steps)
            if routing_params_added and upstream_flow is not None and len(upstream_flow) > 0:
                routed_upstream = muskingum_routing(upstream_flow, best_params.get('k_routing', 2.5), 
                                                    best_params.get('x_routing', 0.25))
                simulated = simulated + routed_upstream
    except Exception:
        simulated = np.full_like(observed_flow, np.nan)
        best_nse = -1e10
    
    return best_params, best_nse, simulated


def _two_stage_optimize(objective, bounds, max_iter, n_params, progress_callback=None):
    if n_params <= 5:
        stage1_iter = max(5, max_iter // 2)
        stage2_iter = max(20, max_iter * 3)
    elif n_params <= 10:
        stage1_iter = max(8, max_iter)
        stage2_iter = max(30, max_iter * 2)
    else:
        stage1_iter = max(5, max_iter // 2)
        stage2_iter = max(15, max_iter)
    
    total_iter = stage1_iter + stage2_iter
    current_iter = [0]
    
    def update_progress_da(x, e, context):
        if progress_callback:
            current_iter[0] += 1
            progress = min(current_iter[0] / total_iter, 1.0)
            progress_callback(progress)
        return False
    
    def update_progress_min(xk):
        if progress_callback:
            current_iter[0] += 1
            progress = min(current_iter[0] / total_iter, 1.0)
            progress_callback(progress)
    
    result1 = dual_annealing(
        objective, bounds=bounds, maxiter=stage1_iter,
        initial_temp=5230, restart_temp_ratio=1e-4,
        visit=2.62, accept=-5.0, no_local_search=True,
        callback=update_progress_da if progress_callback else None,
    )
    current_iter[0] = stage1_iter
    
    result2 = minimize(
        objective, x0=result1.x, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': stage2_iter, 'ftol': 1e-7},
        callback=update_progress_min if progress_callback else None,
    )
    
    if result2.fun <= result1.fun:
        return result2.x, -result2.fun
    return result1.x, -result1.fun


def _pso_optimize(objective, bounds, max_iter, n_params, algo_params, progress_callback=None):
    try:
        from pyswarm import pso
    except ImportError:
        return _two_stage_optimize(objective, bounds, max_iter, n_params, progress_callback)
    
    n_particles = algo_params.get('n_particles', 20)
    omega = algo_params.get('w', 0.7)
    phip = algo_params.get('c1', 1.5)
    phig = algo_params.get('c2', 1.5)
    
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    
    xopt, fopt = pso(objective, lb, ub, swarmsize=n_particles, 
                     maxiter=max_iter, omega=omega, phip=phip, phig=phig)
    if progress_callback:
        progress_callback(1.0)
    return xopt, -fopt


def _ga_optimize(objective, bounds, max_iter, n_params, algo_params, progress_callback=None):
    pop_size = algo_params.get('pop_size', 20)
    n_generations = algo_params.get('n_generations', 50)
    crossover_rate = algo_params.get('crossover_rate', 0.8)
    mutation_rate = algo_params.get('mutation_rate', 0.1)
    
    n_weights = len(bounds)
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    
    np.random.seed(42)
    
    def create_individual():
        return [np.random.uniform(lb[i], ub[i]) for i in range(n_weights)]
    
    pop = [create_individual() for _ in range(pop_size)]
    fitnesses = [objective(np.array(ind)) for ind in pop]
    
    for gen in range(n_generations):
        new_pop = []
        for _ in range(pop_size):
            parent1, parent2 = np.random.choice(len(pop), 2, replace=False)
            child = pop[parent1].copy()
            
            if np.random.random() < crossover_rate:
                crossover_point = np.random.randint(1, n_weights)
                child[crossover_point:] = pop[parent2][crossover_point:]
            
            if np.random.random() < mutation_rate:
                for i in range(n_weights):
                    if np.random.random() < 0.5:
                        child[i] = np.clip(
                            np.random.normal(child[i], 0.1 * (ub[i] - lb[i])),
                            lb[i], ub[i]
                        )
            
            new_pop.append(child)
        
        pop = new_pop
        fitnesses = [objective(np.array(ind)) for ind in pop]
        
        if progress_callback:
            progress_callback((gen + 1) / n_generations)
    
    best_idx = np.argmin(fitnesses)
    return np.array(pop[best_idx]), -fitnesses[best_idx]


def _sce_optimize(objective, bounds, max_iter, n_params, progress_callback=None):
    result = differential_evolution(objective, bounds=bounds, maxiter=max_iter, 
                                     tol=1e-6, polish=True, strategy='best1bin',
                                     callback=lambda xk, convergence: progress_callback(1.0) if progress_callback else None)
    if progress_callback:
        progress_callback(1.0)
    return result.x, -result.fun


def _de_optimize(objective, bounds, max_iter, n_params, algo_params, progress_callback=None):
    mutation_factor = algo_params.get('mutation_factor', 0.8)
    crossover_prob = algo_params.get('crossover_prob', 0.7)
    pop_size = algo_params.get('pop_size', 20)
    
    result = differential_evolution(
        objective, bounds=bounds, maxiter=max_iter, tol=1e-6, polish=True,
        strategy='best1bin', mutation=mutation_factor, recombination=crossover_prob,
        popsize=pop_size,
        callback=lambda xk, convergence: progress_callback(1.0) if progress_callback else None
    )
    if progress_callback:
        progress_callback(1.0)
    return result.x, -result.fun


# ============================================================
# 多模型比较
# ============================================================
def compare_all_models(
    precip: np.ndarray,
    evap: np.ndarray,
    observed_flow: np.ndarray,
    max_iter: int = 30,
    spatial_data: Dict = None,
    temperature: np.ndarray = None,
) -> List[dict]:
    """
    运行所有注册的模型，率定并比较

    Args:
        precip: 降水序列
        evap: 蒸发序列
        observed_flow: 观测流量序列
        max_iter: 最大迭代次数
        spatial_data: 空间数据（如流域面积等）
        temperature: 温度序列（HBV模型需要）

    Returns:
        比较结果列表，按NSE降序排列
    """
    results = []
    model_names = ModelRegistry.list_models()

    for model_name in model_names:
        try:
            best_params, best_nse, simulated = calibrate_model(
                model_name, precip, evap, observed_flow, max_iter,
                spatial_data, temperature
            )

            results.append(
                {
                    "model_name": model_name,
                    "params": best_params,
                    "nse": best_nse,
                    "rmse": calc_rmse(observed_flow, simulated),
                    "mae": calc_mae(observed_flow, simulated),
                    "pbias": calc_pbias(observed_flow, simulated),
                    "simulated": simulated,
                }
            )
        except Exception as e:
            print(f"[ERROR] 模型 {model_name} 率定失败: {e}")
            results.append(
                {
                    "model_name": model_name,
                    "params": {},
                    "nse": -9999,
                    "rmse": 9999,
                    "mae": 9999,
                    "pbias": 9999,
                    "simulated": np.zeros_like(observed_flow),
                }
            )

    # 按 NSE 降序排序
    results.sort(key=lambda x: x["nse"], reverse=True)

    return results


# ============================================================
# 模型参数信息获取
# ============================================================
def get_model_param_info(model_name: str) -> Dict[str, Dict]:
    """
    获取模型的参数描述、单位、范围等信息
    
    Args:
        model_name: 模型名称
        
    Returns:
        参数信息字典 {参数名: {description, unit, bounds}}
    """
    param_info = {
        '水箱模型': {
            'k1': {'description': '快速流调蓄系数', 'unit': '-', 'bounds': (0.01, 0.3)},
            'k2': {'description': '慢速流调蓄系数（基流）', 'unit': '-', 'bounds': (0.001, 0.05)},
            'c': {'description': '产流系数', 'unit': '-', 'bounds': (0.01, 0.3)},
        },
        'HBV模型': {
            'fc': {'description': '田间持水量', 'unit': 'mm', 'bounds': (50.0, 500.0)},
            'beta': {'description': '形状参数', 'unit': '-', 'bounds': (1.0, 5.0)},
            'k0': {'description': '快速出流系数', 'unit': '-', 'bounds': (0.01, 0.5)},
            'k1': {'description': '慢速出流系数', 'unit': '-', 'bounds': (0.001, 0.1)},
            'lp': {'description': '蒸散发限制系数', 'unit': '-', 'bounds': (0.3, 1.0)},
        },
        '新安江模型': {
            'k': {'description': '蒸散发系数', 'unit': '-', 'bounds': (0.5, 1.5)},
            'b': {'description': '蓄水容量曲线指数', 'unit': '-', 'bounds': (0.1, 0.5)},
            'im': {'description': '不透水面积比例', 'unit': '-', 'bounds': (0.01, 0.1)},
            'um': {'description': '上层土壤蓄水容量', 'unit': 'mm', 'bounds': (10, 50)},
            'lm': {'description': '下层土壤蓄水容量', 'unit': 'mm', 'bounds': (50, 150)},
            'dm': {'description': '深层土壤蓄水容量', 'unit': 'mm', 'bounds': (10, 100)},
            'c': {'description': '深层蒸散发系数', 'unit': '-', 'bounds': (0.01, 0.2)},
            'sm': {'description': '自由水蓄水容量', 'unit': 'mm', 'bounds': (10, 80)},
            'ex': {'description': '自由水容量曲线指数', 'unit': '-', 'bounds': (1.0, 2.0)},
            'ki': {'description': '壤中流出流系数', 'unit': '-', 'bounds': (0.3, 0.7)},
            'kg': {'description': '地下水出流系数', 'unit': '-', 'bounds': (0.01, 0.2)},
            'cs': {'description': '流域汇流系数', 'unit': '-', 'bounds': (0.1, 0.5)},
            'l': {'description': '滞后时间', 'unit': 'h', 'bounds': (0, 24)},
            'xg': {'description': '地下水消退系数', 'unit': '-', 'bounds': (0.9, 0.999)},
        },
    }
    
    return param_info.get(model_name, {})


def generate_param_table(model_name: str, params: Dict[str, float]) -> pd.DataFrame:
    """
    生成参数表格
    
    Args:
        model_name: 模型名称
        params: 最优参数字典
        
    Returns:
        DataFrame格式的参数表格
    """
    param_info = get_model_param_info(model_name)
    
    rows = []
    for key, value in params.items():
        info = param_info.get(key, {'description': '-', 'unit': '-', 'bounds': (0, 1)})
        bounds = info.get('bounds', (0, 1))
        rows.append({
            '参数名': key,
            '物理意义': info.get('description', '-'),
            '单位': info.get('unit', '-'),
            '取值范围': f"{bounds[0]:.3f} ~ {bounds[1]:.3f}",
            '最优值': f"{value:.6f}"
        })
    
    return pd.DataFrame(rows)

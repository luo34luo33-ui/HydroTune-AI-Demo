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
    return np.sqrt(np.mean((obs - sim) ** 2))


def calc_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算平均绝对误差"""
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    return np.mean(np.abs(obs - sim))


def calc_pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算百分比偏差"""
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    return 100 * np.sum(obs - sim) / np.sum(obs)


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
) -> Tuple[dict, float, np.ndarray]:
    """
    两阶段快速率定算法
    
    阶段1: dual_annealing - 快速全局搜索
    阶段2: L-BFGS-B - 局部精细优化
    
    Args:
        model_name: 模型名称
        precip: 降水序列
        evap: 蒸发序列
        observed_flow: 观测流量序列
        max_iter: 总迭代次数
        spatial_data: 空间数据
        temperature: 温度序列
        timestep: 时间尺度 'hourly' 或 'daily'
        
    Returns:
        (最优参数字典, 最优NSE值, 模拟流量数组)
    """
    if spatial_data is None:
        spatial_data = {}
    if 'area' not in spatial_data:
        spatial_data['area'] = 150.7944
    spatial_data['timestep'] = timestep
    
    model = ModelRegistry.get_model(model_name)
    bounds = list(model.param_bounds.values())
    param_names = list(model.param_bounds.keys())
    n_params = len(param_names)
    
    def objective(params_array):
        params = {k: v for k, v in zip(param_names, params_array)}
        try:
            simulated = model.run(precip, evap, params, spatial_data, temperature)
            nse = calc_nse(observed_flow, simulated)
            if np.isnan(nse) or np.isinf(nse):
                return 1e10
            return -nse
        except Exception:
            return 1e10
    
    # 根据参数数量调整迭代次数
    if n_params <= 5:
        stage1_iter = max(3, max_iter // 3)
        stage2_iter = max(10, max_iter * 2)
    elif n_params <= 10:
        stage1_iter = max(5, max_iter // 2)
        stage2_iter = max(20, max_iter)
    else:
        stage1_iter = max(3, max_iter // 3)
        stage2_iter = max(10, max_iter)
    
    # 阶段1: dual_annealing 全局搜索
    result1 = dual_annealing(
        objective,
        bounds=bounds,
        maxiter=stage1_iter,
        seed=42,
        initial_temp=5230,
        restart_temp_ratio=1e-4,
        visit=2.62,
        accept=-5.0,
        no_local_search=True,
    )
    
    # 阶段2: L-BFGS-B 局部精细优化
    result2 = minimize(
        objective,
        x0=result1.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': stage2_iter, 'ftol': 1e-7}
    )
    
    # 选择最优结果
    if result2.fun <= result1.fun:
        best_x = result2.x
        best_nse = -result2.fun
    else:
        best_x = result1.x
        best_nse = -result1.fun
    
    best_params = {k: v for k, v in zip(param_names, best_x)}
    simulated = model.run(precip, evap, best_params, spatial_data, temperature)
    
    return best_params, best_nse, simulated


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

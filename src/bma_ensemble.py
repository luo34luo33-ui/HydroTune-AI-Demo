"""
BMA (贝叶斯模型平均) 集成模块
HydroTune-AI - 智能水文模型率定系统

BMA方法通过加权平均多个模型的预测结果来提高预测精度
权重基于各模型在验证集上的NSE表现计算
"""

import numpy as np
from typing import List, Dict, Tuple


def calc_bma_weights(nse_list: List[float], temperature: float = 2.0) -> np.ndarray:
    """
    基于NSE计算BMA权重
    
    参数:
        nse_list: 各模型的NSE值列表
        temperature: 温度参数，用于平滑权重分布（默认2.0）
    
    返回:
        weights: 归一化权重数组
    """
    nse_arr = np.array(nse_list)
    exp_scores = np.exp(nse_arr / temperature)
    weights = exp_scores / np.sum(exp_scores)
    return weights


def apply_bma_ensemble(simulated_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    应用BMA集成
    
    参数:
        simulated_list: 各模型的模拟结果列表
        weights: 各模型的权重
    
    返回:
        ensemble_result: BMA集成结果
    """
    if len(simulated_list) != len(weights):
        raise ValueError("simulated_list长度必须与weights长度一致")
    
    weights = np.array(weights).reshape(-1, 1)
    simulated_stack = np.vstack([s.flatten() for s in simulated_list])
    ensemble = np.sum(simulated_stack * weights, axis=0)
    return ensemble


def calc_bma_metrics(observed: np.ndarray, simulated_list: List[np.ndarray], 
                     weights: np.ndarray) -> Dict[str, float]:
    """
    计算BMA集成的评估指标
    
    参数:
        observed: 实测流量
        simulated_list: 各模型模拟结果
        weights: BMA权重
    
    返回:
        metrics: 包含NSE, RMSE, MAE, PBIAS的字典
    """
    ensemble = apply_bma_ensemble(simulated_list, weights)
    
    nse = calc_nse(observed, ensemble)
    rmse = calc_rmse(observed, ensemble)
    mae = calc_mae(observed, ensemble)
    pbias = calc_pbias(observed, ensemble)
    
    return {
        'nse': nse,
        'rmse': rmse,
        'mae': mae,
        'pbias': pbias
    }


def calc_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算NSE (纳什效率系数)"""
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return -np.inf
    return 1 - numerator / denominator


def calc_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算RMSE (均方根误差)"""
    return np.sqrt(np.mean((observed - simulated) ** 2))


def calc_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算MAE (平均绝对误差)"""
    return np.mean(np.abs(observed - simulated))


def calc_pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算PBIAS (相对偏差)"""
    return 100.0 * np.sum(simulated - observed) / np.sum(observed)


def get_model_weights_dict(model_names: List[str], weights: np.ndarray) -> Dict[str, float]:
    """
    获取模型权重字典
    
    参数:
        model_names: 模型名称列表
        weights: 权重数组
    
    返回:
        weights_dict: 模型名称到权重的字典
    """
    return {name: float(w) for name, w in zip(model_names, weights)}


def format_weights_string(model_names: List[str], weights: np.ndarray) -> str:
    """
    格式化权重字符串用于图例
    
    参数:
        model_names: 模型名称列表
        weights: 权重数组
    
    返回:
        formatted_str: 格式化的权重字符串
    """
    weights_dict = get_model_weights_dict(model_names, weights)
    parts = [f"{name}={w:.2f}" for name, w in weights_dict.items()]
    return ", ".join(parts)

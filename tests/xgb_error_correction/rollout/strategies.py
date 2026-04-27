# -*- coding: utf-8 -*-
"""
误差输入策略模式

根据不同策略，从真实误差和预测误差中选择特征输入
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ErrorInputStrategy(ABC):
    """误差输入策略基类"""
    
    @abstractmethod
    def get_error_lags(self, t: int, predicted_errors: np.ndarray, 
                       true_errors: np.ndarray, k: int, n_lags: int) -> List[float]:
        """获取t时刻的误差滞后序列
        
        Args:
            t: 当前时间步
            predicted_errors: 预测误差序列
            true_errors: 真实误差序列
            k: 窗口大小（k_window策略有效）
            n_lags: 滞后阶数
            
        Returns:
            滞后序列 [e(t-1), e(t-2), ..., e(t-n_lags)]
        """
        pass


class OracleStrategy(ErrorInputStrategy):
    """Oracle策略：理想上界，每步都用真实误差"""
    
    def get_error_lags(self, t: int, predicted_errors: np.ndarray,
                       true_errors: np.ndarray, k: int, n_lags: int) -> List[float]:
        lags = []
        for lag in range(1, n_lags + 1):
            idx = t - lag
            if idx < 0:
                lags.append(0.0)
            else:
                lags.append(true_errors[idx])
        return lags


class FreeRunningStrategy(ErrorInputStrategy):
    """Free-running策略：纯递归，全部用预测误差"""
    
    def get_error_lags(self, t: int, predicted_errors: np.ndarray,
                       true_errors: np.ndarray, k: int, n_lags: int) -> List[float]:
        lags = []
        for lag in range(1, n_lags + 1):
            idx = t - lag
            if idx < 0:
                lags.append(0.0)
            elif idx == 0:
                lags.append(true_errors[0])
            else:
                lags.append(predicted_errors[idx])
        return lags


class KWindowStrategy(ErrorInputStrategy):
    """K-Window策略：前k步用真实误差，后续用预测误差"""
    
    def get_error_lags(self, t: int, predicted_errors: np.ndarray,
                       true_errors: np.ndarray, k: int, n_lags: int) -> List[float]:
        lags = []
        for lag in range(1, n_lags + 1):
            idx = t - lag
            if idx < 0:
                lags.append(0.0)
            elif idx == 0:
                lags.append(true_errors[0])
            elif idx < k:
                lags.append(true_errors[idx])
            else:
                lags.append(predicted_errors[idx])
        return lags


def get_strategy(strategy_name: str) -> ErrorInputStrategy:
    """获取策略实例
    
    Args:
        strategy_name: 策略名称 'oracle', 'free', 'k_window'
        
    Returns:
        策略实例
    """
    strategies = {
        'oracle': OracleStrategy(),
        'free': FreeRunningStrategy(),
        'k_window': KWindowStrategy(),
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name]
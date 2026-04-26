# -*- coding: utf-8 -*-
"""
误差校正模块
从 tests/xgb_error_correction 集成
"""
import numpy as np
from typing import Dict, List, Optional
import os
import sys

XGB_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    pass

DEFAULT_XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 300,
}

N_ERROR_LAGS = 5
N_PRECIP_LAGS = 3


class ErrorCorrector:
    """误差校正器"""
    
    def __init__(
        self,
        n_error_lags: int = N_ERROR_LAGS,
        n_precip_lags: int = N_PRECIP_LAGS,
        xgb_params: Dict = None,
    ):
        self.n_error_lags = n_error_lags
        self.n_precip_lags = n_precip_lags
        self.xgb_params = xgb_params or DEFAULT_XGB_PARAMS
        self.model = None
        self.test_nse = -9999
        self.is_trained = False
    
    def train(
        self,
        precip: np.ndarray,
        flow: np.ndarray,
        simulated: np.ndarray,
        test_ratio: float = 0.3,
    ) -> 'ErrorCorrector':
        """训练误差预测模型
        
        Args:
            precip: 降水序列
            flow: 实测流量
            simulated: 模拟流量
            test_ratio: 测试集比例
            
        Returns:
            self
        """
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost未安装，无法使用误差校正功能")
        
        error = flow - simulated
        
        n = len(error)
        max_lag = max(self.n_error_lags, self.n_precip_lags + 1)
        
        n_test = int((n - max_lag) * test_ratio)
        
        indices = np.random.permutation(n - max_lag)
        test_indices = set(indices[:n_test])
        
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        for t in range(max_lag, n):
            error_lags = [error[t - lag] for lag in range(1, self.n_error_lags + 1)]
            features = self._build_features(error_lags, precip, t)
            target = error[t]
            
            if (t - max_lag) in test_indices:
                test_X.append(features)
                test_y.append(target)
            else:
                train_X.append(features)
                train_y.append(target)
        
        X_train = np.array(train_X)
        y_train = np.array(train_y)
        X_test = np.array(test_X)
        y_test = np.array(test_y)
        
        self.model = XGBRegressor(**self.xgb_params, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.test_nse = calc_nse(y_test, y_pred)
        self.is_trained = True
        
        return self
    
    def train_by_events(
        self,
        precip_events: List[np.ndarray],
        flow_events: List[np.ndarray],
        simulated_events: List[np.ndarray],
        test_event_ratio: float = 0.2,
    ) -> 'ErrorCorrector':
        """按场次划分训练/测试集
        
        Args:
            precip_events: 各场次降水列表
            flow_events: 各场次实测流量列表
            simulated_events: 各场次模拟流量列表
            test_event_ratio: 测试集场次比例（默认20%）
            
        Returns:
            self
        """
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost未安装，无法使用误差校正功能")
        
        n_events = len(precip_events)
        if test_event_ratio > 0:
            n_test_events = max(1, int(n_events * test_event_ratio))
        else:
            n_test_events = 0
        
        indices = np.random.permutation(n_events)
        test_indices = set(indices[:n_test_events]) if n_test_events > 0 else set()
        
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        for idx, (precip, flow, simulated) in enumerate(zip(precip_events, flow_events, simulated_events)):
            error = flow - simulated
            max_lag = max(self.n_error_lags, self.n_precip_lags + 1)
            
            for t in range(max_lag, len(error)):
                error_lags = [error[t - lag] for lag in range(1, self.n_error_lags + 1)]
                features = self._build_features(error_lags, precip, t)
                target = error[t]
                
                if idx in test_indices:
                    test_X.append(features)
                    test_y.append(target)
                else:
                    train_X.append(features)
                    train_y.append(target)
        
        X_train = np.array(train_X)
        y_train = np.array(train_y)
        X_test = np.array(test_X)
        y_test = np.array(test_y)
        
        if len(X_train) == 0:
            raise ValueError("训练数据为空，请确保有足够的场次用于训练")
        
        self.model = XGBRegressor(**self.xgb_params, random_state=42)
        self.model.fit(X_train, y_train)
        
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test)
            self.test_nse = calc_nse(y_test, y_pred)
        else:
            self.test_nse = -9999
        self.is_trained = True
        
        return self
    
    def predict(self, features: np.ndarray) -> float:
        """预测误差"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(features.reshape(1, -1))[0]
    
    def correct(self, precip: np.ndarray, simulated: np.ndarray) -> np.ndarray:
        """校正模拟结果
        
        Args:
            precip: 降水序列
            simulated: 原始模拟流量
            
        Returns:
            校正后的流量
        """
        if not self.is_trained:
            return simulated
        
        n = len(simulated)
        max_lag = max(self.n_error_lags, self.n_precip_lags + 1)
        
        error = np.zeros(n)
        corrected = simulated.copy()
        
        for t in range(max_lag, n):
            error_lags = [error[t - lag] for lag in range(1, self.n_error_lags + 1)]
            features = self._build_features(error_lags, precip, t)
            
            pred_error = self.predict(features)
            error[t] = pred_error
            corrected[t] = simulated[t] + pred_error
        
        return np.maximum(corrected, 0)
    
    def correct_with_true_error_lags(
        self,
        precip: np.ndarray,
        flow: np.ndarray,
        simulated: np.ndarray,
    ) -> np.ndarray:
        """使用真实误差滞后特征进行校正（方案B）
        
        与correct方法的区别：
        - correct: 用XGB预测的误差递归构建特征
        - correct_with_true_error_lags: 用真实误差(flow-simulated)构建特征
        
        Args:
            precip: 降水序列
            flow: 实测流量序列
            simulated: 原始模拟流量
            
        Returns:
            校正后的流量
        """
        if not self.is_trained:
            return simulated
        
        true_error = flow - simulated
        n = len(simulated)
        max_lag = max(self.n_error_lags, self.n_precip_lags + 1)
        
        corrected = simulated.copy()
        
        for t in range(max_lag, n):
            error_lags = [true_error[t - lag] for lag in range(1, self.n_error_lags + 1)]
            features = self._build_features(error_lags, precip, t)
            
            pred_error = self.predict(features)
            corrected[t] = simulated[t] + pred_error
        
        return np.maximum(corrected, 0)
    
    def _build_features(
        self,
        error_lags: List[float],
        precip: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """构建特征向量"""
        features = list(error_lags)
        
        for lag in range(1, self.n_precip_lags + 1):
            if t - lag >= 0:
                features.append(precip[t - lag])
            else:
                features.append(0.0)
        
        return np.array(features)


def calc_nse(o: np.ndarray, s: np.ndarray) -> float:
    """计算NSE"""
    mask = ~(np.isnan(o) | np.isnan(s))
    if mask.sum() < 2:
        return -9999
    o = o[mask]
    s = s[mask]
    denom = np.sum((o - np.mean(o)) ** 2)
    if denom == 0:
        return -9999
    return 1 - np.sum((s - o) ** 2) / denom


def select_best_model(calibration_results: Dict) -> tuple:
    """选择最优模型
    
    Args:
        calibration_results: 率定结果字典
        
    Returns:
        (最优模型名称, 最优结果)
    """
    best_name = None
    best_nse = -9999
    best_result = None
    
    for model_name, result in calibration_results.items():
        if result is None:
            continue
        nse = result.get('nse', -9999)
        if nse > best_nse:
            best_nse = nse
            best_name = model_name
            best_result = result
    
    return best_name, best_result


def apply_error_correction(
    best_name: str,
    best_result: Dict,
    all_precip: np.ndarray,
    all_flow: np.ndarray,
) -> Dict:
    """应用误差校正
    
    Args:
        best_name: 最优模型名称
        best_result: 最优模型结果
        all_precip: 降水序列
        all_flow: 实测流量
        
    Returns:
        包含校正前后结果的字典
    """
    simulated = best_result['simulated']
    
    corrector = ErrorCorrector()
    corrector.train(all_precip, all_flow, simulated)
    
    corrected_sim = corrector.correct(all_precip, simulated)
    
    nse_before = calc_nse(all_flow, simulated)
    nse_after = calc_nse(all_flow, corrected_sim)
    
    return {
        'best_model': best_name,
        'corrector': corrector,
        'simulated': simulated,
        'corrected': corrected_sim,
        'nse_before': nse_before,
        'nse_after': nse_after,
        'nse_improvement': nse_after - nse_before,
    }
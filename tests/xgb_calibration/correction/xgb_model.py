# -*- coding: utf-8 -*-
"""
XGBoost误差修正模型
"""
import numpy as np
from typing import List, Dict, Tuple
from xgboost import XGBRegressor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import N_ERROR_LAGS, N_PRECIP_LAGS, XGB_PARAMS, RANDOM_SEED
from core.data_loader import calc_nse


def build_features(precip: np.ndarray, error: np.ndarray, n_error_lags: int = None) -> np.ndarray:
    """构建XGBoost特征
    
    Args:
        precip: 降水序列
        error: 误差序列
        n_error_lags: 误差滞后阶数（默认使用全局配置）
    """
    n_error_lags = n_error_lags or N_ERROR_LAGS
    n = len(precip)
    max_lag = max(n_error_lags, N_PRECIP_LAGS + 1)
    
    features = []
    for i in range(max_lag, n):
        row = []
        for lag in range(n_error_lags):
            row.append(error[i - lag])
        for lag in range(1, N_PRECIP_LAGS + 1):
            row.append(precip[i - lag])
        features.append(row)
    
    return np.array(features)


class XGBCorrector:
    """XGBoost误差修正器"""
    
    def __init__(self, params: Dict = None, n_error_lags: int = None, mix_warmup: int = None):
        """初始化XGBoost修正器
        
        Args:
            params: XGBoost参数字典
            n_error_lags: 误差滞后阶数（递归步长）
            mix_warmup: 混合特征方案中前期使用真值的步数
        """
        self.params = params or XGB_PARAMS
        self.n_error_lags = n_error_lags or N_ERROR_LAGS
        self.mix_warmup = mix_warmup if mix_warmup is not None else self.n_error_lags
        self.model = None
        self.test_nse = -9999
    
    def train(self, error_events: List[Dict], test_ratio: float = 0.3) -> 'XGBCorrector':
        """训练XGBoost模型（按场次划分训练/测试集）"""
        np.random.seed(RANDOM_SEED)
        
        n_events = len(error_events)
        n_test = int(n_events * test_ratio)
        
        indices = np.random.permutation(n_events)
        test_event_indices = set(indices[:n_test])
        train_event_indices = set(indices[n_test:])
        
        n_lags = self.n_error_lags
        max_lag = max(n_lags, N_PRECIP_LAGS + 1)
        
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        for idx, e in enumerate(error_events):
            precip = e['precip']
            error = e['error']
            
            X = build_features(precip, error, n_lags)
            y = error[max_lag:]
            
            if idx in test_event_indices:
                test_X.append(X)
                test_y.append(y)
            else:
                train_X.append(X)
                train_y.append(y)
        
        X_train = np.vstack(train_X)
        y_train = np.concatenate(train_y)
        X_test = np.vstack(test_X)
        y_test = np.concatenate(test_y)
        
        self.train_event_indices = train_event_indices
        self.test_event_indices = test_event_indices
        
        self.model = XGBRegressor(**self.params, random_state=RANDOM_SEED)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.test_nse = calc_nse(y_test, y_pred)
        
        return self
    
    def correct(self, event: Dict) -> np.ndarray:
        """应用误差修正（混合特征方案）
        
        前期使用真实误差作为特征输入，后续使用预测误差
        这样可以减少误差累积，同时保持一定的预测能力
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        precip = event['precip']
        error = event['error']
        sim = event['sim']
        
        n = len(precip)
        n_lags = self.n_error_lags
        warmup = min(self.mix_warmup, n_lags)
        
        predicted_error = np.zeros_like(error)
        
        predicted_error[:warmup] = error[:warmup]
        
        for t in range(warmup, n):
            features = []
            
            for lag in range(n_lags):
                features.append(predicted_error[t - lag - 1])
            
            for lag in range(1, N_PRECIP_LAGS + 1):
                features.append(precip[t - lag])
            
            e_pred = self.model.predict([features])[0]
            predicted_error[t] = e_pred
        
        corrected_sim = sim - predicted_error
        
        return corrected_sim
    
    def evaluate(self, events: List[Dict], calib_event_names: List[str] = None) -> Dict:
        """评估修正效果"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        calib_set = set(calib_event_names) if calib_event_names else set()
        
        calib_nses_raw = []
        non_calib_nses_raw = []
        calib_nses_corrected = []
        non_calib_nses_corrected = []
        
        for e in events:
            flow = e['flow']
            sim = e['sim']
            name = e['name']
            
            nse_raw = calc_nse(flow, sim)
            corrected_sim = self.correct(e)
            nse_corrected = calc_nse(flow, corrected_sim)
            
            if not np.isnan(nse_raw) and nse_raw > -10:
                if name in calib_set:
                    calib_nses_raw.append(nse_raw)
                else:
                    non_calib_nses_raw.append(nse_raw)
            
            if not np.isnan(nse_corrected) and nse_corrected > -10:
                if name in calib_set:
                    calib_nses_corrected.append(nse_corrected)
                else:
                    non_calib_nses_corrected.append(nse_corrected)
        
        return {
            'calib_nse_raw': np.mean(calib_nses_raw) if calib_nses_raw else -9999,
            'non_calib_nse_raw': np.mean(non_calib_nses_raw) if non_calib_nses_raw else -9999,
            'calib_nse_corrected': np.mean(calib_nses_corrected) if calib_nses_corrected else -9999,
            'non_calib_nse_corrected': np.mean(non_calib_nses_corrected) if non_calib_nses_corrected else -9999,
            'test_nse': self.test_nse,
        }
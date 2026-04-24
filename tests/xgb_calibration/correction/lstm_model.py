# -*- coding: utf-8 -*-
"""
LSTM误差预测模型
"""
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import N_ERROR_LAGS, N_PRECIP_LAGS, LSTM_PARAMS, RANDOM_SEED
from core.data_loader import calc_nse


def build_sequences(precip: np.ndarray, error: np.ndarray, n_error_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """构建LSTM训练序列
    
    Args:
        precip: 降水序列
        error: 误差序列
        n_error_lags: 误差滞后阶数
        
    Returns:
        X: 特征序列 (samples, timesteps, features)
        y: 目标序列 (samples,)
    """
    n = len(precip)
    max_lag = max(n_error_lags, N_PRECIP_LAGS + 1)
    
    X, y = [], []
    
    for i in range(max_lag, n):
        features = []
        
        for lag in range(n_error_lags):
            features.append(error[i - lag])
        
        for lag in range(1, N_PRECIP_LAGS + 1):
            features.append(precip[i - lag])
        
        X.append(features)
        y.append(error[i])
    
    return np.array(X), np.array(y)


class LSTMCorrector:
    """LSTM误差修正器"""
    
    def __init__(self, params: Dict = None, n_error_lags: int = None, mix_warmup: int = None):
        """初始化LSTM修正器
        
        Args:
            params: LSTM参数字典
            n_error_lags: 误差滞后阶数（递归步长）
            mix_warmup: 混合特征方案中前期使用真值的步数
        """
        self.params = params or LSTM_PARAMS
        self.n_error_lags = n_error_lags or N_ERROR_LAGS
        self.mix_warmup = mix_warmup if mix_warmup is not None else self.n_error_lags
        self.model = None
        self.test_nse = -9999
        self.scaler_x = None
        self.scaler_y = None
        
        self._build_model()
    
    def _build_model(self):
        """构建LSTM模型"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            from tensorflow.keras.optimizers import Adam
            
            n_features = self.n_error_lags + N_PRECIP_LAGS
            
            self.model = Sequential([
                LSTM(self.params.get('units', 64), input_shape=(1, n_features)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
                loss='mse'
            )
        except ImportError:
            self.model = None
            warnings.warn("TensorFlow not installed, LSTM model unavailable")
    
    def train(self, error_events: List[Dict], test_ratio: float = 0.3) -> 'LSTMCorrector':
        """训练LSTM模型（按场次划分训练/测试集）"""
        if self.model is None:
            raise ValueError("TensorFlow not available")
        
        np.random.seed(RANDOM_SEED)
        
        n_events = len(error_events)
        n_test = int(n_events * test_ratio)
        
        indices = np.random.permutation(n_events)
        test_event_indices = set(indices[:n_test])
        train_event_indices = set(indices[n_test:])
        
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        for idx, e in enumerate(error_events):
            precip = e['precip']
            error = e['error']
            
            X_seq, y_seq = build_sequences(precip, error, self.n_error_lags)
            
            if idx in test_event_indices:
                test_X.append(X_seq)
                test_y.append(y_seq)
            else:
                train_X.append(X_seq)
                train_y.append(y_seq)
        
        X_train = np.vstack(train_X)
        y_train = np.concatenate(train_y)
        X_test = np.vstack(test_X)
        y_test = np.concatenate(test_y)
        
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        self.model.fit(
            X_train, y_train,
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 32),
            verbose=0
        )
        
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        self.test_nse = calc_nse(y_test, y_pred)
        
        self.train_event_indices = train_event_indices
        self.test_event_indices = test_event_indices
        
        return self
    
    def correct(self, event: Dict) -> np.ndarray:
        """应用误差修正（混合特征方案）"""
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
        
        n_features = n_lags + N_PRECIP_LAGS
        
        for t in range(warmup, n):
            features = []
            
            for lag in range(n_lags):
                features.append(predicted_error[t - lag - 1])
            
            for lag in range(1, N_PRECIP_LAGS + 1):
                features.append(precip[t - lag])
            
            X = np.array(features).reshape(1, 1, n_features)
            e_pred = self.model.predict(X, verbose=0)[0, 0]
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
# -*- coding: utf-8 -*-
"""
线性回归误差预测模型
"""
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LinearRegression

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .base_corrector import BaseCorrector
except ImportError:
    from models.base_corrector import BaseCorrector

try:
    from leadtime_config import LINEAR_PARAMS, RANDOM_SEED
except ImportError:
    from xgb_error_correction.leadtime_config import LINEAR_PARAMS, RANDOM_SEED


class LinearCorrector(BaseCorrector):
    """线性回归误差修正器"""
    
    def __init__(self, params: Dict = None, n_error_lags: int = 5, n_precip_lags: int = 3):
        super().__init__(params, n_error_lags, n_precip_lags)
        self.params = params or LINEAR_PARAMS
    
    def train(self, error_events: List[Dict], test_ratio: float = 0.3) -> 'LinearCorrector':
        """训练线性回归模型"""
        np.random.seed(RANDOM_SEED)
        
        n_events = len(error_events)
        n_test = int(n_events * test_ratio)
        
        indices = np.random.permutation(n_events)
        test_event_indices = set(indices[:n_test])
        
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        n_lags = self.n_error_lags
        max_lag = max(n_lags, self.n_precip_lags + 1)
        
        for idx, e in enumerate(error_events):
            precip = e['precip']
            error = e['error']
            
            for t in range(max_lag, len(error)):
                error_lags = [error[t - lag] for lag in range(1, n_lags + 1)]
                features = self.build_features(error_lags, precip, t)
                target = error[t]
                
                if idx in test_event_indices:
                    test_X.append(features)
                    test_y.append(target)
                else:
                    train_X.append(features)
                    train_y.append(target)
        
        X_train = np.array(train_X)
        y_train = np.array(train_y)
        X_test = np.array(test_X)
        y_test = np.array(test_y)
        
        self.model = LinearRegression(**self.params)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        from ..data_loader import calc_nse
        self.test_nse = calc_nse(y_test, y_pred)
        
        return self
    
    def predict(self, features: np.ndarray) -> float:
        """单步预测误差"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(features.reshape(1, -1))[0]
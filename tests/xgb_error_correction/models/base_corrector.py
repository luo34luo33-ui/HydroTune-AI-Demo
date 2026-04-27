# -*- coding: utf-8 -*-
"""
误差修正器抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np


class BaseCorrector(ABC):
    """误差修正器基类
    
    只负责单步误差预测：e(t) = f(features)
    其中 features = [e(t-1), e(t-2), ..., e(t-n_lags), p(t-1), p(t-2), ...]
    """
    
    def __init__(self, params: Dict = None, n_error_lags: int = 5, n_precip_lags: int = 3):
        """初始化
        
        Args:
            params: 模型参数字典
            n_error_lags: 误差滞后阶数
            n_precip_lags: 降水滞后阶数
        """
        self.params = params or {}
        self.n_error_lags = n_error_lags
        self.n_precip_lags = n_precip_lags
        self.model = None
        self.test_nse = -9999
    
    @abstractmethod
    def train(self, error_events: List[Dict], test_ratio: float = 0.3) -> 'BaseCorrector':
        """训练误差预测模型
        
        Args:
            error_events: 误差场次列表，每个包含 precip, error, sim, flow
            test_ratio: 测试集比例
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """单步预测误差
        
        Args:
            features: 特征数组 [e(t-1), ..., e(t-n_lags), p(t-1), ..., p(t-n_precip_lags)]
            
        Returns:
            预测的误差值 e(t)
        """
        pass
    
    def build_features(self, error_lags: List[float], precip: np.ndarray, t: int) -> np.ndarray:
        """构建特征向量
        
        Args:
            error_lags: 误差滞后序列 [e(t-1), ..., e(t-n_lags)]
            precip: 降水序列
            t: 当前时间步
            
        Returns:
            特征数组
        """
        features = list(error_lags)
        
        for lag in range(1, self.n_precip_lags + 1):
            if t - lag >= 0:
                features.append(precip[t - lag])
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def evaluate(self, events: List[Dict], calib_event_names: List[str] = None) -> Dict:
        """评估修正效果（不进行滚动，只用真实误差特征）"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        try:
            from data_loader import calc_nse
        except ImportError:
            from xgb_error_correction.data_loader import calc_nse
        
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
            if not np.isnan(nse_raw) and nse_raw > -10:
                if name in calib_set:
                    calib_nses_raw.append(nse_raw)
                else:
                    non_calib_nses_raw.append(nse_raw)
            
            corrected_sim = self.correct_with_true_error(e)
            nse_corrected = calc_nse(flow, corrected_sim)
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
    
    def correct_with_true_error(self, event: Dict) -> np.ndarray:
        """使用真实误差进行修正（用于baseline对比）"""
        error = event['error']
        sim = event['sim']
        n_lags = self.n_error_lags
        max_lag = max(n_lags, self.n_precip_lags + 1)
        
        corrected_sim = sim.copy()
        
        for t in range(max_lag, len(error)):
            e_pred = error[t]
            corrected_sim[t] = sim[t] - e_pred
        
        return corrected_sim
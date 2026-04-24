# -*- coding: utf-8 -*-
"""
XGBoost误差修正模型
"""
import numpy as np
from typing import List, Dict, Tuple
from xgboost import XGBRegressor
from .config import N_ERROR_LAGS, N_PRECIP_LAGS, XGB_PARAMS, RANDOM_SEED
from .data_loader import calc_nse


def build_features(precip: np.ndarray, error: np.ndarray) -> np.ndarray:
    """构建XGBoost特征
    
    特征维度: (N_ERROR_LAGS + N_PRECIP_LAGS) = 11
    - 误差滞后: e[t], e[t-1], ..., e[t-7]
    - 降水滞后: p[t-1], p[t-2], p[t-3]
    
    Args:
        precip: 降水序列
        error: 误差序列 (sim - obs)
        
    Returns:
        特征矩阵 shape: (n_samples, n_features)
    """
    n = len(precip)
    max_lag = max(N_ERROR_LAGS, N_PRECIP_LAGS + 1)
    
    features = []
    for i in range(max_lag, n):
        row = []
        for lag in range(N_ERROR_LAGS):
            row.append(error[i - lag])
        for lag in range(1, N_PRECIP_LAGS + 1):
            row.append(precip[i - lag])
        features.append(row)
    
    return np.array(features)


def train_xgb_model(error_events: List[Dict], test_ratio: float = 0.3) -> Tuple[XGBRegressor, float]:
    """训练XGBoost误差修正模型
    
    Args:
        error_events: 误差场次列表，每个包含 precip, error
        test_ratio: 测试集比例
        
    Returns:
        (训练好的XGBoost模型, 测试集NSE)
    """
    np.random.seed(RANDOM_SEED)
    
    all_X = []
    all_y = []
    
    for e in error_events:
        precip = e['precip']
        error = e['error']
        
        X = build_features(precip, error)
        y = error[N_ERROR_LAGS:]
        
        all_X.append(X)
        all_y.append(y)
    
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    
    n_samples = len(X_all)
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test_ratio)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    model = XGBRegressor(**XGB_PARAMS, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    test_nse = calc_nse(y_test, y_pred)
    
    return model, test_nse


def apply_xgb_correction(model: XGBRegressor, event: Dict) -> np.ndarray:
    """应用XGBoost误差修正
    
    Args:
        model: 训练好的XGBoost模型
        event: 场次数据，包含 precip, error
        
    Returns:
        修正后的流量
    """
    precip = event['precip']
    error = event['error']
    sim = event['sim']
    
    X = build_features(precip, error)
    predicted_error = model.predict(X)
    
    n_lags = N_ERROR_LAGS
    corrected_sim = sim.copy()
    corrected_sim[n_lags:] = corrected_sim[n_lags:] - predicted_error
    
    return corrected_sim


def evaluate_with_xgb(train_events: List[Dict], all_events: List[Dict], calib_event_names: list = None) -> Dict:
    """训练XGBoost并在全部场次上评估
    
    Args:
        train_events: 用于训练XGBoost的误差场次
        all_events: 全部场次（用于评估）
        calib_event_names: 率定场次名称列表
        
    Returns:
        dict: 包含各种NSE指标
    """
    xgb_model, xgb_test_nse = train_xgb_model(train_events)
    
    # 区分率定场次和非率定场次
    calib_set = set(calib_event_names) if calib_event_names else set()
    
    calib_nses_raw = []      # 率定期的纯水文模型NSE（未校正）
    non_calib_nses_raw = []  # 非率定场次的纯水文模型NSE
    all_nses_raw = []        # 所有场次的纯水文模型NSE
    
    calib_nses_corrected = []      # 率定期的误差校正后NSE
    non_calib_nses_corrected = []  # 非率定场次的误差校正后NSE
    all_nses_corrected = []        # 所有场次的误差校正后NSE
    
    for e in all_events:
        flow = e['flow']
        sim = e['sim']
        name = e['name']
        
        # 原始NSE
        nse_raw = calc_nse(flow, sim)
        if not np.isnan(nse_raw) and nse_raw > -10:
            all_nses_raw.append(nse_raw)
            if name in calib_set:
                calib_nses_raw.append(nse_raw)
            else:
                non_calib_nses_raw.append(nse_raw)
        
        # 校正后NSE
        corrected_sim = apply_xgb_correction(xgb_model, e)
        nse_corrected = calc_nse(flow, corrected_sim)
        if not np.isnan(nse_corrected) and nse_corrected > -10:
            all_nses_corrected.append(nse_corrected)
            if name in calib_set:
                calib_nses_corrected.append(nse_corrected)
            else:
                non_calib_nses_corrected.append(nse_corrected)
    
    return {
        'calib_nse_raw': np.mean(calib_nses_raw) if calib_nses_raw else -9999,
        'non_calib_nse_raw': np.mean(non_calib_nses_raw) if non_calib_nses_raw else -9999,
        'all_nse_raw': np.mean(all_nses_raw) if all_nses_raw else -9999,
        'calib_nse_corrected': np.mean(calib_nses_corrected) if calib_nses_corrected else -9999,
        'non_calib_nse_corrected': np.mean(non_calib_nses_corrected) if non_calib_nses_corrected else -9999,
        'all_nse_corrected': np.mean(all_nses_corrected) if all_nses_corrected else -9999,
        'xgb_test_nse': xgb_test_nse,
    }
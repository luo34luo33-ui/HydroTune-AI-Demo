# -*- coding: utf-8 -*-
"""
滚动引擎 - 负责多步滚动预测和评估
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from typing import Dict, Tuple

try:
    from .strategies import get_strategy, ErrorInputStrategy
except ImportError:
    from rollout.strategies import get_strategy, ErrorInputStrategy

try:
    from leadtime_config import N_ERROR_LAGS, N_PRECIP_LAGS
except ImportError:
    from xgb_error_correction.leadtime_config import N_ERROR_LAGS, N_PRECIP_LAGS


def rollout(corrector, event: Dict, leadtime: int, strategy_name: str, k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """滚动预测并校正流量
    
    Args:
        corrector: 训练好的误差预测模型
        event: 场次数据，包含 precip, error, sim, flow
        leadtime: 预见期长度（小时），只评估前leadtime步
        strategy_name: 策略名称 'oracle', 'free', 'k_window'
        k: 窗口大小（k_window策略有效）
        
    Returns:
        (predicted_errors, corrected_flow):
            predicted_errors: 预测的误差序列
            corrected_flow: 校正后的流量序列
    """
    precip = event['precip']
    error = event['error']
    sim = event['sim']
    flow = event['flow']
    
    n = len(precip)
    n_lags = N_ERROR_LAGS
    max_lag = max(n_lags, N_PRECIP_LAGS + 1)
    
    strategy = get_strategy(strategy_name)
    
    predicted_errors = np.zeros_like(error)
    predicted_errors[:max_lag] = error[:max_lag]
    
    for t in range(max_lag, n):
        error_lags = strategy.get_error_lags(
            t, predicted_errors, error, k, n_lags
        )
        
        features = corrector.build_features(error_lags, precip, t)
        
        e_pred = corrector.predict(features)
        predicted_errors[t] = e_pred
    
    corrected_flow = sim - predicted_errors
    
    return predicted_errors, corrected_flow


def evaluate_rollout(corrector, event: Dict, leadtime: int, 
                     strategy_name: str, k: int = 0) -> Dict:
    """评估滚动预测效果
    
    只取前leadtime步计算NSE
    
    Args:
        corrector: 训练好的误差预测模型
        event: 场次数据
        leadtime: 预见期长度
        strategy_name: 策略名称
        k: 窗口大小
        
    Returns:
        dict: 包含nse_raw, nse_corrected, delta_nse
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from data_loader import calc_nse
    except ImportError:
        from xgb_error_correction.data_loader import calc_nse
    
    precip = event['precip']
    error = event['error']
    sim = event['sim']
    flow = event['flow']
    
    n = len(precip)
    n_lags = N_ERROR_LAGS
    max_lag = max(n_lags, N_PRECIP_LAGS + 1)
    
    strategy = get_strategy(strategy_name)
    
    predicted_errors = np.zeros_like(error)
    predicted_errors[:max_lag] = error[:max_lag]
    
    for t in range(max_lag, n):
        error_lags = strategy.get_error_lags(
            t, predicted_errors, error, k, n_lags
        )
        
        features = corrector.build_features(error_lags, precip, t)
        
        e_pred = corrector.predict(features)
        predicted_errors[t] = e_pred
    
    corrected_sim = sim - predicted_errors
    
    eval_start = max_lag
    eval_end = n
    
    if eval_end <= eval_start:
        return {
            'nse_raw': -9999,
            'nse_corrected': -9999,
            'delta_nse': -9999,
        }
    
    flow_eval = flow[eval_start:eval_end]
    sim_raw_eval = sim[eval_start:eval_end]
    sim_corrected_eval = corrected_sim[eval_start:eval_end]
    
    nse_raw = calc_nse(flow_eval, sim_raw_eval)
    nse_corrected = calc_nse(flow_eval, sim_corrected_eval)
    
    delta_nse = nse_corrected - nse_raw if nse_corrected > -10 and nse_raw > -10 else -9999
    
    return {
        'nse_raw': nse_raw,
        'nse_corrected': nse_corrected,
        'delta_nse': delta_nse,
    }


def evaluate_all_events(corrector, events: list, leadtime: int,
                        strategy_name: str, k: int = 0,
                        calib_event_names: list = None) -> Dict:
    """在所有场次上评估滚动预测效果
    
    Args:
        corrector: 训练好的误差预测模型
        events: 全部场次列表
        leadtime: 预见期长度
        strategy_name: 策略名称
        k: 窗口大小
        calib_event_names: 率定场次名称列表
        
    Returns:
        dict: 包含各种NSE指标
    """
    calib_set = set(calib_event_names) if calib_event_names else set()
    
    calib_nses_raw = []
    non_calib_nses_raw = []
    calib_nses_corrected = []
    non_calib_nses_corrected = []
    calib_deltas = []
    non_calib_deltas = []
    
    for e in events:
        name = e['name']
        
        result = evaluate_rollout(corrector, e, leadtime, strategy_name, k)
        
        nse_raw = result['nse_raw']
        nse_corrected = result['nse_corrected']
        delta_nse = result['delta_nse']
        
        if nse_raw > -10:
            if name in calib_set:
                calib_nses_raw.append(nse_raw)
            else:
                non_calib_nses_raw.append(nse_raw)
        
        if nse_corrected > -10:
            if name in calib_set:
                calib_nses_corrected.append(nse_corrected)
                calib_deltas.append(delta_nse)
            else:
                non_calib_nses_corrected.append(nse_corrected)
                non_calib_deltas.append(delta_nse)
    
    return {
        'calib_nse_raw': np.mean(calib_nses_raw) if calib_nses_raw else -9999,
        'non_calib_nse_raw': np.mean(non_calib_nses_raw) if non_calib_nses_raw else -9999,
        'all_nse_raw': np.mean(calib_nses_raw + non_calib_nses_raw) if calib_nses_raw + non_calib_nses_raw else -9999,
        'calib_nse_corrected': np.mean(calib_nses_corrected) if calib_nses_corrected else -9999,
        'non_calib_nse_corrected': np.mean(non_calib_nses_corrected) if non_calib_nses_corrected else -9999,
        'all_nse_corrected': np.mean(calib_nses_corrected + non_calib_nses_corrected) if calib_nses_corrected + non_calib_nses_corrected else -9999,
        'calib_delta_nse': np.mean(calib_deltas) if calib_deltas else -9999,
        'non_calib_delta_nse': np.mean(non_calib_deltas) if non_calib_deltas else -9999,
        'all_delta_nse': np.mean(calib_deltas + non_calib_deltas) if calib_deltas + non_calib_deltas else -9999,
    }
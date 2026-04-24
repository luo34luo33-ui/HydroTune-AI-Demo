# -*- coding: utf-8 -*-
"""
基准测试 - 75%大样本率定
"""
import numpy as np
from typing import List, Dict
from .calibrator import Calibrator
from .data_loader import calc_nse
from .config import BENCHMARK_RATIO
from .xgb_model import evaluate_with_xgb


def run_benchmark(model_name: str, events: List[Dict]) -> Dict:
    """运行基准测试 - 75%大样本率定
    
    Args:
        model_name: 模型名称
        events: 全部场次
        
    Returns:
        结果字典
    """
    np.random.seed(42)
    n_events = len(events)
    n_calib = int(n_events * BENCHMARK_RATIO)
    
    indices = np.random.permutation(n_events)
    calib_idx = indices[:n_calib]
    
    calib_events = [events[i] for i in calib_idx]
    calib_names = [events[i]['name'] for i in calib_idx]
    
    print(f"[BENCHMARK] {model_name}: {n_calib}/{n_events} 场率定")
    
    calibrator = Calibrator(model_name)
    params, calib_nse, calib_time, _ = calibrator.calibrate(calib_events)
    
    results = calibrator.run_all_events(params, events)
    
    # 计算各种NSE指标
    xgb_metrics = evaluate_with_xgb(results, results, calib_names)
    
    return {
        'model': model_name,
        'calibration_ratio': f'{int(BENCHMARK_RATIO*100)}%基准',
        'nse_mean': xgb_metrics['all_nse_corrected'],
        'nse_std': 0.0,
        'time_seconds': calib_time,
        'calib_nse': calib_nse,
        'calib_events': ','.join(calib_names),
        # 新增字段
        'calib_nse_raw': xgb_metrics['calib_nse_raw'],
        'non_calib_nse_raw': xgb_metrics['non_calib_nse_raw'],
        'all_nse_raw': xgb_metrics['all_nse_raw'],
        'calib_nse_corrected': xgb_metrics['calib_nse_corrected'],
        'non_calib_nse_corrected': xgb_metrics['non_calib_nse_corrected'],
        'all_nse_corrected': xgb_metrics['all_nse_corrected'],
    }
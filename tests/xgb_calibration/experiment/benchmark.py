# -*- coding: utf-8 -*-
"""
基准测试
"""
import numpy as np
from typing import List, Dict

from algorithms.calibrator import Calibrator
from correction.xgb_model import XGBCorrector
from config import BENCHMARK_RATIO


def run_benchmark(model_name: str, events: List[Dict]) -> Dict:
    """运行基准测试 - 75%大样本率定"""
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
    
    corrector = XGBCorrector()
    corrector.train(results)
    
    metrics = corrector.evaluate(results, calib_names)
    
    return {
        'model': model_name,
        'calibration_ratio': f'{int(BENCHMARK_RATIO*100)}%基准',
        'nse_mean': metrics.get('non_calib_nse_corrected', metrics.get('calib_nse_corrected', -9999)),
        'nse_std': 0.0,
        'time_seconds': calib_time,
        'calib_nse': calib_nse,
        'calib_events': ','.join(calib_names),
        'calib_nse_raw': metrics.get('calib_nse_raw', -9999),
        'non_calib_nse_raw': metrics.get('non_calib_nse_raw', -9999),
        'calib_nse_corrected': metrics.get('calib_nse_corrected', -9999),
        'non_calib_nse_corrected': metrics.get('non_calib_nse_corrected', -9999),
    }
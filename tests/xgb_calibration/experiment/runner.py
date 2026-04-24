# -*- coding: utf-8 -*-
"""
实验运行器
"""
import os
import sys
import numpy as np
import pandas as pd
import time as time_module
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import load_events
from algorithms.calibrator import Calibrator
from correction.xgb_model import XGBCorrector
from config import CALIBRATION_RATIOS, N_RUNS, RANDOM_SEED, OUTPUT_DIR
from visualization.hydrograph import plot_event_list


class Runner:
    """实验运行器"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calibrator = Calibrator(model_name)
    
    def run_experiment(self, events: list, n_calib: int, run_id: int) -> dict:
        """运行单组实验"""
        n_events = len(events)
        indices = np.random.permutation(n_events)
        calib_idx = indices[:n_calib]
        calib_events = [events[i] for i in calib_idx]
        calib_names = [events[i]['name'] for i in calib_idx]
        
        print(f"[EXPERIMENT] {self.model_name}: {n_calib}/{n_events} 场率定 (Run {run_id})")
        
        start_time = time_module.time()
        params, calib_nse, calib_time, _ = self.calibrator.calibrate(calib_events)
        
        results = self.calibrator.run_all_events(params, events)
        
        corrector = XGBCorrector()
        corrector.train(results)
        
        metrics = corrector.evaluate(results, calib_names)
        
        elapsed_time = time_module.time() - start_time
        
        return {
            'model': self.model_name,
            'calibration_ratio': f'{n_calib}场',
            'nse_mean': metrics.get('non_calib_nse_corrected', metrics.get('calib_nse_corrected', -9999)),
            'nse_std': 0.0,
            'time_seconds': elapsed_time,
            'run_iter': run_id,
            'calib_nse': calib_nse,
            'calib_events': ','.join(calib_names),
            'calib_nse_raw': metrics.get('calib_nse_raw', -9999),
            'non_calib_nse_raw': metrics.get('non_calib_nse_raw', -9999),
            'calib_nse_corrected': metrics.get('calib_nse_corrected', -9999),
            'non_calib_nse_corrected': metrics.get('non_calib_nse_corrected', -9999),
        }
    
    def run_batch(self, events: list, n_runs: int = None) -> pd.DataFrame:
        """批量运行实验"""
        from config import N_RUNS
        n_runs = n_runs or N_RUNS
        
        all_results = []
        
        for n_calib in CALIBRATION_RATIOS:
            for run_i in range(n_runs):
                np.random.seed(RANDOM_SEED + run_i)
                
                try:
                    result = self.run_experiment(events, n_calib, run_i + 1)
                    all_results.append(result)
                    print(f"  Run {run_i+1}/{n_runs}: NSE={result['nse_mean']:.4f}")
                except Exception as e:
                    print(f"[ERROR] Run {run_i+1} 失败: {e}")
        
        return pd.DataFrame(all_results)
    
    def run_single(
        self,
        events: List[Dict],
        n_calib: int,
        selector: Optional[object] = None,
        seed: int = None,
    ) -> Dict:
        """单次运行：率定 + XGB校正 + 绘制过程线
        
        Args:
            events: 全部场次
            n_calib: 率定场次数
            selector: 场次选择器（默认为随机选择）
            seed: 随机种子
            
        Returns:
            包含结果和统计信息的字典
        """
        if seed is None:
            seed = RANDOM_SEED
        
        np.random.seed(seed)
        n_events = len(events)
        
        if selector is not None:
            calib_events = selector.select(events, n_calib)
            calib_names = [e['name'] for e in calib_events]
            selector_name = selector.name
        else:
            indices = np.random.permutation(n_events)
            calib_idx = indices[:n_calib]
            calib_events = [events[i] for i in calib_idx]
            calib_names = [events[i]['name'] for i in calib_idx]
            selector_name = "random"
        
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name.upper()}")
        print(f"Selector: {selector_name}")
        print(f"Calibration: {n_calib}/{n_events} events")
        print(f"Selected events: {', '.join(calib_names)}")
        print(f"{'='*60}")
        
        start_time = time_module.time()
        
        params, calib_nse, calib_time, _ = self.calibrator.calibrate(calib_events)
        print(f"[CALIBRATION] Done in {calib_time:.1f}s, NSE={calib_nse:.4f}")
        
        results = self.calibrator.run_all_events(params, events)
        
        calib_names_set = set(calib_names)
        non_calib_results = [r for r in results if r['name'] not in calib_names_set]
        
        print(f"[CORRECTION] Training XGB on {len(non_calib_results)} non-calibration events...")
        corrector = XGBCorrector()
        corrector.train(non_calib_results)
        print(f"[CORRECTION] XGB trained, test NSE={corrector.test_nse:.4f}")
        
        train_events = corrector.train_event_indices
        test_events = corrector.test_event_indices
        
        for result in results:
            result['sim_corrected'] = corrector.correct(result)
            result['params'] = params
        
        print(f"[PLOTTING] Generating hydrographs...")
        plot_stats = plot_event_list(
            events=events,
            results=results,
            model_name=self.model_name,
            selector_name=selector_name,
            n_calib=n_calib,
            test_event_indices=test_events,
        )
        
        elapsed_time = time_module.time() - start_time
        
        print(f"\n[SUMMARY]")
        print(f"  Raw NSE: {plot_stats['nse_raw_mean']:.4f} ± {plot_stats['nse_raw_std']:.4f}")
        print(f"  Corrected NSE: {plot_stats['nse_corrected_mean']:.4f} ± {plot_stats['nse_corrected_std']:.4f}")
        print(f"  Total time: {elapsed_time:.1f}s")
        
        return {
            'model': self.model_name,
            'selector': selector_name,
            'n_calib': n_calib,
            'calib_events': calib_names,
            'params': params,
            'calib_nse': calib_nse,
            'nse_raw_mean': plot_stats['nse_raw_mean'],
            'nse_raw_std': plot_stats['nse_raw_std'],
            'nse_corrected_mean': plot_stats['nse_corrected_mean'],
            'nse_corrected_std': plot_stats['nse_corrected_std'],
            'time_seconds': elapsed_time,
            'results': results,
        }
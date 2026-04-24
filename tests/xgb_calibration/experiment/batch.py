# -*- coding: utf-8 -*-
"""
批量实验运行器
用于系统性地比较不同步长、模型组合的误差校正效果
"""
import os
import sys
import numpy as np
import pandas as pd
import time as time_module
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import load_events, calc_nse
from algorithms.calibrator import Calibrator
from correction.xgb_model import XGBCorrector
from correction.lstm_model import LSTMCorrector
from config import BENCHMARK_RATIO, RANDOM_SEED, MIX_WARMUP_STEPS


class BatchExperiment:
    """批量实验运行器"""
    
    def __init__(
        self,
        model_name: str,
        step_lags: List[int],
        corrector_models: List[str],
        mix_warmup: int = 3,
    ):
        """初始化批量实验
        
        Args:
            model_name: 水文模型名称 (tank/hbv/xaj)
            step_lags: 递归步长列表 [1,2,3,4,5,6,7]
            corrector_models: 误差模型列表 ['xgb', 'lstm']
            mix_warmup: 混合特征方案中前期使用真值的步数
        """
        self.model_name = model_name
        self.step_lags = step_lags
        self.corrector_models = corrector_models
        self.mix_warmup = mix_warmup
        self.results = []
        
        self.calibrator = Calibrator(model_name)
    
    def run_single_config(
        self,
        events: List[Dict],
        step_lag: int,
        corrector_model: str,
    ) -> Dict:
        """运行单组配置"""
        np.random.seed(RANDOM_SEED)
        n_events = len(events)
        n_calib = int(n_events * BENCHMARK_RATIO)
        
        indices = np.random.permutation(n_events)
        calib_idx = indices[:n_calib]
        calib_events = [events[i] for i in calib_idx]
        calib_names = [e['name'] for e in calib_events]
        
        print(f"  Step={step_lag}, Model={corrector_model}...", end=" ", flush=True)
        
        params, calib_nse, calib_time, _ = self.calibrator.calibrate(calib_events)
        
        results = self.calibrator.run_all_events(params, events)
        
        calib_names_set = set(calib_names)
        non_calib_results = [r for r in results if r['name'] not in calib_names_set]
        
        if corrector_model == 'xgb':
            corrector = XGBCorrector(n_error_lags=step_lag, mix_warmup=self.mix_warmup)
        else:
            try:
                corrector = LSTMCorrector(n_error_lags=step_lag, mix_warmup=self.mix_warmup)
            except Exception as e:
                print(f"Error: {e}")
                return None
        
        corrector.train(non_calib_results)
        
        test_events = corrector.test_event_indices
        
        for result in results:
            result['sim_corrected'] = corrector.correct(result)
        
        raw_nses = []
        corrected_nses = []
        
        for idx, r in enumerate(non_calib_results):
            if idx in test_events:
                nse_raw = calc_nse(r['flow'], r['sim'])
                nse_corr = calc_nse(r['flow'], r['sim_corrected'])
                raw_nses.append(nse_raw)
                corrected_nses.append(nse_corr)
        
        result = {
            'step_lag': step_lag,
            'corrector_model': corrector_model,
            'mix_warmup': self.mix_warmup,
            'calib_nse': calib_nse,
            'raw_nse_mean': np.mean(raw_nses) if raw_nses else -9999,
            'raw_nse_std': np.std(raw_nses) if raw_nses else 0,
            'corrected_nse_mean': np.mean(corrected_nses) if corrected_nses else -9999,
            'corrected_nse_std': np.std(corrected_nses) if corrected_nses else 0,
            'xgb_test_nse': corrector.test_nse,
            'n_test_events': len(test_events),
        }
        
        print(f"raw={result['raw_nse_mean']:.4f}, corrected={result['corrected_nse_mean']:.4f}")
        
        return result
    
    def _run_corrector(
        self,
        non_calib_results: List[Dict],
        step_lag: int,
        corrector_model: str,
        calib_nse: float,
    ) -> Dict:
        """运行误差校正（复用率定结果）"""
        print(f"  Step={step_lag}, Model={corrector_model}...", end=" ", flush=True)
        
        if corrector_model == 'xgb':
            corrector = XGBCorrector(n_error_lags=step_lag, mix_warmup=self.mix_warmup)
        else:
            try:
                corrector = LSTMCorrector(n_error_lags=step_lag, mix_warmup=self.mix_warmup)
            except Exception as e:
                print(f"Error: {e}")
                return None
        
        corrector.train(non_calib_results)
        
        test_events = corrector.test_event_indices
        
        for result in non_calib_results:
            result['sim_corrected'] = corrector.correct(result)
        
        raw_nses = []
        corrected_nses = []
        
        for idx, r in enumerate(non_calib_results):
            if idx in test_events:
                nse_raw = calc_nse(r['flow'], r['sim'])
                nse_corr = calc_nse(r['flow'], r['sim_corrected'])
                raw_nses.append(nse_raw)
                corrected_nses.append(nse_corr)
        
        result = {
            'step_lag': step_lag,
            'corrector_model': corrector_model,
            'mix_warmup': self.mix_warmup,
            'calib_nse': calib_nse,
            'raw_nse_mean': np.mean(raw_nses) if raw_nses else -9999,
            'raw_nse_std': np.std(raw_nses) if raw_nses else 0,
            'corrected_nse_mean': np.mean(corrected_nses) if corrected_nses else -9999,
            'corrected_nse_std': np.std(corrected_nses) if corrected_nses else 0,
            'xgb_test_nse': corrector.test_nse,
            'n_test_events': len(test_events),
        }
        
        print(f"raw={result['raw_nse_mean']:.4f}, corrected={result['corrected_nse_mean']:.4f}")
        
        return result
    
    def run(self, events: List[Dict]) -> pd.DataFrame:
        """运行全部实验配置"""
        print(f"\n{'='*60}")
        print(f"Batch Experiment: {self.model_name}")
        print(f"Step lags: {self.step_lags}")
        print(f"Corrector models: {self.corrector_models}")
        print(f"Mix warmup: {self.mix_warmup}")
        print(f"{'='*60}")
        
        np.random.seed(RANDOM_SEED)
        n_events = len(events)
        n_calib = int(n_events * BENCHMARK_RATIO)
        
        indices = np.random.permutation(n_events)
        calib_idx = indices[:n_calib]
        calib_events = [events[i] for i in calib_idx]
        calib_names = [e['name'] for e in calib_events]
        
        print(f"\nCalibrating hydrological model (once for all configs)...")
        params, calib_nse, calib_time, _ = self.calibrator.calibrate(calib_events)
        print(f"Calibration done in {calib_time:.1f}s, NSE={calib_nse:.4f}")
        
        results = self.calibrator.run_all_events(params, events)
        
        calib_names_set = set(calib_names)
        non_calib_results = [r for r in results if r['name'] not in calib_names_set]
        
        all_results = []
        
        for step_lag in self.step_lags:
            for corrector_model in self.corrector_models:
                test_result = self._run_corrector(
                    non_calib_results, step_lag, corrector_model, calib_nse
                )
                
                if test_result:
                    all_results.append(test_result)
        
        self.results = all_results
        return pd.DataFrame(all_results)
    
    def save_results(self, output_path: str):
        """保存结果到CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """打印结果汇总"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"{'='*60}")
        
        for corrector_model in self.corrector_models:
            model_df = df[df['corrector_model'] == corrector_model]
            if model_df.empty:
                continue
            
            print(f"\n{corrector_model.upper()}:")
            print(f"  Step Lag | Raw NSE | Corrected NSE | Delta")
            print(f"  ---------|---------|---------------|------")
            
            for _, row in model_df.iterrows():
                delta = row['corrected_nse_mean'] - row['raw_nse_mean']
                print(f"  {row['step_lag']:8d} | {row['raw_nse_mean']:7.4f} | {row['corrected_nse_mean']:13.4f} | {delta:+7.4f}")
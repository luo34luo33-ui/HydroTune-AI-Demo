# -*- coding: utf-8 -*-
"""
统一运行脚本（增量输出版）
"""
import os
import sys
import numpy as np
import pandas as pd
import time as time_module

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from xgb_error_correction.data_loader import load_events
from xgb_error_correction.calibrator import Calibrator
from xgb_error_correction.xgb_model import evaluate_with_xgb
from xgb_error_correction.benchmark import run_benchmark
from xgb_error_correction.plotter import append_result, update_plots
from xgb_error_correction.config import (
    CALIBRATION_RATIOS, N_RUNS, RANDOM_SEED, OUTPUT_DIR, DEBUG_MODE
)


def run_experiment(model_name: str, events: list, n_calib: int, run_id: int) -> dict:
    """运行单组实验
    
    Args:
        model_name: 模型名称
        events: 全部场次
        n_calib: 率定场次数
        run_id: 当前轮次ID
        
    Returns:
        实验结果
    """
    n_events = len(events)
    indices = np.random.permutation(n_events)
    calib_idx = indices[:n_calib]
    calib_events = [events[i] for i in calib_idx]
    calib_names = [events[i]['name'] for i in calib_idx]
    
    print(f"[EXPERIMENT] {model_name}: {n_calib}/{n_events} 场率定 (Run {run_id})")
    
    calibrator = Calibrator(model_name)
    
    start_time = time_module.time()
    params, calib_nse, calib_time, _ = calibrator.calibrate(calib_events)
    
    results = calibrator.run_all_events(params, events)
    elapsed_time = time_module.time() - start_time
    
    # 计算各种NSE指标
    xgb_metrics = evaluate_with_xgb(results, results, calib_names)
    
    return {
        'model': model_name,
        'calibration_ratio': f'{n_calib}场',
        'nse_mean': xgb_metrics['all_nse_corrected'],
        'nse_std': 0.0,
        'time_seconds': elapsed_time,
        'run_iter': run_id,
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


def main():
    """主函数"""
    print("=" * 60)
    print("XGBoost误差修正模型效果对比实验")
    print("=" * 60)
    
    events = load_events()
    if len(events) == 0:
        print("[ERROR] 未找到洪水场次数据")
        return
    
    print(f"[INFO] 共 {len(events)} 场洪水数据")
    
    models = ['tank', 'hbv', 'xaj']
    all_results = []
    
    for model_name in models:
        print(f"\n{'='*40}")
        print(f"模型: {model_name.upper()}")
        print(f"{'='*40}")
        
        # 基准测试（75%）
        try:
            benchmark_result = run_benchmark(model_name, events)
            benchmark_result['run_iter'] = 0
            benchmark_result['nse_std'] = 0.0
            all_results.append(benchmark_result)
            append_result(benchmark_result)
            print(f"[基准] NSE={benchmark_result['nse_mean']:.4f}, Time={benchmark_result['time_seconds']:.1f}s")
            update_plots()
        except Exception as e:
            print(f"[ERROR] {model_name} 基准测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 实验组（5/10/15场）
        for n_calib in CALIBRATION_RATIOS:
            for run_i in range(N_RUNS):
                np.random.seed(RANDOM_SEED + run_i)
                
                try:
                    result = run_experiment(model_name, events, n_calib, run_i + 1)
                    all_results.append(result)
                    append_result(result)
                    print(f"  Run {run_i+1}/{N_RUNS}: NSE={result['nse_mean']:.4f}")
                    update_plots()
                except Exception as e:
                    print(f"[ERROR] Run {run_i+1} 失败: {e}")
            
            # 每种率定比例完成后打印当前统计
            model_results = [r for r in all_results 
                           if r['model'] == model_name and r['calibration_ratio'] == f'{n_calib}场']
            if model_results:
                avg_nse = np.mean([r['nse_mean'] for r in model_results])
                std_nse = np.std([r['nse_mean'] for r in model_results])
                print(f"[{n_calib}场] 累计 NSE={avg_nse:.4f}±{std_nse:.4f}")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    df = pd.DataFrame(all_results)
    summary = df.groupby(['model', 'calibration_ratio']).agg({
        'nse_mean': ['mean', 'std'],
        'time_seconds': 'mean'
    }).round(4)
    print(summary)
    
    output_path = os.path.join(OUTPUT_DIR, 'xgb_correction_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
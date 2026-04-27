# -*- coding: utf-8 -*-
"""
预见期退化实验调度器（runner2）

自动生成实验组合并执行滚动评估
"""
import os
import sys
import time as time_module
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from leadtime_config import (
    DEBUG_MODE, LEADTIMES, CORRECTOR_MODELS, STRATEGIES, K_VALUES,
    CALIB_RATIOS, SEEDS, OUTPUT_DIR, MAX_ITERATIONS, TIMEOUT_SECONDS,
    RANDOM_SEED
)
from data_loader import load_events
from calibrator import Calibrator
from models.xgb_model import XGBCorrector
from models.linear_model import LinearCorrector
from rollout.rollout_engine import evaluate_all_events


def create_corrector(model_name: str):
    """创建误差预测模型"""
    if model_name == 'xgb':
        return XGBCorrector()
    elif model_name == 'lr':
        return LinearCorrector()
    else:
        raise ValueError(f"Unknown corrector model: {model_name}")


def run_single_experiment(
    events: list,
    model_name: str,
    leadtime: int,
    strategy: str,
    k: int,
    n_calib: int,
    seed: int
) -> dict:
    """运行单组实验
    
    Args:
        events: 全部场次
        model_name: 误差预测模型名称
        leadtime: 预见期
        strategy: 策略名称
        k: 窗口大小
        n_calib: 率定场次数
        seed: 随机种子
        
    Returns:
        实验结果字典
    """
    np.random.seed(RANDOM_SEED + seed)
    
    n_events = len(events)
    indices = np.random.permutation(n_events)
    calib_idx = indices[:n_calib]
    calib_events = [events[i] for i in calib_idx]
    calib_names = [events[i]['name'] for i in calib_idx]
    
    hydro_model = 'xaj'
    
    start_time = time_module.time()
    
    calibrator = Calibrator(hydro_model)
    params, calib_nse, calib_time, _ = calibrator.calibrate(calib_events)
    
    sim_results = calibrator.run_all_events(params, events)
    
    for i, e in enumerate(sim_results):
        e['error'] = e['sim'] - e['flow']
    
    corrector = create_corrector(model_name)
    corrector.train(sim_results, test_ratio=0.3)
    
    eval_results = evaluate_all_events(
        corrector, sim_results, leadtime, strategy, k, calib_names
    )
    
    elapsed_time = time_module.time() - start_time
    
    return {
        'model': model_name,
        'hydro_model': hydro_model,
        'leadtime': leadtime,
        'strategy': strategy,
        'k': k if strategy == 'k_window' else -1,
        'calib_ratio': n_calib,
        'seed': seed,
        'calib_events': ','.join(calib_names),
        'calib_nse_raw': eval_results['calib_nse_raw'],
        'non_calib_nse_raw': eval_results['non_calib_nse_raw'],
        'all_nse_raw': eval_results['all_nse_raw'],
        'calib_nse_corrected': eval_results['calib_nse_corrected'],
        'non_calib_nse_corrected': eval_results['non_calib_nse_corrected'],
        'all_nse_corrected': eval_results['all_nse_corrected'],
        'calib_delta_nse': eval_results['calib_delta_nse'],
        'non_calib_delta_nse': eval_results['non_calib_delta_nse'],
        'all_delta_nse': eval_results['all_delta_nse'],
        'time_seconds': elapsed_time,
    }


def main():
    """主函数"""
    print("=" * 70)
    print("预见期退化实验 (Lead Time Degradation)")
    print("=" * 70)
    
    events = load_events()
    if len(events) == 0:
        print("[ERROR] 未找到洪水场次数据")
        return
    
    print(f"[INFO] 共 {len(events)} 场洪水数据")
    print(f"[INFO] Debug模式: {DEBUG_MODE}")
    print(f"[INFO] Leadtimes: {LEADTIMES}")
    print(f"[INFO] Corrector Models: {CORRECTOR_MODELS}")
    print(f"[INFO] Strategies: {STRATEGIES}")
    print(f"[INFO] K values: {K_VALUES}")
    print(f"[INFO] Calib ratios: {CALIB_RATIOS}")
    print(f"[INFO] Seeds: {SEEDS}")
    
    all_results = []
    total_experiments = 0
    
    for model_name in CORRECTOR_MODELS:
        for leadtime in LEADTIMES:
            for strategy in STRATEGIES:
                k_list = [0] if strategy != 'k_window' else K_VALUES
                for k in k_list:
                    for n_calib in CALIB_RATIOS:
                        for seed in SEEDS:
                            total_experiments += 1
    
    print(f"\n[INFO] 共 {total_experiments} 组实验\n")
    
    exp_id = 0
    
    for model_name in CORRECTOR_MODELS:
        for leadtime in LEADTIMES:
            for strategy in STRATEGIES:
                k_list = [0] if strategy != 'k_window' else K_VALUES
                for k in k_list:
                    for n_calib in CALIB_RATIOS:
                        for seed in SEEDS:
                            exp_id += 1
                            
                            print(f"[{exp_id}/{total_experiments}] "
                                  f"model={model_name}, leadtime={leadtime}, "
                                  f"strategy={strategy}, k={k}, "
                                  f"calib={n_calib}, seed={seed}")
                            
                            try:
                                result = run_single_experiment(
                                    events, model_name, leadtime,
                                    strategy, k, n_calib, seed
                                )
                                result['run_id'] = exp_id
                                all_results.append(result)
                                
                                print(f"  -> NSE_raw={result['all_nse_raw']:.4f}, "
                                      f"NSE_corr={result['all_nse_corrected']:.4f}, "
                                      f"delta={result['all_delta_nse']:.4f}")
                                
                            except Exception as e:
                                print(f"  [ERROR] {e}")
                                import traceback
                                traceback.print_exc()
                                
                                all_results.append({
                                    'run_id': exp_id,
                                    'model': model_name,
                                    'hydro_model': 'xaj',
                                    'leadtime': leadtime,
                                    'strategy': strategy,
                                    'k': k,
                                    'calib_ratio': n_calib,
                                    'seed': seed,
                                    'calib_events': '',
                                    'calib_nse_raw': -9999,
                                    'non_calib_nse_raw': -9999,
                                    'all_nse_raw': -9999,
                                    'calib_nse_corrected': -9999,
                                    'non_calib_nse_corrected': -9999,
                                    'all_nse_corrected': -9999,
                                    'calib_delta_nse': -9999,
                                    'non_calib_delta_nse': -9999,
                                    'all_delta_nse': -9999,
                                    'time_seconds': -1,
                                    'error': str(e),
                                })
    
    df = pd.DataFrame(all_results)
    
    output_path = os.path.join(OUTPUT_DIR, 'leadtime_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] 结果已保存: {output_path}")
    
    print("\n" + "=" * 70)
    print("实验结果汇总 (按 leadtime 和 strategy)")
    print("=" * 70)
    
    summary = df.groupby(['leadtime', 'strategy']).agg({
        'all_nse_raw': 'mean',
        'all_nse_corrected': 'mean',
        'all_delta_nse': 'mean',
    }).round(4)
    print(summary)
    
    summary_path = os.path.join(OUTPUT_DIR, 'leadtime_summary.csv')
    summary.to_csv(summary_path)
    print(f"\n[INFO] 汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
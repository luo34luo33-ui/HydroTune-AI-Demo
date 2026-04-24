# -*- coding: utf-8 -*-
"""
单次运行脚本
用法:
    # 少样本率定 + XGB校正
    python run_single.py --model tank --n-calib 10
    python run_single.py --model tank --n-calib 10 --name tank_experiment
    
    # 基准测试：75%大样本率定 + XGB校正
    python run_single.py --model tank --benchmark
    
    # 指定自定义名称
    python run_single.py --model hbv --n-calib 15
    python run_single.py --model xaj --n-calib 5
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.data_loader import load_events
from experiment.runner import Runner
from config import get_run_output_dir, OUTPUT_DIR, BENCHMARK_RATIO


def run_benchmark(model_name: str, events: list, seed: int = 42):
    """运行基准测试：75%场次率定 + XGB校正"""
    np.random.seed(seed)
    n_events = len(events)
    n_calib = int(n_events * BENCHMARK_RATIO)
    
    indices = np.random.permutation(n_events)
    calib_idx = indices[:n_calib]
    calib_events = [events[i] for i in calib_idx]
    calib_names = [events[i]['name'] for i in calib_idx]
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK MODE")
    print(f"Model: {model_name.upper()}")
    print(f"Calibration: {n_calib}/{n_events} events ({int(BENCHMARK_RATIO*100)}%)")
    print(f"Selected events: {', '.join(calib_names[:5])}... (first 5)")
    print(f"{'='*60}")
    
    from algorithms.calibrator import Calibrator
    from correction.xgb_model import XGBCorrector
    from visualization.hydrograph import plot_event_list
    import time
    
    calibrator = Calibrator(model_name)
    
    start_time = time.time()
    params, calib_nse, calib_time, _ = calibrator.calibrate(calib_events)
    print(f"[CALIBRATION] Done in {calib_time:.1f}s, NSE={calib_nse:.4f}")
    
    results = calibrator.run_all_events(params, events)
    
    calib_names_set = set(calib_names)
    non_calib_results = [r for r in results if r['name'] not in calib_names_set]
    
    print(f"[CORRECTION] Training XGB on {len(non_calib_results)} non-calibration events...")
    corrector = XGBCorrector()
    corrector.train(non_calib_results)
    print(f"[CORRECTION] XGB trained, test NSE={corrector.test_nse:.4f}")
    
    test_events = corrector.test_event_indices
    
    for result in results:
        result['sim_corrected'] = corrector.correct(result)
        result['params'] = params
    
    print(f"[PLOTTING] Generating hydrographs...")
    plot_stats = plot_event_list(
        events=events,
        results=results,
        model_name=model_name,
        selector_name='benchmark',
        n_calib=n_calib,
        test_event_indices=test_events,
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n[SUMMARY]")
    print(f"  Raw NSE: {plot_stats['nse_raw_mean']:.4f} ± {plot_stats['nse_raw_std']:.4f}")
    print(f"  Corrected NSE: {plot_stats['nse_corrected_mean']:.4f} ± {plot_stats['nse_corrected_std']:.4f}")
    print(f"  Total time: {elapsed_time:.1f}s")
    
    return {
        'model': model_name,
        'mode': 'benchmark',
        'n_calib': n_calib,
        'calib_nse': calib_nse,
        'nse_raw_mean': plot_stats['nse_raw_mean'],
        'nse_corrected_mean': plot_stats['nse_corrected_mean'],
        'time_seconds': elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(description='XGB Calibration Single Run')
    parser.add_argument('--model', type=str, default='tank',
                        choices=['tank', 'hbv', 'xaj'],
                        help='Hydrological model name')
    parser.add_argument('--n-calib', type=int, default=10,
                        help='Number of calibration events')
    parser.add_argument('--benchmark', action='store_true',
                        help='Use 75%% of events for calibration (benchmark mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--selector', type=str, default='random',
                        help='Event selection strategy (reserved for future use)')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom run name for output folder (e.g., tank_10ev)')
    
    args = parser.parse_args()
    
    run_dir = get_run_output_dir(args.name)
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Output directory: {run_dir}")
    print("=" * 60)
    
    print("\nLoading flood events...")
    events = load_events()
    
    if len(events) == 0:
        print("[ERROR] No flood events found!")
        return
    
    print(f"Total events: {len(events)}")
    
    if args.benchmark:
        result = run_benchmark(args.model, events, args.seed)
    else:
        if args.n_calib > len(events):
            print(f"[ERROR] n_calib ({args.n_calib}) > total events ({len(events)})")
            args.n_calib = len(events)
        
        runner = Runner(args.model)
        
        result = runner.run_single(
            events=events,
            n_calib=args.n_calib,
            seed=args.seed,
        )
    
    print("\n" + "=" * 60)
    print("Run completed successfully!")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
批量实验脚本
比较不同递归步长和误差模型对误差校正效果的影响

用法:
    python run_batch.py --model xaj --step-lags 1 2 3 4 5 6 7 --models xgb
    python run_batch.py --model xaj --step-lags 1 2 3 4 5 6 7 --models xgb lstm
    python run_batch.py --model xaj --models xgb --mix-warmup 5
"""
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from core.data_loader import load_events
from experiment.batch import BatchExperiment
from config import get_run_output_dir, OUTPUT_DIR, STEP_LAG_RANGE, MIX_WARMUP_STEPS


def main():
    parser = argparse.ArgumentParser(description='Batch Error Correction Experiment')
    parser.add_argument('--model', type=str, default='xaj',
                        choices=['tank', 'hbv', 'xaj'],
                        help='Hydrological model name')
    parser.add_argument('--step-lags', type=int, nargs='+', default=None,
                        help='Recursion step lags (e.g., 1 2 3 4 5 6 7)')
    parser.add_argument('--models', type=str, nargs='+', default=['xgb'],
                        help='Error corrector models (e.g., xgb lstm)')
    parser.add_argument('--mix-warmup', type=int, default=3,
                        help='Number of initial steps using real error values')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom run name for output folder')
    parser.add_argument('--output', type=str, default='batch_results.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    step_lags = args.step_lags or STEP_LAG_RANGE
    run_name = args.name or f"batch_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_dir = get_run_output_dir(run_name)
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
    
    experiment = BatchExperiment(
        model_name=args.model,
        step_lags=step_lags,
        corrector_models=args.models,
        mix_warmup=args.mix_warmup,
    )
    
    results_df = experiment.run(events)
    
    output_path = os.path.join(run_dir, args.output)
    experiment.save_results(output_path)
    
    experiment.print_summary()
    
    print("\n" + "=" * 60)
    print("Batch experiment completed!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
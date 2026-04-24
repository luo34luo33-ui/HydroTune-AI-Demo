# -*- coding: utf-8 -*-
"""
评估器
"""
import numpy as np
from typing import Dict, List

from .data_loader import calc_nse


class Evaluator:
    """评估器 - 计算各种评估指标"""
    
    @staticmethod
    def evaluate_event(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
        """评估单个场次
        
        Returns:
            包含 NSE, RMSE, PBias 等指标的字典
        """
        nse = calc_nse(observed, simulated)
        
        m = ~(np.isnan(observed) | np.isnan(simulated))
        if m.sum() == 0:
            return {'nse': -9999, 'rmse': np.nan, 'pbias': np.nan}
        
        obs = observed[m]
        sim = simulated[m]
        
        rmse = np.sqrt(np.mean((obs - sim) ** 2))
        pbias = 100 * np.sum(sim - obs) / np.sum(obs) if np.sum(obs) != 0 else np.nan
        
        return {
            'nse': nse,
            'rmse': rmse,
            'pbias': pbias,
        }
    
    @staticmethod
    def evaluate_batch(events: List[Dict], sim_key: str = 'sim') -> Dict[str, float]:
        """批量评估
        
        Args:
            events: 场次列表，每个包含 flow 和 sim
            sim_key: 模拟值键名
            
        Returns:
            汇总指标字典
        """
        nses = []
        rmses = []
        pbiases = []
        
        for e in events:
            if sim_key not in e:
                continue
            metrics = Evaluator.evaluate_event(e['flow'], e[sim_key])
            if not np.isnan(metrics['nse']) and metrics['nse'] > -10:
                nses.append(metrics['nse'])
                rmses.append(metrics['rmse'])
                pbiases.append(metrics['pbias'])
        
        return {
            'nse_mean': np.mean(nses) if nses else -9999,
            'nse_std': np.std(nses) if nses else 0,
            'rmse_mean': np.mean(rmses) if rmses else np.nan,
            'pbias_mean': np.mean(pbiases) if pbiases else np.nan,
            'n_events': len(nses),
        }
# -*- coding: utf-8 -*-
"""
数据加载模块
"""
import os
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Dict

from ..configs import DATA_DIR, COL_MAPPING, WARMUP_STEPS


def load_events() -> List[Dict]:
    """加载所有洪水场次数据"""
    events = []
    for f in sorted(glob(os.path.join(DATA_DIR, "*.csv"))):
        try:
            df = pd.read_csv(f)
            df = df.rename(columns={COL_MAPPING[k]: k for k in COL_MAPPING if COL_MAPPING[k] in df.columns})
            if 'precip' not in df.columns or 'flow' not in df.columns:
                continue
            if 'evap' not in df.columns:
                df['evap'] = 0.0
            
            event = {
                'name': os.path.basename(f).replace('.csv', ''),
                'precip': df['precip'].fillna(0).values,
                'evap': df['evap'].fillna(0).values,
                'flow': df['flow'].fillna(0).values,
            }
            
            if 'upstream' in df.columns:
                event['upstream'] = df['upstream'].fillna(0).values
            else:
                event['upstream'] = None
            
            events.append(event)
        except Exception:
            continue
    print(f"[INFO] 加载 {len(events)} 场洪水数据")
    return events


def calc_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算NSE"""
    m = ~(np.isnan(observed) | np.isnan(simulated))
    if m.sum() == 0:
        return -9999
    d = np.sum((observed[m] - np.mean(observed[m])) ** 2)
    return 1 - np.sum((observed[m] - simulated[m]) ** 2) / d if d > 0 else -9999
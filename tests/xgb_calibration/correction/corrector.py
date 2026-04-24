# -*- coding: utf-8 -*-
"""
修正器接口
支持扩展其他误差修正方法
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class Corrector(ABC):
    """误差修正器基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """修正方法名称"""
        pass
    
    @abstractmethod
    def train(self, events: List[Dict]) -> 'Corrector':
        """训练修正模型"""
        pass
    
    @abstractmethod
    def correct(self, event: Dict) -> np.ndarray:
        """对单个场次进行误差修正"""
        pass
    
    def evaluate(self, events: List[Dict], calib_event_names: List[str] = None) -> Dict:
        """评估修正效果"""
        from core.data_loader import calc_nse
        
        calib_set = set(calib_event_names) if calib_event_names else set()
        
        calib_nses_raw = []
        non_calib_nses_raw = []
        calib_nses_corrected = []
        non_calib_nses_corrected = []
        
        for e in events:
            flow = e['flow']
            sim = e['sim']
            name = e['name']
            
            nse_raw = calc_nse(flow, sim)
            corrected_sim = self.correct(e)
            nse_corrected = calc_nse(flow, corrected_sim)
            
            if not np.isnan(nse_raw) and nse_raw > -10:
                if name in calib_set:
                    calib_nses_raw.append(nse_raw)
                else:
                    non_calib_nses_raw.append(nse_raw)
            
            if not np.isnan(nse_corrected) and nse_corrected > -10:
                if name in calib_set:
                    calib_nses_corrected.append(nse_corrected)
                else:
                    non_calib_nses_corrected.append(nse_corrected)
        
        return {
            'calib_nse_raw': np.mean(calib_nses_raw) if calib_nses_raw else -9999,
            'non_calib_nse_raw': np.mean(non_calib_nses_raw) if non_calib_nses_raw else -9999,
            'calib_nse_corrected': np.mean(calib_nses_corrected) if calib_nses_corrected else -9999,
            'non_calib_nse_corrected': np.mean(non_calib_nses_corrected) if non_calib_nses_corrected else -9999,
        }
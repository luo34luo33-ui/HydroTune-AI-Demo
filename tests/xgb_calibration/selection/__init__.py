# -*- coding: utf-8 -*-
"""
场次选择模块

用于扩展智能选择率定场次的算法
- random: 随机选择
- historical: 基于历史NSE选择
- feature: 基于特征选择
- entropy: 基于信息熵选择
"""

from typing import List, Dict

__all__ = ['EventSelector', 'RandomSelector']


class EventSelector:
    """场次选择器基类"""
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    def select(self, events: List[Dict], n_calib: int) -> List[Dict]:
        """选择场次
        
        Args:
            events: 全部场次
            n_calib: 需要选择的场次数
            
        Returns:
            选中的场次列表
        """
        raise NotImplementedError


class RandomSelector(EventSelector):
    """随机选择器"""
    
    @property
    def name(self) -> str:
        return "random"
    
    def select(self, events: List[Dict], n_calib: int) -> List[Dict]:
        import numpy as np
        np.random.seed(42)
        indices = np.random.permutation(len(events))
        return [events[i] for i in indices[:n_calib]]
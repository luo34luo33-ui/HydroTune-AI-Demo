# -*- coding: utf-8 -*-
"""
洪水场次语义编码模块
HydroTune-AI - 智能水文模型率定系统

功能：
1. 将洪水场次数据转化为"事件+阶段+特征+语义描述"的多模态表示
2. 生成结构化JSON编码供LLM使用
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class FloodSemanticCode:
    """洪水场次语义编码数据类"""
    
    event_id: str = ""
    magnitude: str = "small"
    peak_flow: float = 0.0
    flood_volume: float = 0.0
    duration: int = 0
    rise_type: str = "moderate"
    recession_type: str = "moderate"
    peak_shape: str = "rounded"
    rainfall_driven: str = "medium"
    representative_score: float = 0.0
    selection_reason: str = ""
    start_date: str = ""
    end_date: str = ""
    baseflow: float = 0.0
    peak_ratio: float = 0.0
    total_precip: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json_summary(self) -> str:
        """转换为JSON摘要（供LLM使用）"""
        event_id = self.event_id if self.event_id else "Unknown"
        magnitude = self.magnitude if self.magnitude else "unknown"
        rise_type = self.rise_type if self.rise_type else "unknown"
        recession_type = self.recession_type if self.recession_type else "unknown"
        peak_shape = self.peak_shape if self.peak_shape else "unknown"
        rainfall_driven = self.rainfall_driven if self.rainfall_driven else "unknown"
        selection_reason = self.selection_reason if self.selection_reason else "无"
        start_date = self.start_date if self.start_date else "Unknown"
        end_date = self.end_date if self.end_date else "Unknown"
        
        return f'''{{
  "event_id": "{event_id}",
  "magnitude": "{magnitude}",
  "peak_flow": {self.peak_flow:.2f},
  "flood_volume": {self.flood_volume:.2f},
  "duration": {self.duration},
  "rise_type": "{rise_type}",
  "recession_type": "{recession_type}",
  "peak_shape": "{peak_shape}",
  "rainfall_driven": "{rainfall_driven}",
  "representative_score": {self.representative_score:.1f},
  "selection_reason": "{selection_reason}",
  "start_date": "{start_date}",
  "end_date": "{end_date}",
  "baseflow": {self.baseflow:.2f},
  "peak_ratio": {self.peak_ratio:.2f},
  "total_precip": {self.total_precip:.1f}
  }}'''
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FloodSemanticCode':
        """从字典创建"""
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


class FloodSemanticEncoder:
    """洪水场次语义编码器"""
    
    def __init__(self, area: float = 150.7944, timestep: str = 'daily'):
        self.area = area
        self.timestep = timestep
        self.historical_stats = {
            'mean_peak': 0.0,
            'std_peak': 0.0,
            'mean_volume': 0.0,
            'mean_duration': 0.0,
            'mean_precip': 0.0
        }
    
    def encode_single_event(
        self,
        event_id: str,
        flow_data: np.ndarray,
        precip_data: np.ndarray,
        dates: pd.Series,
        start_idx: int,
        end_idx: int,
        baseflow: float = None
    ) -> FloodSemanticCode:
        """对单场洪水进行语义编码
        
        Args:
            event_id: 洪水编号
            flow_data: 流量序列
            precip_data: 降水序列
            dates: 日期序列
            start_idx: 洪水开始索引
            end_idx: 洪水结束索引
            baseflow: 基流（可选）
            
        Returns:
            FloodSemanticCode: 语义编码对象
        """
        event_flow = flow_data[start_idx:end_idx + 1]
        event_precip = precip_data[start_idx:end_idx + 1]
        
        n = len(event_flow)
        if n == 0:
            return FloodSemanticCode(event_id=event_id)
        
        # 基础特征提取
        peak_idx = int(np.argmax(event_flow))
        peak_flow = float(event_flow[peak_idx])
        
        if baseflow is None:
            baseflow = self._estimate_baseflow(event_flow)
        baseflow = float(baseflow)
        
        # 洪水总量 (10⁴ m³)
        direct_runoff = np.sum(event_flow - baseflow)
        
        if self.timestep == 'daily':
            flood_volume = direct_runoff * 24 * 3600 / 10000
        else:
            flood_volume = direct_runoff * 3600 / 10000
        flood_volume = float(flood_volume)
        
        # 历时
        duration = int(n)
        
        # 峰基比
        peak_ratio = peak_flow / baseflow if baseflow > 0 else 0
        
        # 涨洪段分析
        if peak_idx > 0:
            rise_flows = event_flow[:peak_idx + 1]
            rise_rate = (peak_flow - rise_flows[0]) / peak_idx if peak_idx > 0 else 0
        else:
            rise_rate = 0.0
        
        # 退水段分析
        if peak_idx < n - 1:
            recession_flows = event_flow[peak_idx:]
            recession_rate = (recession_flows[0] - recession_flows[-1]) / (n - peak_idx - 1) if (n - peak_idx - 1) > 0 else 0
        else:
            recession_rate = 0.0
        
        # 总降水量
        total_precip = float(np.sum(event_precip))
        
        # 日期处理 - 确保索引正确
        dates_reset = dates.reset_index(drop=True) if hasattr(dates, 'reset_index') else dates
        
        if start_idx < len(dates_reset):
            start_date = dates_reset.iloc[start_idx]
            start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)[:10]
        else:
            start_date_str = "Unknown"
        
        if end_idx < len(dates_reset):
            end_date = dates_reset.iloc[end_idx]
            end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)[:10]
        else:
            end_date_str = "Unknown"
        
        # 语义分类
        magnitude = self._classify_magnitude(peak_flow)
        rise_type = self._classify_rise_type(rise_rate)
        recession_type = self._classify_recession_type(recession_rate)
        peak_shape = self._classify_peak_shape(event_flow, peak_idx)
        rainfall_driven = self._classify_rainfall_driven(total_precip, flood_volume)
        
        # 代表性评分（初步）
        rep_score = self._calc_single_rep_score(peak_flow, flood_volume, magnitude)
        
        # 选取理由
        selection_reason = self._generate_selection_reason(magnitude, peak_flow, rep_score)
        
        return FloodSemanticCode(
            event_id=event_id,
            magnitude=magnitude,
            peak_flow=peak_flow,
            flood_volume=flood_volume,
            duration=duration,
            rise_type=rise_type,
            recession_type=recession_type,
            peak_shape=peak_shape,
            rainfall_driven=rainfall_driven,
            representative_score=rep_score,
            selection_reason=selection_reason,
            start_date=start_date_str,
            end_date=end_date_str,
            baseflow=baseflow,
            peak_ratio=peak_ratio,
            total_precip=total_precip
        )
    
    def encode_multiple_events(
        self,
        flow_data: np.ndarray,
        precip_data: np.ndarray,
        dates: pd.Series,
        event_boundaries: List[Dict],
        historical_peaks: np.ndarray = None
    ) -> List[FloodSemanticCode]:
        """对多场洪水进行语义编码
        
        Args:
            flow_data: 流量序列
            precip_data: 降水序列
            dates: 日期序列
            event_boundaries: 事件边界列表 [{start_idx, end_idx, peak_idx}, ...]
            historical_peaks: 历史峰值数组（用于统计基准）
            
        Returns:
            List[FloodSemanticCode]: 语义编码列表
        """
        # 更新历史统计
        if historical_peaks is not None and len(historical_peaks) > 0:
            self.historical_stats['mean_peak'] = float(np.mean(historical_peaks))
            self.historical_stats['std_peak'] = float(np.std(historical_peaks))
        
        codes = []
        for i, boundary in enumerate(event_boundaries):
            start_idx = boundary.get('start_idx', 0)
            end_idx = boundary.get('end_idx', len(flow_data) - 1)
            
            # 生成基于日期的事件ID - 确保dates索引正确
            dates_reset = dates.reset_index(drop=True) if hasattr(dates, 'reset_index') else dates
            
            if start_idx < len(dates_reset):
                start_date = dates_reset.iloc[start_idx]
                if hasattr(start_date, 'strftime'):
                    event_id = start_date.strftime('%Y%m%d')
                else:
                    try:
                        event_id = pd.to_datetime(start_date).strftime('%Y%m%d')
                    except:
                        event_id = f"F{i+1:03d}"
            else:
                event_id = f"F{i+1:03d}"
            
            # 确保event_id不为空
            if not event_id or event_id.strip() == '':
                event_id = f"F{i+1:03d}"
            
            code = self.encode_single_event(
                event_id=event_id,
                flow_data=flow_data,
                precip_data=precip_data,
                dates=dates,
                start_idx=start_idx,
                end_idx=end_idx
            )
            codes.append(code)
        
        # 计算代表性评分
        self._calculate_representative_scores(codes)
        
        return codes
    
    def _estimate_baseflow(self, flow: np.ndarray, window: int = 5) -> float:
        """估计基流"""
        if len(flow) < window:
            return float(np.mean(flow))
        
        df = pd.Series(flow)
        baseflow = df.rolling(window=window, center=True, min_periods=1).min()
        baseflow = baseflow.ffill().bfill().values
        return float(np.mean(baseflow[:min(window, len(baseflow))]))
    
    def _classify_magnitude(self, peak_flow: float) -> str:
        """分类洪水量级"""
        mean_peak = self.historical_stats['mean_peak']
        std_peak = self.historical_stats['std_peak']
        
        if mean_peak == 0:
            return "medium"
        
        if peak_flow > mean_peak + 2 * std_peak:
            return "extreme"
        elif peak_flow > mean_peak + std_peak:
            return "large"
        elif peak_flow > mean_peak:
            return "medium"
        else:
            return "small"
    
    def _classify_rise_type(self, rise_rate: float) -> str:
        """分类上升段类型"""
        if rise_rate > 5.0:
            return "rapid"
        elif rise_rate > 1.0:
            return "moderate"
        else:
            return "slow"
    
    def _classify_recession_type(self, recession_rate: float) -> str:
        """分类退水段类型"""
        if recession_rate > 3.0:
            return "rapid"
        elif recession_rate > 0.5:
            return "moderate"
        else:
            return "slow"
    
    def _classify_peak_shape(self, flow: np.ndarray, peak_idx: int) -> str:
        """分类峰值形态"""
        n = len(flow)
        if n < 3:
            return "rounded"
        
        # 检查多峰
        local_peaks = []
        for i in range(1, n - 1):
            if flow[i] > flow[i-1] and flow[i] > flow[i+1]:
                local_peaks.append(i)
        
        if len(local_peaks) > 1:
            return "multi-peak"
        
        # 检查峰值锐度
        if peak_idx > 0 and peak_idx < n - 1:
            left_slope = flow[peak_idx] - flow[peak_idx - 1]
            right_slope = flow[peak_idx] - flow[peak_idx + 1]
            
            if left_slope > 3 * right_slope or right_slope > 3 * left_slope:
                return "sharp"
        
        return "rounded"
    
    def _classify_rainfall_driven(self, total_precip: float, flood_volume: float) -> str:
        """分类降雨驱动强度"""
        if total_precip == 0:
            return "low"
        
        runoff_coeff = flood_volume / (total_precip * 0.01) if total_precip > 0 else 0
        
        if runoff_coeff > 0.3:
            return "high"
        elif runoff_coeff > 0.1:
            return "medium"
        else:
            return "low"
    
    def _calc_single_rep_score(self, peak_flow: float, flood_volume: float, magnitude: str) -> float:
        """计算单场洪水的代表性评分"""
        base_score = 50.0
        
        magnitude_scores = {'extreme': 100.0, 'large': 85.0, 'medium': 70.0, 'small': 50.0}
        magnitude_score = magnitude_scores.get(magnitude, 50.0)
        
        return (base_score + magnitude_score) / 2
    
    def _calculate_representative_scores(self, codes: List[FloodSemanticCode]) -> None:
        """计算代表性评分"""
        if not codes:
            return
        
        peaks = np.array([c.peak_flow for c in codes])
        volumes = np.array([c.flood_volume for c in codes])
        
        mean_peak = np.mean(peaks) if len(peaks) > 0 else 1
        mean_volume = np.mean(volumes) if len(volumes) > 0 else 1
        
        for code in codes:
            peak_score = 1 - abs(code.peak_flow - mean_peak) / mean_peak if mean_peak > 0 else 0.5
            volume_score = 1 - abs(code.flood_volume - mean_volume) / mean_volume if mean_volume > 0 else 0.5
            
            magnitude_scores = {'extreme': 100.0, 'large': 85.0, 'medium': 70.0, 'small': 50.0}
            magnitude_score = magnitude_scores.get(code.magnitude, 50.0) / 100.0
            
            code.representative_score = (
                peak_score * 0.3 +
                volume_score * 0.2 +
                magnitude_score * 0.5
            ) * 100
            
            code.selection_reason = self._generate_selection_reason(
                code.magnitude, code.peak_flow, code.representative_score
            )
    
    def _generate_selection_reason(self, magnitude: str, peak_flow: float, rep_score: float) -> str:
        """生成选取理由"""
        if magnitude == 'extreme':
            return f"极端洪水，峰值{peak_flow:.1f}m³/s，代表性最强"
        elif magnitude == 'large':
            return f"较大洪水，峰值{peak_flow:.1f}m³/s"
        elif magnitude == 'medium':
            return f"中等洪水，峰值{peak_flow:.1f}m³/s"
        else:
            return f"小型洪水，峰值{peak_flow:.1f}m³/s"
    
    def to_json_for_llm(self, codes: List[FloodSemanticCode]) -> str:
        """将语义编码列表转换为供LLM使用的JSON格式"""
        if not codes:
            return "[]"
        
        items = []
        for code in codes:
            items.append(code.to_json_summary())
        
        return "[\n  " + ",\n  ".join(items) + "\n]"


def encode_flood_events(
    flow_data: np.ndarray,
    precip_data: np.ndarray,
    dates: pd.Series,
    event_boundaries: List[Dict],
    area: float = 150.7944,
    timestep: str = 'daily'
) -> List[FloodSemanticCode]:
    """便捷函数：对洪水场次进行语义编码
    
    Args:
        flow_data: 流量序列
        precip_data: 降水序列
        dates: 日期序列
        event_boundaries: 事件边界列表
        area: 流域面积
        timestep: 时间尺度
        
    Returns:
        List[FloodSemanticCode]: 语义编码列表
    """
    encoder = FloodSemanticEncoder(area=area, timestep=timestep)
    return encoder.encode_multiple_events(flow_data, precip_data, dates, event_boundaries)
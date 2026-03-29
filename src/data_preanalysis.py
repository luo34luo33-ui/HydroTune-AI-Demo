"""
数据预分析模块
HydroTune-AI - 智能水文模型率定系统

功能：
1. 数据质量评估
2. 洪水事件自动识别
3. 洪水特征统计分析
4. 皮尔逊III型频率分析
5. 代表性洪水智能选取
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import warnings


@dataclass
class FloodEvent:
    """洪水事件数据类"""
    event_id: str
    start_idx: int
    end_idx: int
    start_date: Any
    end_date: Any
    peak_idx: int
    
    # 基础特征
    peak_flow: float = 0.0
    flood_volume: float = 0.0
    duration: int = 0
    baseflow: float = 0.0
    direct_runoff: float = 0.0
    
    # 过程特征
    rise_rate: float = 0.0
    recession_rate: float = 0.0
    rise_duration: int = 0
    recession_duration: int = 0
    
    # 频率特征
    经验频率: float = 0.0
    重现期: float = 0.0
    
    # 代表性评分
    representative_score: float = 0.0
    selection_reason: str = ""
    
    # 原始数据
    flow_data: np.ndarray = field(default_factory=lambda: np.array([]))
    precip_data: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'peak_flow': f"{self.peak_flow:.2f}",
            'flood_volume': f"{self.flood_volume:.2f}",
            'duration': self.duration,
            'baseflow': f"{self.baseflow:.2f}",
            'direct_runoff': f"{self.direct_runoff:.2f}",
            'rise_rate': f"{self.rise_rate:.3f}",
            'recession_rate': f"{self.recession_rate:.3f}",
            'experience_frequency': f"{self.经验频率:.2%}",
            'return_period': f"{self.重现期:.1f}年",
            'score': f"{self.representative_score:.2f}",
            'reason': self.selection_reason
        }


@dataclass
class DataQualityResult:
    """数据质量评估结果"""
    completeness: float = 100.0
    continuity: float = 100.0
    correlation: float = 0.0
    outlier_ratio: float = 0.0
    quality_level: str = "良好"
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            '完整率': f"{self.completeness:.1f}%",
            '连续性': f"{self.continuity:.1f}%",
            '降水-径流相关性': f"{self.correlation:.3f}",
            '极值比例': f"{self.outlier_ratio:.1f}%",
            '质量等级': self.quality_level
        }


@dataclass 
class FrequencyAnalysisResult:
    """频率分析结果"""
    n_samples: int = 0
    mean: float = 0.0
    std: float = 0.0
    cv: float = 0.0
    cs: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # PIII参数
    Cs: float = 0.0
    Cv: float = 0.0
    
    # 设计值
    design_values: Dict[str, float] = field(default_factory=dict)
    
    # 拟合优度
    r_squared: float = 0.0
    
    def to_dict(self) -> Dict:
        result = {
            '样本数': self.n_samples,
            '均值': f"{self.mean:.2f}",
            '标准差': f"{self.std:.2f}",
            '变差系数Cv': f"{self.cv:.3f}",
            '偏度系数Cs': f"{self.cs:.3f}",
        }
        for rp, val in self.design_values.items():
            result[f'{rp}年一遇'] = f"{val:.2f}"
        result['拟合优度R²'] = f"{self.r_squared:.4f}"
        return result


@dataclass
class PreAnalysisResult:
    """预分析完整结果"""
    quality: DataQualityResult
    events: List[FloodEvent]
    frequency: FrequencyAnalysisResult
    selected_events: List[FloodEvent] = field(default_factory=list)
    area: float = 150.7944
    timestep: str = 'daily'
    
    def summary(self) -> Dict:
        return {
            'n_events': len(self.events),
            'n_selected': len(self.selected_events),
            'mean_peak': np.mean([e.peak_flow for e in self.events]) if self.events else 0,
            'mean_volume': np.mean([e.flood_volume for e in self.events]) if self.events else 0,
            'mean_duration': np.mean([e.duration for e in self.events]) if self.events else 0,
        }


class DataPreAnalyzer:
    """数据预分析主类"""
    
    def __init__(self, area: float = 150.7944):
        self.area = area
    
    def analyze(
        self,
        dates: pd.Series,
        precip: np.ndarray,
        flow: np.ndarray,
        timestep: str = 'daily',
        n_select: int = 5
    ) -> PreAnalysisResult:
        """
        执行完整的数据预分析
        
        Args:
            dates: 日期序列
            precip: 降水序列
            flow: 流量序列
            timestep: 时间尺度 ('daily' 或 'hourly')
            n_select: 选取代表性洪水数量
            
        Returns:
            PreAnalysisResult: 预分析结果
        """
        self.area = self.area
        self.timestep = timestep
        
        quality = self.evaluate_quality(precip, flow, dates)
        
        events = self.detect_flood_events(dates, precip, flow)
        
        freq_result = self.frequency_analysis(events)
        
        selected = self.select_representative_floods(events, freq_result, n_select)
        
        return PreAnalysisResult(
            quality=quality,
            events=events,
            frequency=freq_result,
            selected_events=selected,
            area=self.area,
            timestep=timestep
        )
    
    def evaluate_quality(
        self,
        precip: np.ndarray,
        flow: np.ndarray,
        dates: pd.Series
    ) -> DataQualityResult:
        """数据质量评估"""
        n = len(precip)
        
        completeness = (1 - np.isnan(precip).sum() / n) * 100
        completeness = (1 - np.isnan(flow).sum() / n) * 100
        
        continuity = 100.0
        if len(dates) >= 2:
            if pd.api.types.is_datetime64_any_dtype(dates):
                diffs = pd.Series(dates.diff().dropna())
                if len(diffs) > 0:
                    expected_diff = pd.Timedelta(hours=24 if self.timestep == 'daily' else 1)
                    continuous_count = ((diffs - expected_diff).dt.total_seconds().abs().sum())
                    continuity = max(0, 100 - continuous_count / len(diffs) * 10)
        
        correlation = 0.0
        valid_mask = ~(np.isnan(precip) | np.isnan(flow))
        if valid_mask.sum() > 10:
            correlation = np.corrcoef(precip[valid_mask], flow[valid_mask])[0, 1]
            correlation = max(0, correlation)
        
        precip_mean = np.nanmean(precip)
        precip_std = np.nanstd(precip)
        outlier_count = np.sum(np.abs(precip - precip_mean) > 3 * precip_std)
        precip_outlier_ratio = outlier_count / n * 100
        
        flow_mean = np.nanmean(flow)
        flow_std = np.nanstd(flow)
        outlier_count = np.sum(np.abs(flow - flow_mean) > 3 * flow_std)
        flow_outlier_ratio = outlier_count / n * 100
        
        outlier_ratio = max(precip_outlier_ratio, flow_outlier_ratio)
        
        issues = []
        if completeness < 95:
            issues.append(f"数据完整率仅{completeness:.1f}%，存在缺失值")
        if continuity < 95:
            issues.append(f"时间连续性仅{continuity:.1f}%，可能存在断点")
        if correlation < 0.3:
            issues.append(f"降水-径流相关性偏低(r={correlation:.2f})，需检验数据一致性")
        if outlier_ratio > 5:
            issues.append(f"极值比例为{outlier_ratio:.1f}%，存在异常值")
        
        if completeness >= 95 and continuity >= 95 and correlation >= 0.5 and outlier_ratio <= 5:
            quality_level = "优秀"
        elif completeness >= 90 and continuity >= 90 and correlation >= 0.3 and outlier_ratio <= 10:
            quality_level = "良好"
        elif completeness >= 80:
            quality_level = "一般"
        else:
            quality_level = "较差"
        
        return DataQualityResult(
            completeness=completeness,
            continuity=continuity,
            correlation=correlation,
            outlier_ratio=outlier_ratio,
            quality_level=quality_level,
            issues=issues
        )
    
    def detect_flood_events(
        self,
        dates: pd.Series,
        precip: np.ndarray,
        flow: np.ndarray,
        threshold_ratio: float = 0.2
    ) -> List[FloodEvent]:
        """
        基于斜率变化的洪水事件识别
        
        Args:
            dates: 日期序列
            precip: 降水序列
            flow: 流量序列
            threshold_ratio: 阈值比例（相对于峰值）
            
        Returns:
            洪水事件列表
        """
        n = len(flow)
        if n < 10:
            return []
        
        flow = np.array(flow)
        precip = np.array(precip)
        
        baseflow = self.estimate_baseflow(flow)
        
        flow_anomaly = flow - baseflow
        
        peak_threshold = np.mean(flow_anomaly) + threshold_ratio * np.std(flow_anomaly)
        
        is_flood = flow_anomaly > peak_threshold
        
        diff = np.diff(is_flood.astype(int))
        
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]:
                starts = np.concatenate([[0], starts])
            if starts[-1] > ends[-1]:
                ends = np.concatenate([ends, [n - 1]])
        
        events = []
        event_id = 1
        
        valid_pairs = []
        for start in starts:
            for end in ends:
                if end > start:
                    duration = end - start + 1
                    if duration >= 5:
                        valid_pairs.append((start, end))
                        break
        
        for start, end in valid_pairs:
            duration = end - start + 1
            
            event_flow = flow[start:end + 1]
            event_precip = precip[start:end + 1]
            
            peak_idx = np.argmax(event_flow)
            peak_flow = event_flow[peak_idx]
            
            event_baseflow = np.mean([baseflow[start], baseflow[end]])
            
            direct_runoff = np.sum(event_flow - event_baseflow)
            flood_volume = direct_runoff * (24 if self.timestep == 'daily' else 1) * 3600 / 10000
            
            rise_idx = np.argmax(event_flow[:peak_idx + 1]) if peak_idx > 0 else 0
            rise_duration = peak_idx - rise_idx
            rise_rate = (peak_flow - event_flow[rise_idx]) / max(1, rise_duration) if rise_duration > 0 else 0
            
            recession_duration = duration - peak_idx
            if recession_duration > 0 and peak_idx < len(event_flow) - 1:
                recession_rate = (event_flow[peak_idx] - event_flow[-1]) / recession_duration
            else:
                recession_rate = 0
            
            if hasattr(dates, 'iloc'):
                start_date = dates.iloc[start] if start < len(dates) else dates.iloc[0]
                end_date = dates.iloc[end] if end < len(dates) else dates.iloc[-1]
                peak_date = dates.iloc[start + peak_idx] if (start + peak_idx) < len(dates) else dates.iloc[0]
            else:
                start_date = dates[start] if start < len(dates) else dates[0]
                end_date = dates[end] if end < len(dates) else dates[-1]
                peak_date = dates[start + peak_idx] if (start + peak_idx) < len(dates) else dates[0]
            
            event = FloodEvent(
                event_id=f"F{event_id:03d}",
                start_idx=start,
                end_idx=end,
                start_date=start_date,
                end_date=end_date,
                peak_idx=peak_idx,
                peak_flow=peak_flow,
                flood_volume=flood_volume,
                duration=duration,
                baseflow=event_baseflow,
                direct_runoff=direct_runoff,
                rise_rate=rise_rate,
                recession_rate=recession_rate,
                rise_duration=rise_duration,
                recession_duration=recession_duration,
                flow_data=event_flow,
                precip_data=event_precip
            )
            
            events.append(event)
            event_id += 1
        
        events.sort(key=lambda x: x.start_idx)
        
        return events
    
    def detect_flood_events_by_slope(
        self,
        flow: np.ndarray,
        dates: pd.Series,
        precip_threshold: float = 1.0,
        flow_threshold: float = None,
        min_duration: int = 5,
        min_peak_ratio: float = 1.5
    ) -> List[Dict]:
        """
        基于斜率变化的洪水事件识别（仅使用流量数据）
        
        Args:
            flow: 流量序列
            dates: 日期序列
            precip_threshold: 降水阈值（仅用于参考）
            flow_threshold: 流量阈值，默认使用70分位值
            min_duration: 最小洪水历时
            min_peak_ratio: 最小峰基比
            
        Returns:
            洪水事件字典列表
        """
        n = len(flow)
        if n < 10:
            return []
        
        flow = np.array(flow)
        
        if flow_threshold is None:
            flow_threshold = np.percentile(flow, 70)
        
        baseflow = self.estimate_baseflow(flow)
        
        slope = np.diff(flow)
        slope = np.concatenate([[0], slope])
        
        rise_threshold = np.std(slope) * 0.5
        
        is_rising = slope > rise_threshold
        
        is_flood = flow > flow_threshold
        
        combined = is_flood & (np.abs(slope) > rise_threshold * 0.3)
        
        transitions = np.diff(combined.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            return []
        
        if starts[0] > ends[0]:
            starts = np.concatenate([[0], starts])
        if starts[-1] > ends[-1]:
            ends = np.concatenate([ends, [n - 1]])
        
        events = []
        
        for start, end in zip(starts, ends):
            duration = end - start + 1
            if duration < min_duration:
                continue
            
            event_flow = flow[start:end + 1]
            event_baseflow = np.mean([baseflow[start], baseflow[end]])
            
            peak_idx = np.argmax(event_flow)
            peak_flow = event_flow[peak_idx]
            
            if peak_flow < flow_threshold * min_peak_ratio:
                continue
            
            rise_rates = []
            for i in range(1, peak_idx + 1):
                if event_flow[i] > event_flow[i-1]:
                    rise_rates.append(event_flow[i] - event_flow[i-1])
            
            rise_rate = np.mean(rise_rates) if rise_rates else 0
            
            recession_rates = []
            for i in range(peak_idx, len(event_flow) - 1):
                if event_flow[i] > event_flow[i+1]:
                    recession_rates.append(event_flow[i] - event_flow[i+1])
            
            recession_rate = np.mean(recession_rates) if recession_rates else 0
            
            direct_runoff = np.sum(event_flow - event_baseflow)
            flood_volume = direct_runoff * (24 if self.timestep == 'daily' else 1) * 3600 / 10000
            
            if rise_rate > 0 and recession_rate > 0:
                peak_ratio = rise_rate / recession_rate
                if peak_ratio > 2:
                    peak_type = "陡涨陡落型"
                elif peak_ratio > 0.8:
                    peak_type = "均匀对称型"
                else:
                    peak_type = "缓涨缓落型"
            elif rise_rate > recession_rate:
                peak_type = "陡涨缓落型"
            else:
                peak_type = "缓涨陡落型"
            
            event_start_date = dates.iloc[start] if start < len(dates) else dates.iloc[0]
            event_end_date = dates.iloc[end] if end < len(dates) else dates.iloc[-1]
            
            events.append({
                'event_id': f'F{len(events)+1:03d}',
                'start_idx': start,
                'end_idx': end,
                'start_date': event_start_date,
                'end_date': event_end_date,
                'peak_idx': start + peak_idx,
                'peak_flow': peak_flow,
                'baseflow': event_baseflow,
                'direct_runoff': direct_runoff,
                'flood_volume': flood_volume,
                'duration': duration,
                'rise_rate': rise_rate,
                'recession_rate': recession_rate,
                'peak_type': peak_type
            })
        
        return events
    
    def frequency_analysis_pearson(self, peaks: np.ndarray) -> 'FrequencyAnalysisResult':
        """
        皮尔逊III型频率分析（直接使用峰值数组）
        
        Args:
            peaks: 洪峰流量数组
            
        Returns:
            频率分析结果
        """
        from scipy import stats
        
        if len(peaks) < 3:
            return FrequencyAnalysisResult(
                n_samples=len(peaks),
                mean=np.mean(peaks) if len(peaks) > 0 else 0,
                std=np.std(peaks, ddof=1) if len(peaks) > 0 else 0,
                cv=np.std(peaks)/np.mean(peaks) if np.mean(peaks) > 0 else 0,
                cs=0,
                design_values={}
            )
        
        peaks = np.array(peaks)
        n = len(peaks)
        
        mean_val = np.mean(peaks)
        std_val = np.std(peaks, ddof=1)
        cv = std_val / mean_val if mean_val > 0 else 0
        
        sorted_peaks = np.sort(peaks)[::-1]
        
        cs = 0
        if std_val > 0:
            cs = np.sum((peaks - mean_val) ** 3) / (n * std_val ** 3)
        
        design_values = {}
        return_periods = [2, 5, 10, 20, 50, 100]
        
        for rp in return_periods:
            p = 1 - 1 / rp
            try:
                if abs(cs) < 0.01:
                    z = stats.norm.ppf(p)
                    k = z
                else:
                    shape = max(0.1, 4 / (cs ** 2))
                    scale = cv / np.sqrt(shape)
                    loc = 1 - shape * scale
                    
                    x = stats.gamma.ppf(p, a=shape, loc=loc, scale=scale)
                    k = (x - loc) / scale - 1 / np.sqrt(shape)
                
                design_value = mean_val * (1 + k * cv)
                design_values[f'{rp}年'] = max(0, design_value)
            except:
                design_values[f'{rp}年'] = mean_val
        
        return FrequencyAnalysisResult(
            n_samples=n,
            mean=mean_val,
            std=std_val,
            cv=cv,
            cs=cs,
            skewness=cs,
            design_values=design_values
        )
    
    def estimate_baseflow(self, flow: np.ndarray, window: int = 5) -> np.ndarray:
        """
        估计基流（滚动最小值法）
        """
        if len(flow) < window:
            return np.full_like(flow, np.nanmean(flow))
        
        df = pd.Series(flow)
        baseflow = df.rolling(window=window, center=True, min_periods=1).min()
        baseflow = baseflow.ffill().bfill().values
        
        return baseflow
    
    def frequency_analysis(self, events: List[FloodEvent]) -> FrequencyAnalysisResult:
        """
        皮尔逊III型频率分析
        """
        if len(events) < 3:
            return FrequencyAnalysisResult()
        
        flood_volumes = np.array([e.flood_volume for e in events])
        peak_flows = np.array([e.peak_flow for e in events])
        
        n = len(flood_volumes)
        
        mean_vol = np.mean(flood_volumes)
        std_vol = np.std(flood_volumes, ddof=1)
        mean_peak = np.mean(peak_flows)
        std_peak = np.std(peak_flows, ddof=1)
        
        cv_vol = std_vol / mean_vol if mean_vol > 0 else 0
        cv_peak = std_peak / mean_peak if mean_peak > 0 else 0
        
        cv = cv_peak
        
        sorted_peaks = np.sort(peak_flows)[::-1]
        m = np.arange(1, n + 1)
        experience_freqs = m / (n + 1)
        
        cs = 0
        if n > 2:
            mean_val = np.mean(sorted_peaks)
            std_val = np.std(sorted_peaks, ddof=1)
            if std_val > 0:
                cs = np.sum((sorted_peaks - mean_val) ** 3) / (n * std_val ** 3)
        
        design_values = {}
        return_periods = ['2', '5', '10', '20', '50', '100']
        
        for rp_str in return_periods:
            rp = float(rp_str)
            p = 1 - 1 / rp
            k = self._calculate_piii_quantile(p, cv, cs)
            design_value = mean_peak * (1 + k * cv)
            design_values[f'{rp_str}年'] = max(0, design_value)
            
            for event in events:
                if event.重现期 == 0:
                    if event.peak_flow >= design_value:
                        pass
        
        r_squared = self._calculate_r_squared(sorted_peaks, experience_freqs, mean_peak, cv)
        
        return FrequencyAnalysisResult(
            n_samples=n,
            mean=mean_peak,
            std=std_peak,
            cv=cv,
            cs=cs,
            skewness=cs,
            design_values=design_values,
            r_squared=r_squared
        )
    
    def _calculate_piii_quantile(self, p: float, cv: float, cs: float) -> float:
        """
        计算PIII型分布的分位数值（近似）
        """
        from scipy import stats
        
        if abs(cs) < 0.01:
            z = stats.norm.ppf(p)
            return z
        
        try:
            shape = 4 / (cs ** 2)
            scale = cv / np.sqrt(shape)
            loc = 1 - shape * scale
            
            x = stats.gamma.ppf(p, a=shape, loc=loc, scale=scale)
            k = (x - loc) / scale - 1 / np.sqrt(shape)
            return k
        except:
            z = stats.norm.ppf(p)
            return z
    
    def _calculate_r_squared(self, observed, theoretical, mean, cv):
        """计算拟合优度"""
        if len(observed) < 2:
            return 0
        
        ss_res = np.sum((observed - theoretical) ** 2)
        ss_tot = np.sum((observed - mean) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return max(0, 1 - ss_res / ss_tot)
    
    def select_representative_floods(
        self,
        events: List[FloodEvent],
        freq_result: FrequencyAnalysisResult,
        n_select: int = 5
    ) -> List[FloodEvent]:
        """
        智能选取代表性洪水
        """
        if len(events) <= n_select:
            for i, e in enumerate(events):
                e.representative_score = 100.0
                e.selection_reason = "洪水场次较少，全部选取"
            return events
        
        peaks = np.array([e.peak_flow for e in events])
        volumes = np.array([e.flood_volume for e in events])
        
        peak_rank = np.argsort(np.argsort(-peaks))
        volume_rank = np.argsort(np.argsort(-volumes))
        
        mean_peak = np.mean(peaks)
        mean_volume = np.mean(volumes)
        
        for i, event in enumerate(events):
            peak_score = 1 - abs(event.peak_flow - mean_peak) / mean_peak
            volume_score = 1 - abs(event.flood_volume - mean_volume) / mean_volume
            
            freq_score = 0.5
            if event.peak_flow >= freq_result.design_values.get('10年', float('inf')):
                freq_score = 1.0
                event.selection_reason = "接近设计洪水"
            elif event.peak_flow >= freq_result.design_values.get('5年', float('inf')):
                freq_score = 0.8
                event.selection_reason = "中等量级洪水"
            else:
                freq_score = 0.6
                event.selection_reason = "常遇洪水"
            
            event.representative_score = (
                freq_score * 0.25 +
                peak_score * 0.20 +
                volume_score * 0.20 +
                (1 - peak_rank[i] / len(events)) * 0.15 +
                0.20
            ) * 100
        
        sorted_events = sorted(events, key=lambda x: x.representative_score, reverse=True)
        
        selected = []
        remaining = sorted_events.copy()
        
        target_counts = {
            'extreme': max(1, n_select // 5),
            'high': max(1, n_select // 3),
            'medium': n_select - 2 * max(1, n_select // 5)
        }
        
        counts = {'extreme': 0, 'high': 0, 'medium': 0}
        
        for event in sorted_events:
            if len(selected) >= n_select:
                break
            
            event_floor = event.peak_flow // (mean_peak * 0.5)
            
            if counts['extreme'] < target_counts['extreme']:
                selected.append(event)
                counts['extreme'] += 1
                event.selection_reason = f"极端洪水(>{mean_peak * 2:.1f}m³/s)，率定极端响应"
            elif counts['high'] < target_counts['high']:
                selected.append(event)
                counts['high'] += 1
                event.selection_reason = f"较大洪水({mean_peak * 1.5:.1f}-{mean_peak * 2:.1f}m³/s)"
            elif counts['medium'] < target_counts['medium']:
                selected.append(event)
                counts['medium'] += 1
                event.selection_reason = f"中等洪水({mean_peak * 0.5:.1f}-{mean_peak * 1.5:.1f}m³/s)"
        
        while len(selected) < n_select and remaining:
            for event in remaining:
                if event not in selected:
                    selected.append(event)
                    event.selection_reason = "补充选取，确保多样性"
                    break
        
        selected.sort(key=lambda x: x.peak_flow, reverse=True)
        
        return selected[:n_select]


def analyze_flood_data(
    dates: pd.Series,
    precip: np.ndarray,
    flow: np.ndarray,
    area: float = 150.7944,
    timestep: str = 'daily',
    n_select: int = 5
) -> PreAnalysisResult:
    """
    便捷函数：执行完整的数据预分析
    
    Args:
        dates: 日期序列
        precip: 降水序列 (mm)
        flow: 流量序列 (m³/s)
        area: 流域面积 (km²)
        timestep: 时间尺度 ('daily' 或 'hourly')
        n_select: 选取代表性洪水数量
        
    Returns:
        PreAnalysisResult: 预分析结果
    """
    analyzer = DataPreAnalyzer(area=area)
    return analyzer.analyze(dates, precip, flow, timestep, n_select)

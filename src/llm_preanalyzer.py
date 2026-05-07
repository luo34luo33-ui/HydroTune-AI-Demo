# -*- coding: utf-8 -*-
"""
LLM智能预分析模块
HydroTune-AI - 智能水文模型率定系统

功能：
1. 基于语义编码进行洪水场次分析
2. 调用LLM生成图文并茂的数据分析报告
3. 生成Base64图片嵌入报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import warnings
import numpy as np
from scipy import stats

from src.flood_semantic_encoder import FloodSemanticCode, FloodSemanticEncoder
from src.llm_api import call_minimax


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float = 100.0
    continuity: float = 100.0
    correlation: float = 0.0
    quality_level: str = "良好"
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class SemanticPreAnalyzer:
    """基于语义编码的LLM智能预分析器"""
    
    def __init__(self, area: float = 150.7944, timestep: str = 'daily'):
        self.area = area
        self.timestep = timestep
        self.encoder = FloodSemanticEncoder(area=area, timestep=timestep)
    
    def parse_dates(self, dates_input) -> pd.Series:
        """智能解析时间列
        
        支持：
        - pd.Series (datetime, string, numeric)
        - list of strings
        - numpy array of strings
        - DataFrame (自动查找date/Date/日期列)
        """
        if dates_input is None:
            return pd.Series(pd.date_range(start='2020-01-01', periods=100, freq='D'))
        
        n = 100
        
        if hasattr(dates_input, '__len__'):
            n = len(dates_input)
        
        if n == 0:
            return pd.Series(pd.date_range(start='2020-01-01', periods=100, freq='D'))
        
        # 如果是DataFrame，自动查找日期列
        if hasattr(dates_input, 'columns'):
            df = dates_input
            date_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if col_lower in ['date', 'datetime', 'time', '日期', '时间']:
                    date_col = col
                    break
            
            if date_col:
                dates_input = df[date_col]
            elif 'date' in df.columns:
                dates_input = df['date']
        
        # 已经 是datetime64
        if isinstance(dates_input, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(dates_input):
                return dates_input.reset_index(drop=True)
        
        # 尝试多种格式解析
        common_formats = [
            '%Y/%m/%d %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y%m%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y/%d/%m %H:%M',
        ]
        
        # 方法1: 直接to_datetime
        try:
            parsed = pd.to_datetime(dates_input, errors='coerce')
            valid_ratio = parsed.notna().sum() / n if n > 0 else 0
            if valid_ratio > 0.5:
                return parsed.reset_index(drop=True)
        except:
            pass
        
        # 方法2: 尝试常见格式
        if isinstance(dates_input, (list, np.ndarray, pd.Series)):
            for fmt in common_formats:
                try:
                    if isinstance(dates_input, pd.Series):
                        parsed = dates_input.apply(lambda x: pd.to_datetime(x, format=fmt, errors='coerce'))
                    else:
                        parsed = pd.to_datetime(dates_input, format=fmt, errors='coerce')
                    
                    valid_ratio = parsed.notna().sum() / n if n > 0 else 0
                    if valid_ratio > 0.5:
                        return parsed.reset_index(drop=True)
                except:
                    continue
        
        # 方法3: 解析字符串中的数字（Unix时间戳）
        try:
            if isinstance(dates_input, (list, np.ndarray)):
                numeric_dates = []
                for x in dates_input:
                    try:
                        numeric_dates.append(pd.to_datetime(float(x), unit='s'))
                    except:
                        try:
                            numeric_dates.append(pd.to_datetime(x))
                        except:
                            numeric_dates.append(pd.NaT)
                
                parsed = pd.Series(numeric_dates)
                valid_ratio = parsed.notna().sum() / n if n > 0 else 0
                if valid_ratio > 0.5:
                    return parsed.reset_index(drop=True)
        except:
            pass
        
        # 方法4: infer_datetime_format
        try:
            parsed = pd.to_datetime(dates_input, infer_datetime_format=True, errors='coerce')
            valid_ratio = parsed.notna().sum() / n if n > 0 else 0
            if valid_ratio > 0.5:
                return parsed.reset_index(drop=True)
        except:
            pass
        
        # 无法解析，返回默认日期序列
        return pd.Series(pd.date_range(start='2020-01-01', periods=n, freq='D'))
    
    def infer_timestep_from_dates(self, dates: pd.Series) -> str:
        """从日期序列推断时间尺度"""
        if len(dates) < 2:
            return 'daily'
        
        try:
            diffs = dates.diff().dropna()
            if len(diffs) == 0:
                return 'daily'
            
            median_diff = diffs.median()
            hours = median_diff.total_seconds() / 3600 if hasattr(median_diff, 'total_seconds') else 24
            
            if hours <= 1:
                return 'hourly'
            elif hours <= 24:
                return 'daily'
            else:
                return 'multi-daily'
        except:
            return 'daily'
    
    def evaluate_quality(
        self,
        precip: np.ndarray,
        flow: np.ndarray,
        dates: pd.Series
    ) -> DataQualityMetrics:
        """评估数据质量"""
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
            corr_matrix = np.corrcoef(precip[valid_mask], flow[valid_mask])
            if corr_matrix.shape == (2, 2):
                correlation = corr_matrix[0, 1]
                correlation = max(0, correlation) if not np.isnan(correlation) else 0.0
        
        precip_mean = np.nanmean(precip)
        precip_std = np.nanstd(precip)
        outlier_count = np.sum(np.abs(precip - precip_mean) > 3 * precip_std) if precip_std > 0 else 0
        precip_outlier_ratio = outlier_count / n * 100
        
        flow_mean = np.nanmean(flow)
        flow_std = np.nanstd(flow)
        outlier_count = np.sum(np.abs(flow - flow_mean) > 3 * flow_std) if flow_std > 0 else 0
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
        
        return DataQualityMetrics(
            completeness=completeness,
            continuity=continuity,
            correlation=correlation,
            quality_level=quality_level,
            issues=issues
        )
    
    def detect_flood_events(
        self,
        flow: np.ndarray,
        dates: pd.Series,
        precip: np.ndarray = None,
        precip_threshold: float = 1.0,
        flow_threshold: float = None
    ) -> List[Dict]:
        """识别洪水场次（基于基流分离+斜率变化）"""
        dates = dates.reset_index(drop=True) if hasattr(dates, 'reset_index') else pd.Series(dates).reset_index(drop=True)
        
        n = len(flow)
        if n < 10:
            return []
        
        if len(dates) != n:
            dates = pd.Series(pd.date_range(start='2020-01-01', periods=n, freq='D'))
        
        inferred_timestep = self.infer_timestep_from_dates(dates)
        if inferred_timestep != self.timestep:
            self.timestep = inferred_timestep
        
        baseflow = self._estimate_baseflow(flow)
        
        flow_anomaly = flow - baseflow
        
        mean_anomaly = np.mean(flow_anomaly[flow_anomaly > 0]) if np.any(flow_anomaly > 0) else 0
        std_anomaly = np.std(flow_anomaly[flow_anomaly > 0]) if np.any(flow_anomaly > 0) else 1
        
        peak_threshold = mean_anomaly + 0.2 * std_anomaly
        
        is_flood = flow_anomaly > peak_threshold
        
        diff = np.diff(is_flood.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            if flow_threshold is None:
                flow_threshold = np.percentile(flow, 70)
            is_flood_simple = flow > flow_threshold
            diff = np.diff(is_flood_simple.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            return []
        
        if starts[0] > ends[0]:
            starts = np.concatenate([[0], starts])
        if starts[-1] > ends[-1]:
            ends = np.concatenate([ends, [n - 1]])
        
        min_duration = 3 if self.timestep == 'hourly' else 5
        events = []
        
        for start, end in zip(starts, ends):
            duration = end - start + 1
            if duration < min_duration:
                continue
            
            event_flow = flow[start:end + 1]
            peak_idx = int(np.argmax(event_flow))
            peak_flow = float(event_flow[peak_idx])
            
            if peak_flow < np.mean(baseflow) * 1.5:
                continue
            
            events.append({
                'start_idx': start,
                'end_idx': end,
                'peak_idx': start + peak_idx,
                'peak_flow': peak_flow
            })
        
        return events
    
    def _estimate_baseflow(self, flow: np.ndarray, window: int = 5) -> np.ndarray:
        """估计基流"""
        if len(flow) < window:
            return np.full_like(flow, np.mean(flow))
        
        df = pd.Series(flow)
        baseflow = df.rolling(window=window, center=True, min_periods=1).min()
        baseflow = baseflow.ffill().bfill().values
        
        return baseflow
    
    def generate_semantic_codes(
        self,
        flow_data: np.ndarray,
        precip_data: np.ndarray,
        dates: pd.Series,
        event_boundaries: List[Dict] = None,
        flow_threshold: float = None
    ) -> List[FloodSemanticCode]:
        """生成语义编码"""
        if event_boundaries is None:
            event_boundaries = self.detect_flood_events(flow_data, dates, precip_data)
        
        if not event_boundaries:
            return []
        
        peaks = np.array([e['peak_flow'] for e in event_boundaries])
        
        return self.encoder.encode_multiple_events(
            flow_data, precip_data, dates, event_boundaries, peaks
        )
    
    def generate_base64_plot(
        self,
        flow_data: np.ndarray,
        precip_data: np.ndarray,
        dates: pd.Series,
        semantic_codes: List[FloodSemanticCode],
        quality: DataQualityMetrics,
        multi_file_mode: bool = False
    ) -> str:
        """生成分析图片并转换为Base64
        
        Args:
            multi_file_mode: 是否为多文件多场洪水模式
        """
        try:
            n_events = len(semantic_codes)
            is_multi_flood = n_events > 1 or (n_events == 1 and len(flow_data) > 500)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('洪水场次数据预分析', fontsize=16, fontweight='bold')
            
            if is_multi_flood or multi_file_mode:
                peaks = np.array([c.peak_flow for c in semantic_codes])
                volumes = np.array([c.flood_volume for c in semantic_codes])
                
                ax1 = axes[0, 0]
                if len(peaks) >= 3:
                    self._plot_pearson3_curve(ax1, peaks, '洪峰流量', 'm³/s')
                else:
                    ax1.text(0.5, 0.5, '样本不足\n无法进行频率分析', ha='center', va='center', fontsize=14)
                    ax1.set_title('洪峰频率曲线（P-III型）', fontsize=12)
                
                ax2 = axes[0, 1]
                if len(volumes) >= 3:
                    self._plot_pearson3_curve(ax2, volumes, '洪水总量', '10⁴m³')
                else:
                    ax2.text(0.5, 0.5, '样本不足\n无法进行频率分析', ha='center', va='center', fontsize=14)
                    ax2.set_title('洪水总量频率曲线（P-III型）', fontsize=12)
            else:
                xlabel = '时间(天)' if self.timestep == 'daily' else '时间(h)'
                
                ax1 = axes[0, 0]
                time_axis = range(len(flow_data))
                ax1.plot(time_axis, flow_data, 'b-', linewidth=1.2, label='流量')
                ax1.fill_between(time_axis, 0, flow_data, alpha=0.3, color='steelblue')
                
                ax1.set_xlabel(xlabel, fontsize=11)
                ax1.set_ylabel(r'流量 ($m^3/s$)', fontsize=11)
                ax1.set_title('洪水过程线', fontsize=12)
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                
                ax2 = axes[0, 1]
                peak_idx = int(np.argmax(flow_data))
                peak_time = peak_idx if self.timestep == 'daily' else peak_idx
                ax2.bar([0], [flow_data[peak_idx]], color='steelblue', alpha=0.8)
                ax2.set_ylabel(r'峰值流量 ($m^3/s$)', fontsize=11)
                ax2.set_title(f'洪峰: {flow_data[peak_idx]:.1f} m³/s', fontsize=12)
                ax2.set_xticks([])
                ax2.grid(True, alpha=0.3, axis='y')
            
            ax3 = axes[1, 0]
            if semantic_codes and len(semantic_codes) > 1:
                volumes = [c.flood_volume for c in semantic_codes]
                precips = [c.total_precip for c in semantic_codes]
                ax3.scatter(precips, volumes, c='steelblue', s=80, alpha=0.7, edgecolors='navy')
                
                if len(precips) > 1:
                    z = np.polyfit(precips, volumes, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(precips), max(precips), 100)
                    ax3.plot(x_line, p(x_line), 'r--', linewidth=1.5, alpha=0.7, label='趋势线')
                
                ax3.set_xlabel('总降水量 (mm)', fontsize=11)
                ax3.set_ylabel(r'洪水总量 ($10^4 m^3$)', fontsize=11)
                ax3.set_title('降水-径流关系', fontsize=12)
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
            elif semantic_codes:
                ax3.text(0.5, 0.5, '单场洪水\n无对比数据', ha='center', va='center', fontsize=14)
                ax3.set_title('降水-径流关系', fontsize=12)
            
            ax4 = axes[1, 1]
            metrics = ['完整率', '连续性', '相关性', '质量等级']
            values = [
                quality.completeness,
                quality.continuity,
                quality.correlation * 100,
                {'优秀': 100, '良好': 80, '一般': 60, '较差': 40}.get(quality.quality_level, 60)
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values_plot = values + [values[0]]
            angles += angles[:1]
            
            ax4.remove()
            ax4 = fig.add_subplot(2, 2, 4, projection='polar')
            ax4.plot(angles, values_plot, 'o-', linewidth=2, color='steelblue')
            ax4.fill(angles, values_plot, alpha=0.25, color='steelblue')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics, fontsize=10)
            ax4.set_ylim(0, 100)
            ax4.set_title('数据质量雷达图', fontsize=12, pad=20)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            warnings.warn(f"生成图片失败: {e}")
            return ""
    
    def _plot_pearson3_curve(self, ax, data: np.ndarray, label: str, unit: str):
        """
        绘制皮尔逊III型（P-III）频率曲线
        符合水文惯例：采用正态概率坐标横轴、经验点据降序排列、显示统计参数
        """
        n = len(data)
        if n < 3:
            ax.text(0.5, 0.5, f'样本不足(n={n})', ha='center', va='center', fontsize=14)
            ax.set_title(f'{label}频率曲线（P-III型）', fontsize=12)
            return
        
        # 1. 数据排序与经验频率计算（水文惯例：针对洪水等极大值，采用降序）
        # 左低频（大洪水），右高频（枯水）
        sorted_data = np.sort(data)[::-1] 
        m = np.arange(1, n + 1)
        empirical_freq = m / (n + 1)  # 韦伯(Weibull)经验频率公式
        
        # 2. 计算统计参数
        mean_val = np.mean(sorted_data)
        std_val = np.std(sorted_data, ddof=1)
        cv = std_val / mean_val if mean_val > 0 else 0
        
        # 计算偏态系数 Cs (采用无偏估计或直接计算，此处为样本标准矩计算)
        if std_val > 0:
            cs = np.sum((sorted_data - mean_val) ** 3) / (n * std_val ** 3)
        else:
            cs = 0
            
        # 3. 概率坐标系转换函数（模拟海森频率纸）
        def to_prob_scale(p):
            """将频率转换为标准正态分位数，以拉伸X轴"""
            p = np.clip(p, 1e-5, 1 - 1e-5)
            return stats.norm.ppf(p)
            
        # 4. 绘制经验点据
        x_emp = to_prob_scale(empirical_freq)
        # 水文经验点常用空心圆，避免遮挡曲线
        ax.scatter(x_emp, sorted_data, facecolors='none', edgecolors='steelblue', 
                s=50, linewidths=1.5, zorder=3, label='经验频率点')
        
        # 5. 计算并绘制 P-III 理论曲线
        # 使用极密集的频率点序列，保证曲线在概率纸上极度平滑
        p_dense = np.concatenate([
            np.linspace(0.0001, 0.01, 50),
            np.linspace(0.01, 0.99, 100),
            np.linspace(0.99, 0.9999, 50)
        ])
        
        # 利用 scipy.stats.pearson3 的逆生存函数(isf)直接计算给定超越概率的设计值
        theoretical_freqs = stats.pearson3.isf(p_dense, skew=cs, loc=mean_val, scale=std_val)
        x_theo = to_prob_scale(p_dense)
        ax.plot(x_theo, theoretical_freqs, 'r-', linewidth=1.8, zorder=2, label='P-III理论曲线')
        
        # 6. 设置水文特色横坐标轴刻度 (典型的频率纸刻度)
        xtick_probs = np.array([0.01, 0.1, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.9]) / 100
        xtick_locs = to_prob_scale(xtick_probs)
        xtick_labels = ['0.01', '0.1', '1', '5', '10', '20', '50', '80', '90', '95', '99', '99.9']
        
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xlim(to_prob_scale(0.0001), to_prob_scale(0.999))
        
        # 7. 绘制网格线
        ax.grid(True, which='major', axis='x', color='gray', linestyle='--', alpha=0.4)
        ax.grid(True, which='major', axis='y', color='gray', linestyle='-', alpha=0.2)
        
        # 8. 图面信息补充：在图上标注三大统计参数 (水文规范核心要求)
        param_text = (
            f"$\\bar{{X}}$ = {mean_val:.2f}\n"
            f"$C_v$ = {cv:.3f}\n"
            f"$C_s$ = {cs:.3f}"
        )
        # 将参数框放置在右上角
        ax.text(0.95, 0.95, param_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray'),
                zorder=4)
        
        # 9. 标题与图例
        ax.set_xlabel('频率 P (%)', fontsize=11)
        ax.set_ylabel(f'{label} ({unit})', fontsize=11)
        ax.set_title(f'{label}理论与经验频率曲线 (P-III)', fontsize=12, pad=10)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

    def generate_analysis_prompt(
        self,
        semantic_codes: List[FloodSemanticCode],
        quality: DataQualityMetrics
    ) -> str:
        """生成LLM分析提示词"""
        codes_json = self.encoder.to_json_for_llm(semantic_codes)
        
        prompt = f"""你是一位资深水文数据分析师。请对以下洪水场次数据进行预分析评价，生成一份专业的数据预分析报告。

## 数据质量指标
- 数据完整率: {quality.completeness:.1f}%
- 时间连续性: {quality.continuity:.1f}%
- 降水-径流相关性: {quality.correlation:.3f}
- 质量等级: {quality.quality_level}

## 洪水场次语义编码
{codes_json}

## 分析要求
请严格按照以下结构进行分析，强制使用Markdown标题分层呈现：

### 一、洪水特征总结
请按"量级 → 峰型 → 历时 → 涨落特性 → 降水驱动"顺序对各场次进行分类汇总，并标出每类的典型特征范围。

### 二、异常数据识别
请明确标注以下异常：
- 负值或异常洪水体积
- 极端峰值（超过均值+2倍标准差）
- 异常历时（过短<3天或过长>30天）
- 离群洪量

### 三、时空分布规律
分析洪水事件的时间分布特征（季节性、集中程度等）。

### 四、数据代表性评估
结合representative_score，分析代表性洪水的覆盖程度。

### 五、数据质量评价
对数据完整性、连续性、相关性进行专业评价，如有数据问题需明确指出。

### 六、模型率定策略建议
请分量级提出：
- 小型(small)洪水：建议率定策略
- 中型(medium)洪水：建议率定策略  
- 大型(large)洪水：建议率定策略
- 极端(extreme)洪水：建议率定策略

### 七、异常数据处理建议
对识别出的异常数据提出具体处理意见。

---
在文本中同时引用数值特征（峰值流量m³/s、洪水总量10⁴m³、历时天）与语义标签（rapid/slow涨落、sharp/rounded峰型、high/medium/low降水驱动），增强专业性。

请用Markdown格式回复，保持专业简洁，覆盖所有洪水量级类别。"""
        
        return prompt
    
    def analyze(
        self,
        flow_data: np.ndarray,
        precip_data: np.ndarray,
        dates: pd.Series,
        call_llm=call_minimax,
        multi_file_mode: bool = False
    ) -> Dict[str, Any]:
        """执行完整的LLM智能分析
        
        Args:
            multi_file_mode: 是否为多文件多场洪水模式
        
        Returns:
            Dict: 包含以下键
                - semantic_codes: 语义编码列表
                - quality: 数据质量指标
                - llm_report: LLM生成的文字报告
                - base64_plot: Base64编码的分析图片
                - summary: 统计摘要
        """
        quality = self.evaluate_quality(precip_data, flow_data, dates)
        
        event_boundaries = self.detect_flood_events(flow_data, dates, precip_data)
        
        if not event_boundaries:
            event_boundaries = [{
                'start_idx': 0,
                'end_idx': len(flow_data) - 1,
                'peak_idx': int(np.argmax(flow_data)),
                'peak_flow': float(np.max(flow_data))
            }]
        
        semantic_codes = self.generate_semantic_codes(
            flow_data, precip_data, dates, event_boundaries
        )
        
        base64_plot = ""
        llm_report = ""
        
        if semantic_codes and call_llm:
            base64_plot = self.generate_base64_plot(
                flow_data, precip_data, dates, semantic_codes, quality,
                multi_file_mode=multi_file_mode
            )
            
            prompt = self.generate_analysis_prompt(semantic_codes, quality)
            
            system_prompt = """你是HydroTune-AI水文数据智能分析师。你擅长分析洪水数据、提取水文特征、生成专业的数据解读报告。"""
            
            llm_result = call_llm(prompt, system_prompt)
            
            if llm_result and not llm_result.startswith("[ERROR]"):
                llm_report = llm_result.replace('***', '').replace('---', '').strip()
            else:
                llm_report = f"**LLM分析暂时不可用**\n\n基础分析结果：\n\n" + self._generate_fallback_report(semantic_codes, quality)
        else:
            llm_report = self._generate_fallback_report(semantic_codes, quality)
        
        summary = {
            'n_events': len(semantic_codes),
            'mean_peak': np.mean([c.peak_flow for c in semantic_codes]) if semantic_codes else 0,
            'max_peak': np.max([c.peak_flow for c in semantic_codes]) if semantic_codes else 0,
            'total_volume': np.sum([c.flood_volume for c in semantic_codes]) if semantic_codes else 0,
            'quality_level': quality.quality_level
        }
        
        return {
            'semantic_codes': semantic_codes,
            'quality': quality,
            'llm_report': llm_report,
            'base64_plot': base64_plot,
            'summary': summary
        }
    
    def _generate_fallback_report(
        self,
        semantic_codes: List[FloodSemanticCode],
        quality: DataQualityMetrics
    ) -> str:
        """生成回退报告（当LLM不可用时）"""
        lines = [
            f"## 数据质量评价",
            f"- 质量等级: **{quality.quality_level}**",
            f"- 完整率: {quality.completeness:.1f}%",
            f"- 连续性: {quality.continuity:.1f}%",
            f"- 相关性: {quality.correlation:.3f}",
            ""
        ]
        
        if quality.issues:
            lines.append("### 数据问题")
            for issue in quality.issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        if semantic_codes:
            lines.append("## 洪水场次特征")
            lines.append("")
            lines.append("| 场次 | 量级 | 峰值(m³/s) | 洪量(10⁴m³) | 历时 | 峰型 |")
            lines.append("|------|------|-----------|------------|-----|------|")
            
            for code in semantic_codes:
                lines.append(
                    f"| {code.event_id} | {code.magnitude} | {code.peak_flow:.1f} | "
                    f"{code.flood_volume:.1f} | {code.duration} | {code.peak_shape} |"
                )
            lines.append("")
            
            lines.append("### 统计摘要")
            peaks = [c.peak_flow for c in semantic_codes]
            lines.append(f"- 识别场次: {len(semantic_codes)} 场")
            lines.append(f"- 平均峰值: {np.mean(peaks):.2f} m³/s")
            lines.append(f"- 最大峰值: {np.max(peaks):.2f} m³/s")
        
        return "\n".join(lines)


def analyze_flood_data(
    flow_data: np.ndarray,
    precip_data: np.ndarray,
    dates: pd.Series,
    area: float = 150.7944,
    timestep: str = 'daily',
    call_llm=call_minimax
) -> Dict[str, Any]:
    """便捷函数：执行LLM智能预分析
    
    Args:
        flow_data: 流量序列
        precip_data: 降水序列
        dates: 日期序列
        area: 流域面积
        timestep: 时间尺度
        call_llm: LLM调用函数
        
    Returns:
        分析结果字典
    """
    analyzer = SemanticPreAnalyzer(area=area, timestep=timestep)
    return analyzer.analyze(flow_data, precip_data, dates, call_llm)
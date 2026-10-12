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
from src.llm_api import call_minimax, get_language


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float = 100.0
    continuity: float = 100.0
    correlation: float = 0.0
    quality_level: str = "Good"
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
            quality_level = "Excellent"
        elif completeness >= 90 and continuity >= 90 and correlation >= 0.3 and outlier_ratio <= 10:
            quality_level = "Good"
        elif completeness >= 80:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
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
        """Generate an English, publication-style analysis plot and convert to Base64."""
        try:
            n_events = len(semantic_codes)
            is_multi_flood = n_events > 1 or (n_events == 1 and len(flow_data) > 500)

            style = {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "axes.linewidth": 0.8,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.unicode_minus": False,
            }

            with plt.rc_context(style):
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
                fig.suptitle("Flood Event Data Pre-Analysis", fontsize=13, fontweight="bold")

                if is_multi_flood or multi_file_mode:
                    peaks = np.array([c.peak_flow for c in semantic_codes])
                    volumes = np.array([c.flood_volume for c in semantic_codes])

                    ax1 = axes[0, 0]
                    if len(peaks) >= 3:
                        self._plot_pearson3_curve(ax1, peaks, "Peak Flow", "m^3/s")
                    else:
                        ax1.text(
                            0.5, 0.5,
                            "Insufficient samples\nfor frequency analysis",
                            ha="center", va="center", fontsize=11
                        )
                        ax1.set_title("Peak Flow Frequency Curve (P-III)")
                        self._style_axis(ax1)

                    ax2 = axes[0, 1]
                    if len(volumes) >= 3:
                        self._plot_pearson3_curve(ax2, volumes, "Flood Volume", "10^4 m^3")
                    else:
                        ax2.text(
                            0.5, 0.5,
                            "Insufficient samples\nfor frequency analysis",
                            ha="center", va="center", fontsize=11
                        )
                        ax2.set_title("Flood Volume Frequency Curve (P-III)")
                        self._style_axis(ax2)
                else:
                    xlabel = "Time (days)" if self.timestep == "daily" else "Time (h)"

                    ax1 = axes[0, 0]
                    time_axis = np.arange(len(flow_data))
                    ax1.plot(time_axis, flow_data, color="#1f4e79", linewidth=1.3, label="Observed flow")
                    ax1.fill_between(time_axis, 0, flow_data, color="#1f4e79", alpha=0.12)
                    ax1.set_xlabel(xlabel)
                    ax1.set_ylabel(r"Flow ($m^3/s$)")
                    ax1.set_title("Flood Hydrograph")
                    ax1.legend(loc="upper right", frameon=False)
                    self._style_axis(ax1)

                    ax2 = axes[0, 1]
                    peak_idx = int(np.argmax(flow_data))
                    ax2.bar([0], [flow_data[peak_idx]], color="#1f4e79", alpha=0.85, width=0.45)
                    ax2.set_ylabel(r"Peak Flow ($m^3/s$)")
                    ax2.set_title(f"Peak Flow = {flow_data[peak_idx]:.1f} m^3/s")
                    ax2.set_xticks([])
                    self._style_axis(ax2)

                ax3 = axes[1, 0]
                if semantic_codes and len(semantic_codes) > 1:
                    volumes = [c.flood_volume for c in semantic_codes]
                    precips = [c.total_precip for c in semantic_codes]
                    ax3.scatter(
                        precips, volumes, marker="o", s=42,
                        facecolors="white", edgecolors="#1f4e79",
                        linewidths=1.1, label="Flood events"
                    )

                    if len(precips) > 1 and min(precips) != max(precips):
                        z = np.polyfit(precips, volumes, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(min(precips), max(precips), 100)
                        ax3.plot(
                            x_line, p(x_line), color="#b22222",
                            linestyle="--", linewidth=1.2, label="Linear fit"
                        )

                    ax3.set_xlabel("Total Precipitation (mm)")
                    ax3.set_ylabel(r"Flood Volume ($10^4$ m$^3$)")
                    ax3.set_title("Precipitation-Runoff Relationship")
                    ax3.legend(loc="best", frameon=False)
                    self._style_axis(ax3)
                elif semantic_codes:
                    ax3.text(
                        0.5, 0.5, "Single event\nNo comparison data",
                        ha="center", va="center", fontsize=11
                    )
                    ax3.set_title("Precipitation-Runoff Relationship")
                    self._style_axis(ax3)

                ax4 = axes[1, 1]
                metrics = ["Completeness", "Continuity", "Correlation", "Quality"]
                quality_score = self._quality_score_for_plot(quality)
                values = [
                    quality.completeness,
                    quality.continuity,
                    quality.correlation * 100,
                    quality_score,
                ]

                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                values_plot = values + [values[0]]
                angles += angles[:1]

                ax4.remove()
                ax4 = fig.add_subplot(2, 2, 4, projection="polar")
                ax4.plot(angles, values_plot, marker="o", linewidth=1.4, color="#1f4e79")
                ax4.fill(angles, values_plot, alpha=0.14, color="#1f4e79")
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(metrics)
                ax4.set_ylim(0, 100)
                ax4.set_yticks([25, 50, 75, 100])
                ax4.set_yticklabels(["25", "50", "75", "100"])
                ax4.grid(color="#d0d0d0", linewidth=0.6)
                ax4.spines["polar"].set_color("#4a4a4a")
                ax4.spines["polar"].set_linewidth(0.8)
                ax4.set_title("Data Quality Radar Chart", pad=18)

                buffer = BytesIO()
                fig.savefig(
                    buffer, format="png", dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none"
                )
                buffer.seek(0)

                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plt.close(fig)

                return img_base64

        except Exception as e:
            warnings.warn(f"Failed to generate plot: {e}")
            return ""

    @staticmethod
    def _style_axis(ax) -> None:
        """Apply restrained, paper-style axis formatting."""
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4a4a4a")
        ax.spines["bottom"].set_color("#4a4a4a")
        ax.tick_params(direction="out", length=3, width=0.8, colors="#333333")

    @staticmethod
    def _quality_score_for_plot(quality: DataQualityMetrics) -> float:
        """Convert quality indicators into a 0-100 plot score without localized labels."""
        if quality.completeness >= 95 and quality.continuity >= 95 and quality.correlation >= 0.5:
            return 100.0
        if quality.completeness >= 90 and quality.continuity >= 90 and quality.correlation >= 0.3:
            return 80.0
        if quality.completeness >= 80:
            return 60.0
        return 40.0

    def _plot_pearson3_curve(self, ax, data: np.ndarray, label: str, unit: str):
        """Plot an English Pearson Type III (P-III) frequency curve."""
        n = len(data)
        if n < 3:
            ax.text(0.5, 0.5, f"Insufficient samples (n={n})", ha="center", va="center", fontsize=11)
            ax.set_title(f"{label} Frequency Curve (P-III)")
            self._style_axis(ax)
            return

        sorted_data = np.sort(data)[::-1]
        m = np.arange(1, n + 1)
        empirical_freq = m / (n + 1)

        mean_val = np.mean(sorted_data)
        std_val = np.std(sorted_data, ddof=1)
        cv = std_val / mean_val if mean_val > 0 else 0
        cs = np.sum((sorted_data - mean_val) ** 3) / (n * std_val ** 3) if std_val > 0 else 0

        def to_prob_scale(p):
            p = np.clip(p, 1e-5, 1 - 1e-5)
            return stats.norm.ppf(p)

        x_emp = to_prob_scale(empirical_freq)
        ax.scatter(
            x_emp, sorted_data, facecolors="white", edgecolors="#1f4e79",
            s=42, linewidths=1.1, zorder=3, label="Empirical points"
        )

        p_dense = np.concatenate([
            np.linspace(0.0001, 0.01, 50),
            np.linspace(0.01, 0.99, 100),
            np.linspace(0.99, 0.9999, 50),
        ])
        theoretical_freqs = stats.pearson3.isf(p_dense, skew=cs, loc=mean_val, scale=std_val)
        x_theo = to_prob_scale(p_dense)
        ax.plot(x_theo, theoretical_freqs, color="#b22222", linewidth=1.3, zorder=2, label="P-III fitted curve")

        xtick_probs = np.array([0.01, 0.1, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.9]) / 100
        xtick_locs = to_prob_scale(xtick_probs)
        xtick_labels = ["0.01", "0.1", "1", "5", "10", "20", "50", "80", "90", "95", "99", "99.9"]

        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(to_prob_scale(0.0001), to_prob_scale(0.999))
        ax.grid(True, which="major", axis="x", color="#d0d0d0", linestyle="--", linewidth=0.6)
        ax.grid(True, which="major", axis="y", color="#e0e0e0", linestyle="-", linewidth=0.5)

        param_text = (
            f"$\\bar{{X}}$ = {mean_val:.2f}\n"
            f"$C_v$ = {cv:.3f}\n"
            f"$C_s$ = {cs:.3f}"
        )
        ax.text(
            0.95, 0.95, param_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, edgecolor="#bdbdbd", linewidth=0.6),
            zorder=4,
        )

        ax.set_xlabel("Exceedance Probability P (%)")
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(f"{label} Frequency Curve (P-III)", pad=8)
        ax.legend(loc="lower left", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4a4a4a")
        ax.spines["bottom"].set_color("#4a4a4a")
        ax.tick_params(direction="out", length=3, width=0.8, colors="#333333")

    def generate_analysis_prompt(
        self,
        semantic_codes: List[FloodSemanticCode],
        quality: DataQualityMetrics
    ) -> str:
        """Generate LLM analysis prompt"""
        lang = get_language()
        codes_json = self.encoder.to_json_for_llm(semantic_codes)
        
        if lang == 'en':
            prompt = f"""You are a senior hydrological data analyst. Please analyze the following flood event data and generate a professional pre-analysis report.

## Data Quality Metrics
- Data Completeness: {quality.completeness:.1f}%
- Time Continuity: {quality.continuity:.1f}%
- Precipitation-Runoff Correlation: {quality.correlation:.3f}
- Quality Level: {quality.quality_level}

## Flood Event Semantic Encoding
{codes_json}

## Analysis Requirements
Please strictly follow the structure below, using Markdown headings:

### 1. Flood Characteristics Summary
Summarize events by "magnitude → peak shape → duration → rise/recession characteristics → rainfall driving", noting typical feature ranges for each category.

### 2. Anomaly Detection
Clearly identify the following anomalies:
- Negative or abnormal flood volumes
- Extreme peaks (exceeding mean + 2 standard deviations)
- Abnormal duration (too short <3 days or too long >30 days)
- Outlier flood volumes

### 3. Spatiotemporal Distribution
Analyze the temporal distribution characteristics of flood events (seasonality, concentration, etc.).

### 4. Data Representativeness Assessment
Evaluate the coverage of representative floods based on representative_score.

### 5. Data Quality Evaluation
Professionally evaluate data completeness, continuity, and correlation. Clearly point out any data issues.

### 6. Model Calibration Strategy Recommendations
Suggest strategies by magnitude:
- Small floods: recommended calibration strategy
- Medium floods: recommended calibration strategy
- Large floods: recommended calibration strategy
- Extreme floods: recommended calibration strategy

### 7. Anomaly Data Handling Recommendations
Provide specific suggestions for handling identified anomalies.

---
Reference both numerical features (peak flow m³/s, flood volume 10⁴m³, duration days) and semantic labels (rapid/slow rise, sharp/rounded peak shape, high/medium/low rainfall driving) in the text to enhance professionalism.

Please respond in Markdown format, keeping it professional and concise, covering all flood magnitude categories."""
        else:
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
        multi_file_mode: bool = False,
        event_boundaries: List[Dict] = None
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
        
        if event_boundaries is None:
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
            
            lang = get_language()
            if lang == 'en':
                system_prompt = """You are HydroTune-AI hydrological data intelligence analyst. You are skilled at analyzing flood data, extracting hydrological features, and generating professional data interpretation reports."""
            else:
                system_prompt = """你是HydroTune-AI水文数据智能分析师。你擅长分析洪水数据、提取水文特征、生成专业的数据解读报告。"""
            
            llm_result = call_llm(prompt, system_prompt)
            
            if llm_result and not llm_result.startswith("[ERROR]"):
                llm_report = llm_result.replace('***', '').replace('---', '').strip()
            else:
                if lang == 'en':
                    llm_report = f"**LLM analysis temporarily unavailable**\n\nBasic analysis results:\n\n" + self._generate_fallback_report(semantic_codes, quality)
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
            'event_boundaries': event_boundaries,
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

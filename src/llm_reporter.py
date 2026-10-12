# -*- coding: utf-8 -*-
"""
LLM 水文报告生成模块
HydroTune-AI - 智能水文模型率定系统

功能：
1. 生成数据预分析报告
2. 生成模型率定分析报告
3. 生成综合分析报告
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

from src.llm_api import call_minimax, get_language
from src.data_preanalysis import (
    PreAnalysisResult, DataQualityResult, FloodEvent, FrequencyAnalysisResult
)


def generate_preanalysis_report(
    result: PreAnalysisResult,
    call_llm=call_minimax
) -> str:
    """
    Generate data pre-analysis report
    """
    lang = get_language()
    quality = result.quality
    events = result.events
    freq = result.frequency
    selected = result.selected_events
    
    if lang == 'en':
        ts_label = 'hourly' if result.timestep == 'hourly' else 'daily'
        report_lines = [
            "# Data Pre-Analysis Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Catchment Area**: {result.area} km2",
            f"**Timestep**: {ts_label}",
            "",
            "---",
            "",
            "## 1. Data Quality Assessment",
            "",
        ]
        report_lines.append("| Metric | Value | Rating |")
        report_lines.append("|--------|-------|--------|")
        report_lines.append(f"| Data Completeness | {quality.completeness:.1f}% | {'Good' if quality.completeness >= 95 else 'Low'} |")
        report_lines.append(f"| Time Continuity | {quality.continuity:.1f}% | {'Good' if quality.continuity >= 95 else 'Low'} |")
        report_lines.append(f"| Precip-Runoff Correlation | {quality.correlation:.3f} | {'Good' if quality.correlation >= 0.5 else 'Low'} |")
        report_lines.append(f"| Outlier Ratio | {quality.outlier_ratio:.1f}% | {'Normal' if quality.outlier_ratio <= 5 else 'High'} |")
        report_lines.append(f"| **Quality Level** | **{quality.quality_level}** | - |")
        report_lines.append("")
        if quality.issues:
            report_lines.append("**Data Issues:**")
            for issue in quality.issues:
                report_lines.append(f"- {issue}")
            report_lines.append("")
        report_lines.extend([
            "## 2. Flood Event Statistics",
            "",
            f"**{len(events)}** flood event(s) identified",
            "",
            "| Event | Period | Peak Flow (m3/s) | Flood Volume (mm) | Duration (days) | Rise Rate | Recession Rate |",
            "|-------|--------|-----------------|-------------------|-----------------|-----------|----------------|",
        ])
        for e in events:
            start_str = e.start_date.strftime('%Y-%m-%d') if hasattr(e.start_date, 'strftime') else str(e.start_date)[:10]
            end_str = e.end_date.strftime('%Y-%m-%d') if hasattr(e.end_date, 'strftime') else str(e.end_date)[:10]
            report_lines.append(
                f"| {e.event_id} | {start_str} ~ {end_str} | {e.peak_flow:.1f} | "
                f"{e.flood_volume:.1f} | {e.duration} | {e.rise_rate:.3f} | {e.recession_rate:.3f} |"
            )
        report_lines.append("")
        report_lines.extend([
            "## 3. Frequency Analysis",
            "",
            "### 3.1 Statistical Parameters",
            "",
        ])
        report_lines.append(f"- Sample Size: {freq.n_samples} events")
        report_lines.append(f"- Mean Peak Flow: {freq.mean:.2f} m3/s")
        report_lines.append(f"- Standard Deviation: {freq.std:.2f} m3/s")
        report_lines.append(f"- Coefficient of Variation (Cv): {freq.cv:.3f}")
        report_lines.append(f"- Skewness (Cs): {freq.cs:.3f}")
        report_lines.append("")
        report_lines.append("### 3.2 Design Flood Results")
        report_lines.append("")
        report_lines.append("| Return Period | Design Flow (m3/s) |")
        report_lines.append("|---------------|-------------------|")
        for rp, val in freq.design_values.items():
            report_lines.append(f"| {rp} | {val:.2f} |")
        report_lines.append("")
        report_lines.extend([
            "## 4. Representative Flood Event Selection",
            "",
            f"**{len(selected)}** representative flood event(s) selected for model calibration:",
            "",
            "| Event | Peak Flow | Selection Reason |",
            "|-------|-----------|-----------------|",
        ])
        for e in selected:
            report_lines.append(f"| {e.event_id} | {e.peak_flow:.2f} m3/s | {e.selection_reason} |")
        report_lines.append("")
        
        if call_llm:
            llm_prompt = _build_preanalysis_llm_prompt(quality, events, freq, selected)
            report_lines.extend([
                "## 5. LLM Intelligent Analysis",
                "",
                "### 5.1 Data Quality Evaluation",
                "",
            ])
            stt = call_llm(llm_prompt)
            if not stt.startswith("[ERROR]"):
                report_lines.append(stt)
            else:
                report_lines.append("*LLM analysis temporarily unavailable*")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report auto-generated by HydroTune-AI*")
    else:
        report_lines = [
            "# 数据预分析报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**流域面积**: {result.area} km2",
            f"**时间尺度**: {'小时' if result.timestep == 'hourly' else '日'}尺度",
            "",
            "---",
            "",
            "## 1. 数据质量评估",
            "",
        ]
        report_lines.append("| 指标 | 数值 | 评价 |")
        report_lines.append("|------|------|------|")
        report_lines.append(f"| 数据完整率 | {quality.completeness:.1f}% | {'良好' if quality.completeness >= 95 else '偏低'} |")
        report_lines.append(f"| 时间连续性 | {quality.continuity:.1f}% | {'良好' if quality.continuity >= 95 else '偏低'} |")
        report_lines.append(f"| 降水-径流相关性 | {quality.correlation:.3f} | {'良好' if quality.correlation >= 0.5 else '偏低'} |")
        report_lines.append(f"| 极值比例 | {quality.outlier_ratio:.1f}% | {'正常' if quality.outlier_ratio <= 5 else '偏高'} |")
        report_lines.append(f"| **质量等级** | **{quality.quality_level}** | - |")
        report_lines.append("")
        if quality.issues:
            report_lines.append("**数据问题说明**：")
            for issue in quality.issues:
                report_lines.append(f"- {issue}")
            report_lines.append("")
        report_lines.extend([
            "## 2. 洪水事件统计",
            "",
            f"共识别到 **{len(events)}** 场洪水事件",
            "",
            "| 场次 | 起止日期 | 峰值流量(m3/s) | 洪水总量(mm) | 历时(天) | 涨率 | 落率 |",
            "|------|----------|---------------|-------------|---------|------|------|",
        ])
        for e in events:
            start_str = e.start_date.strftime('%Y-%m-%d') if hasattr(e.start_date, 'strftime') else str(e.start_date)[:10]
            end_str = e.end_date.strftime('%Y-%m-%d') if hasattr(e.end_date, 'strftime') else str(e.end_date)[:10]
            report_lines.append(
                f"| {e.event_id} | {start_str} ~ {end_str} | {e.peak_flow:.1f} | "
                f"{e.flood_volume:.1f} | {e.duration} | {e.rise_rate:.3f} | {e.recession_rate:.3f} |"
            )
        report_lines.append("")
        report_lines.extend([
            "## 3. 频率分析结果",
            "",
            "### 3.1 统计参数",
            "",
        ])
        report_lines.append(f"- 样本数量: {freq.n_samples} 场")
        report_lines.append(f"- 峰值流量均值: {freq.mean:.2f} m3/s")
        report_lines.append(f"- 标准差: {freq.std:.2f} m3/s")
        report_lines.append(f"- 变差系数(Cv): {freq.cv:.3f}")
        report_lines.append(f"- 偏度系数(Cs): {freq.cs:.3f}")
        report_lines.append("")
        report_lines.append("### 3.2 设计洪水成果")
        report_lines.append("")
        report_lines.append("| 重现期 | 设计流量(m3/s) |")
        report_lines.append("|--------|---------------|")
        for rp, val in freq.design_values.items():
            report_lines.append(f"| {rp} | {val:.2f} |")
        report_lines.append("")
        report_lines.extend([
            "## 4. 代表性洪水选取",
            "",
            f"根据多准则分析，选取以下 **{len(selected)}** 场代表性洪水用于模型率定：",
            "",
            "| 场次 | 峰值流量 | 选取原因 |",
            "|------|---------|---------|",
        ])
        for e in selected:
            report_lines.append(f"| {e.event_id} | {e.peak_flow:.2f} m3/s | {e.selection_reason} |")
        report_lines.append("")
        if call_llm:
            llm_prompt = _build_preanalysis_llm_prompt(quality, events, freq, selected)
            report_lines.extend([
                "## 5. LLM智能分析",
                "",
                "### 5.1 数据质量评价",
                "",
            ])
            stt = call_llm(llm_prompt)
            if not stt.startswith("[ERROR]"):
                report_lines.append(stt)
            else:
                report_lines.append("*LLM分析暂时不可用*")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*报告由 HydroTune-AI 系统自动生成*")
    
    return "\n".join(report_lines)


def _build_preanalysis_llm_prompt(
    quality: DataQualityResult,
    events: List[FloodEvent],
    freq: FrequencyAnalysisResult,
    selected: List[FloodEvent]
) -> str:
    """构建预分析的LLM提示词"""
    
    lang = get_language()
    event_summary = []
    for e in events[:10]:
        event_summary.append({
            'id': e.event_id,
            'peak': e.peak_flow,
            'volume': e.flood_volume,
            'duration': e.duration,
            'rise': e.rise_rate,
            'recession': e.recession_rate
        })
    
    if lang == 'en':
        prompt = f"""You are a senior hydrological data analyst. Please analyze the following hydrological data:

## Data Overview
- Catchment Area: {150.7944} km2
- Flood Events Identified: {len(events)}
- Representative Floods Selected: {len(selected)}

## Data Quality Assessment
- Completeness: {quality.completeness:.1f}%
- Continuity: {quality.continuity:.1f}%
- Correlation: {quality.correlation:.3f}
- Quality Level: {quality.quality_level}

## Flood Statistics
| Event | Peak (m3/s) | Volume (mm) | Duration | Rise Rate | Recession Rate |
"""
        for e in event_summary:
            prompt += f"| {e['id']} | {e['peak']:.1f} | {e['volume']:.1f} | {e['duration']} | {e['rise']:.3f} | {e['recession']:.3f} |\n"

        prompt += f"""
## Frequency Analysis
- Mean: {freq.mean:.2f} m3/s
- Cv: {freq.cv:.3f}
- Cs: {freq.cs:.3f}

Please analyze and evaluate from the following perspectives:
1. Does the data quality meet the requirements for model calibration?
2. Is the spatiotemporal distribution of flood events reasonable?
3. Are the selected representative floods truly representative?
4. How reliable are the frequency analysis results?
5. Recommendations for subsequent model calibration

Please respond in English with concise professional language, using Markdown format."""
    else:
        prompt = f"""你是一位资深水文数据分析师。请对以下水文数据进行预分析评价：

## 数据概况
- 流域面积: {150.7944} km2
- 识别洪水事件: {len(events)} 场
- 选取代表性洪水: {len(selected)} 场

## 数据质量评估
- 完整率: {quality.completeness:.1f}%
- 连续性: {quality.continuity:.1f}%
- 相关性: {quality.correlation:.3f}
- 质量等级: {quality.quality_level}

## 洪水统计
| 场次 | 峰值(m3/s) | 洪量(mm) | 历时 | 涨率 | 落率 |
"""
        for e in event_summary:
            prompt += f"| {e['id']} | {e['peak']:.1f} | {e['volume']:.1f} | {e['duration']} | {e['rise']:.3f} | {e['recession']:.3f} |\n"

        prompt += f"""
## 频率分析
- 均值: {freq.mean:.2f} m3/s
- Cv: {freq.cv:.3f}
- Cs: {freq.cs:.3f}

请从以下角度进行分析评价：
1. 数据质量是否满足模型率定要求？
2. 洪水事件的时空分布是否合理？
3. 代表性洪水选取是否具有代表性？
4. 频率分析结果的可靠性如何？
5. 对后续模型率定工作的建议

请用中文回复，语言简洁专业，使用Markdown格式。"""
    
    return prompt


def generate_calibration_report(
    calibration_results: Dict[str, Any],
    all_results: Dict[str, Dict[str, List[Dict]]],
    catchment_area: float,
    call_llm=call_minimax
) -> str:
    """
    Generate model calibration analysis report
    """
    lang = get_language()
    
    if lang == 'en':
        report_lines = [
            "# Hydrological Model Calibration Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Catchment Area**: {catchment_area} km2",
            "",
            "---",
            "",
            "## 1. Calibration Model Overview",
            "",
        ]
    else:
        report_lines = [
            "# 水文模型率定分析报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**流域面积**: {catchment_area} km2",
            "",
            "---",
            "",
            "## 1. 率定模型概览",
            "",
        ]
    
    for model_name, result in calibration_results.items():
        report_lines.append(f"### {model_name}")
        report_lines.append(f"- **NSE**: {result['nse']:.4f}")
        report_lines.append(f"- **RMSE**: {result['rmse']:.4f} m3/s")
        report_lines.append(f"- **MAE**: {result['mae']:.4f} m3/s")
        report_lines.append(f"- **PBIAS**: {result['pbias']:.1f}%")
        report_lines.append("")
    
    file_key = "File" if lang == 'en' else "文件"
    event_key = "Event" if lang == 'en' else "场次"
    model_key = "Model" if lang == 'en' else "模型"
    
    summary_data = []
    for file_name, file_results in all_results.items():
        for event_name, event_results in file_results.items():
            for r in event_results:
                summary_data.append({
                    file_key: file_name,
                    event_key: event_name,
                    model_key: r["model_name"],
                    "NSE": r["nse"],
                    "RMSE": r["rmse"],
                    "PBIAS": f"{r['pbias']:.1f}%"
                })
    
    if summary_data:
        if lang == 'en':
            report_lines.extend(["## 2. Per-Event Flood Simulation Results", "", ""])
        else:
            report_lines.extend(["## 2. 逐场次洪水模拟结果", "", ""])
        
        summary_df = pd.DataFrame(summary_data)
        
        for file_name in all_results.keys():
            report_lines.append(f"### {file_name}")
            file_data = summary_df[summary_df[file_key] == file_name]
            if not file_data.empty:
                report_lines.append(file_data.to_markdown(index=False))
            report_lines.append("")
    
    if lang == 'en':
        report_lines.extend(["## 3. Model Parameters", "", ""])
    else:
        report_lines.extend(["## 3. 模型参数", "", ""])
    
    for model_name, result in calibration_results.items():
        report_lines.append(f"### {model_name}")
        params = result['params']
        if isinstance(params, dict):
            for param_name, param_value in params.items():
                report_lines.append(f"- {param_name}: {param_value}")
        report_lines.append("")
    
    if call_llm and summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_md = summary_df.to_markdown(index=False)
        
        if lang == 'en':
            llm_prompt = f"""You are a senior hydrological expert. Please write a professional analysis report based on the following multi-event flood calibration results.

**Calibration Results Summary:**
{summary_md}

**Requirements:**
1. Analyze each model's performance across different flood events
2. Evaluate model stability and applicability
3. Provide overall recommendations
4. Use Markdown format
5. Keep language concise and professional, highlight key conclusions
"""
        else:
            llm_prompt = f"""你是一位资深水文专家。请基于以下多场次洪水率定结果，撰写一份专业的分析报告。

**率定结果汇总：**
{summary_md}

**要求：**
1. 分析各模型在不同场次洪水中的表现
2. 评估模型的稳定性和适用性
3. 给出综合推荐
4. 使用 Markdown 格式
5. 语言简洁专业，突出关键结论
"""
        stt = call_llm(llm_prompt)
        if not stt.startswith("[ERROR]"):
            if lang == 'en':
                report_lines.extend(["## 4. LLM Intelligent Analysis", "", stt])
            else:
                report_lines.extend(["## 4. LLM智能分析", "", stt])
        else:
            if lang == 'en':
                report_lines.extend(["## 4. Overall Assessment", "", "*LLM analysis temporarily unavailable*"])
            else:
                report_lines.extend(["## 4. 综合评价", "", "*LLM分析暂时不可用*"])
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    if lang == 'en':
        report_lines.append("*Report auto-generated by HydroTune-AI*")
    else:
        report_lines.append("*报告由 HydroTune-AI 系统自动生成*")
    
    return "\n".join(report_lines)


def generate_comprehensive_report(
    preanalysis_result: PreAnalysisResult,
    calibration_results: Dict[str, Any],
    all_results: Dict[str, Dict[str, List[Dict]]],
    call_llm=call_minimax
) -> str:
    """
    Generate comprehensive analysis report (pre-analysis + calibration results)
    """
    lang = get_language()
    
    if lang == 'en':
        ts_label = 'hourly' if preanalysis_result.timestep == 'hourly' else 'daily'
        report_lines = [
            "# Comprehensive Hydrological Model Calibration Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d')}",
            f"**Catchment Area**: {preanalysis_result.area} km²",
            f"**Timestep**: {ts_label}",
            "",
            "---",
            "",
        ]
        model_key = "Model"
        calib_nse_key = "Calib NSE"
        calib_rmse_key = "Calib RMSE"
    else:
        report_lines = [
            "# 水文模型率定综合分析报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d')}",
            f"**流域面积**: {preanalysis_result.area} km²",
            f"**时间尺度**: {'小时' if preanalysis_result.timestep == 'hourly' else '日'}尺度",
            "",
            "---",
            "",
        ]
        model_key = "模型"
        calib_nse_key = "率定NSE"
        calib_rmse_key = "率定RMSE"
    
    if call_llm:
        summary_data = []
        calib_data = []
        for file_name, file_results in all_results.items():
            for event_name, event_results in file_results.items():
                for r in event_results:
                    summary_data.append({
                        model_key: r["model_name"],
                        "NSE": round(r["nse"], 4),
                        "RMSE": round(r["rmse"], 2),
                        "PBIAS": r["pbias"]
                    })
        
        for model_name, result in calibration_results.items():
            calib_data.append({
                model_key: model_name,
                calib_nse_key: round(result['nse'], 4),
                calib_rmse_key: round(result['rmse'], 2)
            })
        
        if lang == 'en':
            llm_prompt = f"""You are a senior hydrological expert. Please write a concise and rigorous comprehensive analysis report based on the following hydrological model calibration results.

## 1. Basic Information
- Catchment Area: {preanalysis_result.area} km²
- Flood Events Identified: {len(preanalysis_result.events)}
- Representative Floods (Calibration): {len(preanalysis_result.selected_events)}
- Data Quality: {preanalysis_result.quality.quality_level}

## 2. Calibration Results
| Model | Calib NSE | Calib RMSE |
|-------|-----------|------------|
""" + "\n".join([f"| {d[model_key]} | {d[calib_nse_key]} | {d[calib_rmse_key]} |" for d in calib_data]) + f"""

## 3. Validation Results
| Model | Avg NSE | Avg RMSE |
|-------|---------|----------|
"""
        else:
            llm_prompt = f"""你是一位资深水文专家。请基于以下水文模型率定结果，撰写一份精炼严谨的综合分析报告。

## 1. 基本信息
- 流域面积: {preanalysis_result.area} km²
- 识别洪水: {len(preanalysis_result.events)} 场
- 代表性洪水(率定): {len(preanalysis_result.selected_events)} 场
- 数据质量: {preanalysis_result.quality.quality_level}

## 2. 率定结果统计
| 模型 | 率定NSE | 率定RMSE |
|------|---------|----------|
""" + "\n".join([f"| {d[model_key]} | {d[calib_nse_key]} | {d[calib_rmse_key]} |" for d in calib_data]) + f"""

## 3. 验证结果
| 模型 | 平均NSE | 平均RMSE |
|------|--------|----------|
"""
        
        model_stats = {}
        for d in summary_data:
            m = d[model_key]
            if m not in model_stats:
                model_stats[m] = {'nse': [], 'rmse': []}
            model_stats[m]['nse'].append(d['NSE'])
            model_stats[m]['rmse'].append(d['RMSE'])
        
        for m, stats in model_stats.items():
            avg_nse = sum(stats['nse']) / len(stats['nse'])
            avg_rmse = sum(stats['rmse']) / len(stats['rmse'])
            llm_prompt += f"| {m} | {avg_nse:.4f} | {avg_rmse:.2f} |\n"
        
        if lang == 'en':
            llm_prompt += f"""
## 4. Representative Flood Selection
"""
            for e in preanalysis_result.selected_events[:5]:
                llm_prompt += f"- {e.event_id}: Peak {e.peak_flow:.1f} m³/s ({e.selection_reason})\n"
            
            llm_prompt += f"""
## 5. Deep Learning Integration Recommendations
Please provide suggestions from the following perspectives:
1. Limitations of current conceptual models (Tank, HBV, Xinanjiang)
2. Feasibility of deep learning (LSTM, Transformer, Neural ODE, etc.) for improving hydrological simulation
3. Model-data hybrid architecture recommendations (e.g., LSTM embedding, model error correction)
4. Key input feature recommendations (weather forecasts, remote sensing ET, soil moisture, etc.)
5. Technical implementation path and data requirements

Write a professional, rigorous report within 1200 words, using Markdown format, highlighting key conclusions."""
        else:
            llm_prompt += f"""
## 4. 代表性洪水选取
"""
            for e in preanalysis_result.selected_events[:5]:
                llm_prompt += f"- {e.event_id}: 峰值{e.peak_flow:.1f}m³/s ({e.selection_reason})\n"
            
            llm_prompt += f"""
## 5. 深度学习耦合建议
请从以下角度给出建议：
1. 当前概念性模型（如Tank、HBV、新安江）的局限性
2. 深度学习（ LSTM、Transformer、Neural ODE等）改进水文模拟的可行性
3. 模型-数据双驱动混合架构建议（如LSTM嵌入、模型误差修正）
4. 关键输入特征建议（气象预报、遥感蒸散发、土壤湿度等）
5. 技术实施路径与数据需求

请用专业、严谨的语言撰写报告，控制在800字以内，使用Markdown格式，突出关键结论。"""
        
        stt = call_llm(llm_prompt)
        if not stt.startswith("[ERROR]"):
            report_lines.append(stt)
        else:
            if lang == 'en':
                report_lines.extend(["## Calibration Results", "", "*LLM analysis temporarily unavailable*", ""])
            else:
                report_lines.extend(["## 率定结果", "", "*LLM分析暂时不可用*", ""])
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    if lang == 'en':
        report_lines.append("*Report auto-generated by HydroTune-AI*")
    else:
        report_lines.append("*本报告由 HydroTune-AI 系统自动生成*")
    
    return "\n".join(report_lines)


def generate_multifile_report(
    file_data_list: list,
    calibration_results: Dict[str, Any],
    file_simulation_results: Dict[str, Dict[str, Any]],
    call_llm=call_minimax,
    warmup_hours: int = 0,
    xgb_correction_results: List[Dict] = None,
    comparison_results: Optional[List[Dict]] = None,
) -> str:
    """Compare chart results; legacy callers retain the original signature.

    comparison_results accepts canonical metric records or bilingual chart rows.
    Legacy XGB NSE deltas lack the full metrics and are not treated as evidence
    for a five-model ranking.
    """
    from src.model_comparison import build_comparison_report
    return build_comparison_report(
        file_data_list, calibration_results, file_simulation_results,
        call_llm, warmup_hours, comparison_results, get_language(),
    )

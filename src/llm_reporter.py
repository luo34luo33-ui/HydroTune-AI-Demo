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

from src.llm_api import call_minimax
from src.data_preanalysis import (
    PreAnalysisResult, DataQualityResult, FloodEvent, FrequencyAnalysisResult
)


def generate_preanalysis_report(
    result: PreAnalysisResult,
    call_llm=call_minimax
) -> str:
    """
    生成数据预分析报告
    
    Args:
        result: 预分析结果
        call_llm: LLM调用函数
        
    Returns:
        Markdown格式的报告
    """
    quality = result.quality
    events = result.events
    freq = result.frequency
    selected = result.selected_events
    
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
    生成模型率定分析报告
    
    Args:
        calibration_results: 率定结果
        all_results: 所有洪水模拟结果
        catchment_area: 流域面积
        call_llm: LLM调用函数
        
    Returns:
        Markdown格式的报告
    """
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
    
    summary_data = []
    for file_name, file_results in all_results.items():
        for event_name, event_results in file_results.items():
            for r in event_results:
                summary_data.append({
                    "文件": file_name,
                    "场次": event_name,
                    "模型": r["model_name"],
                    "NSE": r["nse"],
                    "RMSE": r["rmse"],
                    "PBIAS": f"{r['pbias']:.1f}%"
                })
    
    if summary_data:
        report_lines.extend([
            "## 2. 逐场次洪水模拟结果",
            "",
        ])
        
        summary_df = pd.DataFrame(summary_data)
        
        for file_name in all_results.keys():
            report_lines.append(f"### {file_name}")
            file_data = summary_df[summary_df['文件'] == file_name]
            if not file_data.empty:
                report_lines.append(file_data.to_markdown(index=False))
            report_lines.append("")
    
    report_lines.extend([
        "## 3. 模型参数",
        "",
    ])
    
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
            report_lines.extend([
                "## 4. LLM智能分析",
                "",
                stt
            ])
        else:
            report_lines.extend([
                "## 4. 综合评价",
                "",
                "*LLM分析暂时不可用*"
            ])
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*报告由 HydroTune-AI 系统自动生成*")
    
    return "\n".join(report_lines)


def generate_comprehensive_report(
    preanalysis_result: PreAnalysisResult,
    calibration_results: Dict[str, Any],
    all_results: Dict[str, Dict[str, List[Dict]]],
    call_llm=call_minimax
) -> str:
    """
    生成综合分析报告（包含预分析和率定结果）
    """
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
    
    if call_llm:
        summary_data = []
        calib_data = []
        for file_name, file_results in all_results.items():
            for event_name, event_results in file_results.items():
                for r in event_results:
                    summary_data.append({
                        "模型": r["model_name"],
                        "NSE": round(r["nse"], 4),
                        "RMSE": round(r["rmse"], 2),
                        "PBIAS": r["pbias"]
                    })
        
        for model_name, result in calibration_results.items():
            calib_data.append({
                "模型": model_name,
                "率定NSE": round(result['nse'], 4),
                "率定RMSE": round(result['rmse'], 2)
            })
        
        llm_prompt = f"""你是一位资深水文专家。请基于以下水文模型率定结果，撰写一份精炼严谨的综合分析报告。

## 1. 基本信息
- 流域面积: {preanalysis_result.area} km²
- 识别洪水: {len(preanalysis_result.events)} 场
- 代表性洪水(率定): {len(preanalysis_result.selected_events)} 场
- 数据质量: {preanalysis_result.quality.quality_level}

## 2. 率定结果统计
| 模型 | 率定NSE | 率定RMSE |
|------|---------|----------|
""" + "\n".join([f"| {d['模型']} | {d['率定NSE']} | {d['率定RMSE']} |" for d in calib_data]) + f"""

## 3. 验证结果
| 模型 | 平均NSE | 平均RMSE |
|------|--------|----------|
"""
        
        model_stats = {}
        for d in summary_data:
            m = d['模型']
            if m not in model_stats:
                model_stats[m] = {'nse': [], 'rmse': []}
            model_stats[m]['nse'].append(d['NSE'])
            model_stats[m]['rmse'].append(d['RMSE'])
        
        for m, stats in model_stats.items():
            avg_nse = sum(stats['nse']) / len(stats['nse'])
            avg_rmse = sum(stats['rmse']) / len(stats['rmse'])
            llm_prompt += f"| {m} | {avg_nse:.4f} | {avg_rmse:.2f} |\n"
        
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
            report_lines.extend([
                "## 率定结果",
                "",
                "*LLM分析暂时不可用*",
                ""
            ])
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*本报告由 HydroTune-AI 系统自动生成*")
    
    return "\n".join(report_lines)


def generate_multifile_report(
    file_data_list: list,
    calibration_results: Dict[str, Any],
    file_simulation_results: Dict[str, Dict[str, Any]],
    call_llm=call_minimax,
    warmup_hours: int = 0
) -> str:
    """
    生成多文件模式分析报告（率定-验证分开）
    
    Args:
        file_data_list: 文件数据列表
        calibration_results: 率定结果
        file_simulation_results: 各文件模拟结果，包含 is_calib 标记
        call_llm: LLM调用函数
        warmup_hours: 预热期小时数
        
    Returns:
        Markdown格式的报告
    """
    report_lines = [
        "# 水文模型率定验证分析报告（多文件模式）",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 分析概述",
        "",
    ]
    
    n_files = len(file_data_list)
    n_models = len(calibration_results)
    
    report_lines.append(f"- **分析场次**: {n_files} 场洪水")
    report_lines.append(f"- **率定模型**: {n_models} 个")
    report_lines.append(f"- **率定策略**: 随机选取约1/3场次率定，其余2/3验证")
    if warmup_hours > 0:
        report_lines.append(f"- **预热期**: {warmup_hours} 小时（已从计算中剔除）")
    else:
        report_lines.append(f"- **预热期**: 无")
    report_lines.append("")
    
    calib_file_names = set()
    valid_file_names = set()
    for model_name, file_results in file_simulation_results.items():
        for file_name, result in file_results.items():
            if result.get('is_calib', False):
                calib_file_names.add(file_name)
            else:
                valid_file_names.add(file_name)
    
    report_lines.extend([
        "## 2. 场次分组",
        "",
        f"- **率定场次** ({len(calib_file_names)}场): {', '.join(sorted(calib_file_names))}",
        f"- **验证场次** ({len(valid_file_names)}场): {', '.join(sorted(valid_file_names))}",
        "",
    ])
    
    if call_llm:
        table_data = []
        for model_name, file_results in file_simulation_results.items():
            for file_name, result in file_results.items():
                is_calib = result.get('is_calib', False)
                table_data.append({
                    "文件": file_name,
                    "类型": "率定" if is_calib else "验证",
                    "模型": model_name,
                    "NSE": round(result['nse'], 4),
                    "RMSE": round(result['rmse'], 2),
                    "PBIAS(%)": round(result['pbias'], 1)
                })
        
        model_stats = {}
        for model_name in file_simulation_results.keys():
            calib_nses = []
            valid_nses = []
            for file_name, result in file_simulation_results[model_name].items():
                if result.get('is_calib', False):
                    calib_nses.append(result['nse'])
                else:
                    valid_nses.append(result['nse'])
            
            model_stats[model_name] = {
                'calib_nse_avg': np.mean(calib_nses) if calib_nses else 0,
                'calib_nse_std': np.std(calib_nses) if calib_nses else 0,
                'valid_nse_avg': np.mean(valid_nses) if valid_nses else 0,
                'valid_nse_std': np.std(valid_nses) if valid_nses else 0,
                'calib_n': len(calib_nses),
                'valid_n': len(valid_nses),
            }
        
        llm_prompt = f"""你是一位资深水文专家。请基于以下多场次洪水模型率定验证结果，撰写一份专业的分析报告。

## 1. 场次分组
- 率定场次: {len(calib_file_names)}场 - {', '.join(sorted(calib_file_names))}
- 验证场次: {len(valid_file_names)}场 - {', '.join(sorted(valid_file_names))}

## 2. 各场次模拟结果
| 文件 | 类型 | 模型 | NSE | RMSE | PBIAS(%) |
|------|------|------|-----|------|-----------|
"""
        for d in table_data:
            llm_prompt += f"| {d['文件']} | {d['类型']} | {d['模型']} | {d['NSE']:.4f} | {d['RMSE']:.2f} | {d['PBIAS(%)']:.1f} |\n"
        
        llm_prompt += f"""
## 3. 模型表现统计
| 模型 | 率定NSE(均) | 率定NSE(标) | 验证NSE(均) | 验证NSE(标) |
|------|------------|------------|------------|------------|
"""
        for model_name, stats in model_stats.items():
            llm_prompt += f"| {model_name} | {stats['calib_nse_avg']:.4f} | {stats['calib_nse_std']:.4f} | {stats['valid_nse_avg']:.4f} | {stats['valid_nse_std']:.4f} |\n"
        
        llm_prompt += f"""
## 4. 率定参数
"""
        for model_name, calib_result in calibration_results.items():
            llm_prompt += f"\n### {model_name}\n"
            llm_prompt += f"- 率定NSE: {calib_result['nse']:.4f}\n"
            params = calib_result['params']
            for k, v in list(params.items())[:8]:
                llm_prompt += f"- {k}: {v:.4f}\n"
        
        llm_prompt += """
## 5. 分析要求
请从以下角度撰写报告（控制在600字以内）：

1. **模型泛化能力评估**: 对比率定期和验证期的NSE，判断模型是否过拟合
2. **模型稳定性分析**: 比较不同场次间NSE的变异系数，评估模型稳定性
3. **最优模型推荐**: 综合率定和验证表现，推荐最适合的模型
4. **改进建议**: 如存在过拟合或表现不佳，提出改进方向

请用专业、严谨的语言，突出关键结论。"""

        stt = call_llm(llm_prompt)
        if not stt.startswith("[ERROR]"):
            report_lines.append("## 6. LLM 智能分析")
            report_lines.append("")
            report_lines.append(stt)
        else:
            report_lines.extend([
                "## 6. 模型性能小结",
                "",
                "*LLM分析暂时不可用，基于数据生成小结如下：*",
                "",
            ])
            
            for model_name, stats in model_stats.items():
                report_lines.append(f"**{model_name}**:")
                report_lines.append(f"- 率定期平均NSE: {stats['calib_nse_avg']:.4f} (±{stats['calib_nse_std']:.4f})")
                report_lines.append(f"- 验证期平均NSE: {stats['valid_nse_avg']:.4f} (±{stats['valid_nse_std']:.4f})")
                
                diff = stats['valid_nse_avg'] - stats['calib_nse_avg']
                if diff < -0.1:
                    report_lines.append("- ⚠️ 验证期表现明显下降，存在过拟合风险")
                elif diff > 0.1:
                    report_lines.append("- ✓ 验证期表现优于率定期，泛化能力较强")
                else:
                    report_lines.append("- ✓ 率定和验证表现一致，模型稳定")
                report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "*本报告由 HydroTune-AI 系统自动生成*",
    ])
    
    return "\n".join(report_lines)

"""
Hydromind-Demo 流域水文模型智能率定系统
Streamlit 主入口
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from typing import List, Dict, Tuple
from io import StringIO

from src.llm_api import call_minimax
from src.data_agent import (
    clean_data_with_sandbox, infer_timestep, get_timestep_info,
    detect_flood_events, FloodEvent
)
from src.hydro_calc import (
    calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias,
    get_model_param_info, generate_param_table
)
from src.models.registry import ModelRegistry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="Hydromind - 流域水文模型智能率定系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 标题区域
# ============================================================
st.title("Hydromind - 流域水文模型智能率定系统")
st.caption("上传数据 → 智能清洗 → 多模型率定 → 自动报告")

# ============================================================
# 侧边栏
# ============================================================
RECOMMENDED_MODELS = ['水箱模型', 'HBV模型', '新安江模型']
SKIP_MODELS = ['Tank水箱模型', 'HBV模型(完整版)']

with st.sidebar:
    st.header("📁 数据上传")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "上传水文数据文件",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="支持多文件上传",
        )
    with col2:
        catchment_area = st.number_input(
            "流域面积",
            min_value=0.1,
            max_value=100000.0,
            value=150.7944,
            step=1.0,
            format="%.4f",
            help="单位：km²",
        )

    upload_mode = st.radio(
        "上传模式",
        options=["单文件（自动识别多场次）", "多文件（每文件一场洪水）"],
        index=0,
        horizontal=True,
    )

    st.divider()

    st.header("⚙️ 率定设置")
    max_iter = st.slider(
        "迭代次数",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="两阶段算法：阶段1全局搜索，阶段2局部优化",
    )

    st.divider()

    st.header("📊 率定模型")
    for model_name in RECOMMENDED_MODELS:
        st.write(f"✅ {model_name}")
    
    st.caption("注：完整版模型暂不支持，使用简化版")


# ============================================================
# 主流程
# ============================================================
if uploaded_files and len(uploaded_files) > 0:

    all_results = {}
    all_flood_events = []
    report_sections = []
    
    # ---- 处理每个文件 ----
    for file_idx, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(f"📂 文件 {file_idx + 1}: {uploaded_file.name}")
        
        # 读取数据
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            st.write(f"✅ 读取成功，共 {raw_df.shape[0]} 行")
        except Exception as e:
            st.error(f"读取失败: {e}")
            continue
        
        # 数据清洗
        with st.expander("🧠 数据清洗"):
            try:
                clean_df, detected_timestep = clean_data_with_sandbox(raw_df, call_minimax)
                st.write("✅ 清洗完成")
                st.dataframe(clean_df.head(10))
            except Exception as e:
                st.error(f"清洗失败: {e}")
                continue
        
        if "precip" not in clean_df.columns or "flow" not in clean_df.columns:
            st.error("缺少必要列")
            continue
        
        clean_df = clean_df.fillna(0)
        
        # 时间尺度确认
        col1, col2 = st.columns([2, 1])
        with col1:
            timestep_info = get_timestep_info(detected_timestep)
            st.info(f"⏱️ {timestep_info['label']}")
        with col2:
            user_timestep = st.radio(
                "尺度",
                options=['hourly', 'daily'],
                index=0 if detected_timestep == 'hourly' else 1,
                horizontal=True,
                label_visibility="collapsed",
                key=f"ts_{file_idx}"
            )
        
        # 洪水场次识别
        st.write("**🔍 洪水场次识别**")
        precip_arr = np.array(clean_df['precip'].values)
        flow_arr = np.array(clean_df['flow'].values)
        evap_arr = np.array(clean_df['evap'].values)
        
        flood_events = detect_flood_events(
            clean_df['date'],
            precip_arr,
            flow_arr,
            evap_arr
        )
        
        if len(flood_events) == 0:
            flood_events = [FloodEvent(
                name="全部数据",
                start_idx=0,
                end_idx=len(clean_df) - 1,
                start_date=clean_df['date'].iloc[0],
                end_date=clean_df['date'].iloc[-1],
                precip=precip_arr,
                evap=evap_arr,
                observed_flow=flow_arr
            )]
        
        st.write(f"识别到 {len(flood_events)} 场洪水")
        
        file_results = {}
        
        # 率定每场洪水
        for event in flood_events:
            event_key = f"{uploaded_file.name}_{event.name}"
            all_flood_events.append(event_key)
            
            st.write(f"**📊 率定 {event.name}** ({event.start_date} ~ {event.end_date})")
            
            def calibrate_event(event_obj, model_name):
                try:
                    spatial_data = {'area': catchment_area}
                    return calibrate_model_fast(
                        model_name,
                        event_obj.precip,
                        event_obj.evap,
                        event_obj.observed_flow,
                        max_iter=max_iter,
                        spatial_data=spatial_data,
                        timestep=user_timestep
                    )
                except Exception as e:
                    return None
            
            event_results = []
            with st.spinner(f"正在率定..."):
                for model_name in RECOMMENDED_MODELS:
                    if model_name in SKIP_MODELS:
                        continue
                    result = calibrate_event(event, model_name)
                    if result:
                        params, nse, simulated = result
                        event_results.append({
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(event.observed_flow, simulated),
                            "mae": calc_mae(event.observed_flow, simulated),
                            "pbias": calc_pbias(event.observed_flow, simulated),
                            "simulated": simulated,
                            "observed": event.observed_flow,
                        })
                        st.write(f"  ✅ {model_name}: NSE = {nse:.4f}")
            
            if event_results:
                event_results.sort(key=lambda x: x["nse"], reverse=True)
                file_results[event.name] = event_results
        
        all_results[uploaded_file.name] = file_results
    
    # ============================================================
    # 结果展示
    # ============================================================
    st.divider()
    st.subheader("📈 率定结果")
    
    # 汇总指标表
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
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # 每个模型的参数表格
        st.divider()
        st.subheader("📋 模型参数详情")
        
        for file_name, file_results in all_results.items():
            with st.expander(f"📁 {file_name}"):
                for event_name, event_results in file_results.items():
                    st.markdown(f"**{event_name}**")
                    
                    for r in event_results:
                        st.markdown(f"*{r['model_name']}* (NSE={r['nse']:.4f})")
                        param_df = generate_param_table(r['model_name'], r['params'])
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
        
        # 流量过程线对比图
        st.divider()
        st.subheader("📉 流量过程线对比")
        
        n_events = len(all_flood_events)
        n_cols = min(2, n_events)
        n_rows = (n_events + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_events == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        
        for idx, (file_name, file_results) in enumerate(all_results.items()):
            if idx < len(axes):
                ax = axes[idx]
                
                first_observed = None
                for e_idx, (event_name, event_results) in enumerate(file_results.items()):
                    for m_idx, r in enumerate(event_results):
                        label = f"{r['model_name']} (NSE={r['nse']:.2f})"
                        ax.plot(r["simulated"], color=colors[m_idx % len(colors)], 
                               label=label, linewidth=1, alpha=0.7)
                        if first_observed is None:
                            first_observed = r["observed"]
                
                if first_observed is not None:
                    ax.plot(first_observed, "k-", label="实测", linewidth=2)
                ax.set_title(f"{file_name}")
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("时间步")
                ax.set_ylabel("流量 (m³/s)")
        
        for idx in range(len(all_results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # ============================================================
    # 智能分析报告
    # ============================================================
    st.divider()
    st.subheader("📝 智能分析报告")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_md = summary_df.to_markdown(index=False)
        
        report_prompt = f"""你是一位资深水文专家。请基于以下多场次洪水率定结果，撰写一份专业的分析报告。

**率定结果汇总：**
{summary_md}

**要求：**
1. 分析各模型在不同场次洪水中的表现
2. 评估模型的稳定性和适用性
3. 给出综合推荐
4. 使用 Markdown 格式
"""
        
        with st.spinner("正在生成报告..."):
            report = call_minimax(report_prompt)
            if report.startswith("[ERROR]"):
                report = f"## 水文模型率定分析报告\n\n报告生成失败: {report}"
    else:
        report = "## 暂无率定结果"
    
    st.markdown(report)
    
    # 导出报告按钮
    if summary_data and not report.startswith("[ERROR]"):
        report_with_header = f"""# 水文模型率定分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{report}
"""
        
        st.download_button(
            "📥 导出报告 (Markdown)",
            data=report_with_header,
            file_name=f"calibration_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

else:
    # 欢迎页面
    st.info("👈 请在左侧上传水文数据文件开始分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("支持模型", f"{len(RECOMMENDED_MODELS)} 个")
    with col2:
        st.metric("支持格式", "CSV / Excel")
    with col3:
        st.metric("率定算法", "两阶段(并行)")
    
    st.divider()
    
    st.subheader("📖 使用说明")
    st.markdown("""
    1. **上传数据**：上传一个或多个水文数据文件
    2. **洪水场次**：系统自动识别每场洪水
    3. **自动率定**：对每场洪水分别进行多模型率定
    4. **结果对比**：查看性能指标、参数表格、流量过程线
    5. **智能报告**：自动生成分析报告并支持导出
    """)
    
    st.subheader("📊 数据格式要求")
    st.markdown("""
    数据应包含以下列（列名支持中英文）：
    - **时间列**: date / time / 日期 / 时间
    - **降水列**: precip / rainfall / p / 降水 / 降雨
    - **蒸发列**: evap / et / 蒸发 (可选)
    - **流量列**: flow / discharge / q / 流量 / 径流
    """)

# 欢迎页面（无文件上传时显示）
if not uploaded_files:
    st.info("👈 请在左侧上传水文数据文件开始分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("支持模型", f"{len(RECOMMENDED_MODELS)} 个")
    with col2:
        st.metric("支持格式", "CSV / Excel")
    with col3:
        st.metric("率定算法", "两阶段(并行)")
    
    st.divider()
    
    st.subheader("📖 使用说明")
    st.markdown("""
    1. **上传数据**：上传一个或多个水文数据文件
    2. **洪水场次**：系统自动识别每场洪水
    3. **自动率定**：对每场洪水分别进行多模型率定
    4. **结果对比**：查看性能指标、参数表格、流量过程线
    5. **智能报告**：自动生成分析报告并支持导出
    """)
    
    st.subheader("📊 数据格式要求")
    st.markdown("""
    数据应包含以下列（列名支持中英文）：
    - **时间列**: date / time / 日期 / 时间
    - **降水列**: precip / rainfall / p / 降水 / 降雨
    - **蒸发列**: evap / et / 蒸发 (可选)
    - **流量列**: flow / discharge / q / 流量 / 径流
    """)

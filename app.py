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
        options=["单文件（整场洪水）", "多文件（每文件一场洪水）"],
        index=0,
        horizontal=True,
    )

    st.divider()

    # 列名配置
    st.header("📋 列名配置")
    with st.expander("配置数据列名映射"):
        st.write("将原始列名映射到标准列名（date, precip, evap, flow）：")
        date_col = st.text_input("时间列名", value="date", key="date_col")
        precip_col = st.text_input("降水列名", value="precip", key="precip_col")
        evap_col = st.text_input("蒸发列名（可选）", value="evap", key="evap_col")
        flow_col = st.text_input("流量列名", value="flow", key="flow_col")
    
    column_mapping = {
        'date': date_col if date_col else 'date',
        'precip': precip_col if precip_col else 'precip',
        'evap': evap_col if evap_col else 'evap',
        'flow': flow_col if flow_col else 'flow',
    }

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
    
    # ---- 列名确认（第一个文件）----
    st.divider()
    st.subheader("📋 列名确认")
    
    # 读取第一个文件显示列名
    first_file = uploaded_files[0]
    try:
        if first_file.name.endswith(".csv"):
            first_df = pd.read_csv(first_file)
        else:
            first_df = pd.read_excel(first_file)
        st.write(f"**{first_file.name}** 的列名: `{list(first_df.columns)}`")
    except Exception as e:
        st.error(f"读取失败: {e}")
        st.stop()
    
    # 列名映射确认
    col_mapping_accepted = st.checkbox(
        f"使用列名映射: {column_mapping['date']}→date, {column_mapping['precip']}→precip, {column_mapping['evap']}→evap, {column_mapping['flow']}→flow",
        value=True
    )
    
    if not col_mapping_accepted:
        st.warning("请先配置正确的列名映射")
        st.stop()
    
    # ---- 处理每个文件 ----
    all_precip = []
    all_evap = []
    all_flow = []
    all_dates = []
    all_file_events = []
    detected_timestep = 'daily'
    user_timestep = 'daily'
    
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
        
        # 使用用户配置的列名进行映射
        rename_map = {}
        for std_name, orig_name in column_mapping.items():
            if orig_name and orig_name in raw_df.columns:
                rename_map[orig_name] = std_name
        
        if rename_map:
            raw_df = raw_df.rename(columns=rename_map)
        
        # 确保必要列存在
        if 'precip' not in raw_df.columns or 'flow' not in raw_df.columns:
            st.error(f"缺少必要列。当前列: {list(raw_df.columns)}")
            continue
        
        if 'evap' not in raw_df.columns:
            raw_df['evap'] = 0.0
        
        if 'date' not in raw_df.columns:
            raw_df['date'] = range(len(raw_df))
        
        clean_df = raw_df.fillna(0)
        
        # 检测时间尺度（使用第一个文件的时间尺度）
        if file_idx == 0:
            detected_timestep = infer_timestep(clean_df['date'])
        
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
        
        # 收集每场洪水数据用于后续绘图
        for event in flood_events:
            if hasattr(event.start_date, 'strftime'):
                event_date_str = event.start_date.strftime('%Y%m%d')
            else:
                event_date_str = str(event.start_date)[:10].replace('-', '')
            
            all_file_events.append({
                'file_name': uploaded_file.name,
                'event_name': event_date_str,
                'start_date': event.start_date,
                'end_date': event.end_date,
                'precip': event.precip,
                'evap': event.evap,
                'observed_flow': event.observed_flow,
            })
        
        # 收集所有数据用于率定
        all_precip.extend(precip_arr.tolist())
        all_evap.extend(evap_arr.tolist())
        all_flow.extend(flow_arr.tolist())
        
        with st.expander("查看数据"):
            st.dataframe(clean_df.head(10))
    
    # 时间尺度确认
    st.divider()
    st.subheader("⏱️ 时间尺度确认")
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
            label_visibility="collapsed"
        )
    
    # ============================================================
    # 合并所有数据，统一率定
    # ============================================================
    st.divider()
    st.subheader("🌊 统一率定（共用最优参数）")
    
    all_precip = np.array(all_precip)
    all_evap = np.array(all_evap)
    all_flow = np.array(all_flow)
    
    st.write(f"📊 合并数据：共 {len(all_precip)} 个时间步")
    
    # 使用合并后的数据进行率定
    calibration_results = {}
    
    def calibrate_model(model_name):
        try:
            spatial_data = {'area': catchment_area}
            return calibrate_model_fast(
                model_name,
                all_precip,
                all_evap,
                all_flow,
                max_iter=max_iter,
                spatial_data=spatial_data,
                timestep=user_timestep
            )
        except Exception as e:
            return None
    
    with st.spinner("正在率定..."):
        for model_name in RECOMMENDED_MODELS:
            if model_name in SKIP_MODELS:
                continue
            result = calibrate_model(model_name)
            if result:
                params, nse, simulated = result
                calibration_results[model_name] = {
                    "model_name": model_name,
                    "params": params,
                    "nse": nse,
                    "rmse": calc_rmse(all_flow, simulated),
                    "mae": calc_mae(all_flow, simulated),
                    "pbias": calc_pbias(all_flow, simulated),
                    "simulated": simulated,
                }
                st.write(f"  ✅ {model_name}: NSE = {nse:.4f}")
    
    # ============================================================
    # 使用统一参数模拟每场洪水
    # ============================================================
    st.divider()
    st.subheader("📊 各场次洪水模拟")
    
    all_results = {}
    
    for model_name, calib_result in calibration_results.items():
        params = calib_result['params']
        
        for event_info in all_file_events:
            file_name = event_info['file_name']
            event_name = event_info['event_name']
            
            if file_name not in all_results:
                all_results[file_name] = {}
            
            try:
                spatial_data = {'area': catchment_area, 'timestep': user_timestep}
                model = ModelRegistry.get_model(model_name)
                simulated = model.run(
                    event_info['precip'],
                    event_info['evap'],
                    params,
                    spatial_data
                )
                
                event_result = {
                    "model_name": model_name,
                    "params": params,
                    "nse": calc_nse(event_info['observed_flow'], simulated),
                    "rmse": calc_rmse(event_info['observed_flow'], simulated),
                    "mae": calc_mae(event_info['observed_flow'], simulated),
                    "pbias": calc_pbias(event_info['observed_flow'], simulated),
                    "simulated": simulated,
                    "observed": event_info['observed_flow'],
                }
                
                if event_name not in all_results[file_name]:
                    all_results[file_name][event_name] = []
                all_results[file_name][event_name].append(event_result)
                
            except Exception as e:
                st.write(f"  ❌ {file_name}/{event_name}/{model_name}: {e}")
    
    # 排序
    for file_name in all_results:
        for event_name in all_results[file_name]:
            all_results[file_name][event_name].sort(key=lambda x: x["nse"], reverse=True)
    
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
        
        model_colors = {
            '新安江模型': '#e74c3c',  # 红色
            '水箱模型': '#3498db',     # 蓝色
            'HBV模型': '#2ecc71',      # 绿色
        }
        
        # 计算总场次数
        n_events = sum(len(file_results) for file_results in all_results.values())
        n_cols = min(2, n_events)
        n_rows = (n_events + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        if n_events == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, (file_name, file_results) in enumerate(all_results.items()):
            if idx < len(axes):
                ax = axes[idx]
                
                first_observed = None
                for event_name, event_results in file_results.items():
                    for r in event_results:
                        color = model_colors.get(r['model_name'], '#999999')
                        label = f"{r['model_name']} (NSE={r['nse']:.2f})"
                        ax.plot(r["simulated"], color=color, 
                               label=label, linewidth=2, alpha=0.8)
                        if first_observed is None:
                            first_observed = r["observed"]
                
                if first_observed is not None:
                    ax.plot(first_observed, "k-", label="实测", linewidth=2.5)
                
                ax.set_title(f"{file_name}")
                ax.legend(fontsize=9, loc='upper right')
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

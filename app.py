"""
HydroTune-AI 流域水文模型智能率定系统
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
    clean_data_with_sandbox, infer_timestep, infer_timestep_by_llm, get_timestep_info,
    detect_flood_events, FloodEvent
)
from src.hydro_calc import (
    calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias,
    get_model_param_info, generate_param_table
)
from src.data_preanalysis import (
    DataPreAnalyzer, PreAnalysisResult, FloodEvent as PAFloodEvent,
    DataQualityResult, FrequencyAnalysisResult
)
from src.llm_reporter import (
    generate_preanalysis_report, generate_calibration_report,
    generate_comprehensive_report, generate_multifile_report
)
from src.models.registry import ModelRegistry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="HydroTune-AI - 流域水文模型智能率定系统",
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
st.title("HydroTune-AI - 流域水文模型智能率定系统")
st.caption("上传数据 → 智能清洗 → 多模型率定 → 自动报告")

# ============================================================
# 侧边栏
# ============================================================
RECOMMENDED_MODELS = ['水箱模型', 'HBV模型', '新安江模型']
SKIP_MODELS = ['Tank水箱模型', 'HBV模型(完整版)']

with st.sidebar:
    st.markdown("### 🤖 AI Agent 状态")
    agent_status = st.empty()
    agent_status.success("🟢 智能Agent就绪")
    
    st.divider()
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
        "📂 上传模式",
        options=[
            "单文件（一场洪水）",      # 不识别场次，整场分析
            "单文件（连续序列）",      # 连续序列，识别多场洪水
            "多文件（每文件一场洪水）" # 每文件一场洪水
        ],
        index=0,
        horizontal=False,
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
# 欢迎页面
# ============================================================
if not uploaded_files:
    st.markdown("""
    <style>
    .hero-section {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 40px;
        border-radius: 20px;
        margin: 20px 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    .hero-title {
        font-size: 3em;
        color: white;
        margin-bottom: 10px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 1.3em;
        color: rgba(255,255,255,0.9);
        margin-bottom: 20px;
    }
    .feature-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 15px;
    }
    .feature-title {
        font-size: 1.2em;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .feature-desc {
        color: #666;
        font-size: 0.95em;
        line-height: 1.6;
    }
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85em;
        margin: 3px;
    }
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🌊 HydroTune-AI</div>
        <div class="hero-subtitle">流域水文模型智能率定系统</div>
        <div style="margin-top: 20px;">
            <span class="tech-badge">🤖 LLM 智能分析</span>
            <span class="tech-badge">📊 多模型融合</span>
            <span class="tech-badge">🔬 频率分析</span>
            <span class="tech-badge">📈 深度学习建议</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # AI Features Section
    st.subheader("🚀 AI 智能核心能力")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">智能数据认知</div>
            <div class="feature-desc">
                <b>LLM 驱动的数据理解</b><br><br>
                • 自动识别数据时间尺度（小时/日）<br>
                • 智能分析数据质量与一致性<br>
                • 识别异常值与数据缺失<br>
                • 理解水文过程的物理意义
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">智能场次识别</div>
            <div class="feature-desc">
                <b>自动化洪水事件分析</b><br><br>
                • 基于斜率变化识别洪水起止<br>
                • 自动计算峰型特征参数<br>
                • 多准则代表性洪水选取<br>
                • 峰现时间比精准分析
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">智能报告生成</div>
            <div class="feature-desc">
                <b>专业级分析报告</b><br><br>
                • Pearson III 频率曲线拟合<br>
                • 设计洪水智能计算<br>
                • 多模型对比分析<br>
                • 深度学习改进建议
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Models & Workflow
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 支持的水文模型")
        st.markdown("""
        | 模型 | 特点 | 适用场景 |
        |------|------|----------|
        | **水箱模型** | 简化双层调蓄 | 通用流域 |
        | **HBV模型** | 概念性水文 | 湿润半湿润 |
        | **新安江模型** | 三水源产流 | 中国湿润区 |
        """)
    
    with col2:
        st.subheader("📖 使用流程")
        st.markdown("""
        <div style="line-height: 2.2;">
            <span class="step-number">1</span>上传水文数据文件<br>
            <span class="step-number">2</span>配置列名映射关系<br>
            <span class="step-number">3</span>设置流域面积等参数<br>
            <span class="step-number">4</span>启动智能分析流程<br>
            <span class="step-number">5</span>查看结果导出报告
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Data Format
    st.subheader("📋 数据格式要求")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **必需列：**
        - 时间列：`date`, `time`, `日期`, `时间`
        - 降水列：`precip`, `rainfall`, `p`, `降水`, `降雨`
        - 流量列：`flow`, `discharge`, `q`, `流量`, `径流`
        
        **可选列：**
        - 蒸发列：`evap`, `et`, `蒸发`
        """)
    
    with col2:
        st.markdown("""
        **支持格式：**
        - CSV 文件
        - Excel 文件 (.xlsx, .xls)
        
        **数据要求：**
        - 数值型数据
        - 支持缺失值处理
        - 自动时间尺度检测
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 请在左侧上传水文数据文件开始使用 HydroTune-AI")

# ============================================================
# 主流程
# ============================================================
if uploaded_files and len(uploaded_files) > 0:

    all_results = {}
    all_flood_events = []
    report_sections = []
    
    # ---- 读取并缓存所有文件数据 ----
    file_dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            file_dfs.append((uploaded_file.name, df))
        except Exception as e:
            st.error(f"读取 {uploaded_file.name} 失败: {e}")
            continue
    
    if not file_dfs:
        st.error("没有可用的文件")
        st.stop()
    
    # ---- 列名确认（第一个文件）----
    st.divider()
    st.subheader("📋 配置确认")
    
    first_file_name, first_df = file_dfs[0]
    st.write(f"**{first_file_name}** 的列名: `{list(first_df.columns)}`")
    
    col_mapping_accepted = st.checkbox(
        f"使用列名映射: {column_mapping['date']}→date, {column_mapping['precip']}→precip, {column_mapping['evap']}→evap, {column_mapping['flow']}→flow",
        value=True
    )
    
    # ============================================================
    # 📥 数据文件处理（仅在按钮点击后执行）
    # ============================================================
    if st.button("🚀 开始分析", type="primary", disabled=not col_mapping_accepted):
        
        # ---- 处理每个文件 ----
        all_precip = []
        all_evap = []
        all_flow = []
        all_dates = []
        all_file_events = []
        detected_timestep = 'daily'
        user_timestep = 'daily'
        file_summary = []
        
        with st.expander("📥 数据文件处理详情", expanded=False):
            for file_idx, (file_name, raw_df) in enumerate(file_dfs):
                st.write(f"**📄 {file_name}** - {raw_df.shape[0]} 行")
                
                rename_map = {}
                for std_name, orig_name in column_mapping.items():
                    if orig_name and orig_name in raw_df.columns:
                        rename_map[orig_name] = std_name
                
                if rename_map:
                    raw_df = raw_df.rename(columns=rename_map)
                
                if 'precip' not in raw_df.columns or 'flow' not in raw_df.columns:
                    st.error(f"缺少必要列。当前列: {list(raw_df.columns)}")
                    continue
                
                if 'evap' not in raw_df.columns:
                    raw_df['evap'] = 0.0
                
                if 'date' not in raw_df.columns:
                    raw_df['date'] = range(len(raw_df))
                
                clean_df = raw_df.fillna(0)
                
                if file_idx == 0:
                    detected_timestep = infer_timestep(clean_df['date'])
                
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
                
                file_summary.append({
                    'file': file_name,
                    'rows': raw_df.shape[0],
                    'events': len(flood_events)
                })
                
                for event in flood_events:
                    if hasattr(event.start_date, 'strftime'):
                        event_date_str = event.start_date.strftime('%Y%m%d')
                    else:
                        event_date_str = str(event.start_date)[:10].replace('-', '')
                    
                    all_file_events.append({
                        'file_name': file_name,
                        'event_name': event_date_str,
                        'start_date': event.start_date,
                        'end_date': event.end_date,
                        'precip': event.precip,
                        'evap': event.evap,
                        'observed_flow': event.observed_flow,
                    })
                
                all_precip.extend(precip_arr.tolist())
                all_evap.extend(evap_arr.tolist())
                all_flow.extend(flow_arr.tolist())
                
                st.write(f"  ✅ 识别到 **{len(flood_events)}** 场洪水")
        
        # 文件汇总信息
        total_events = sum(f['events'] for f in file_summary)
        total_rows = sum(f['rows'] for f in file_summary)
        st.success(f"📊 已处理 {len(file_summary)} 个文件，共 {total_rows} 行数据，识别 {total_events} 场洪水")
        
        # 时间尺度确认
        st.divider()
        st.subheader("⏱️ 时间尺度确认")
        
        first_file_name, first_raw_df = file_dfs[0]
        rename_map_first = {}
        for std_name, orig_name in column_mapping.items():
            if orig_name and orig_name in first_raw_df.columns:
                rename_map_first[orig_name] = std_name
        first_clean_df = first_raw_df.rename(columns=rename_map_first)
        if 'date' not in first_clean_df.columns:
            first_clean_df['date'] = range(len(first_raw_df))
        first_clean_df = first_clean_df.fillna(0)
        
        with st.spinner("🤖 AI Agent 正在分析数据时间尺度..."):
            detected_timestep = infer_timestep_by_llm(first_clean_df['date'], call_minimax)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"🤖 AI Agent 检测到 **小时尺度**" if detected_timestep == 'hourly' else f"🤖 AI Agent 检测到 **日尺度**")
        with col2:
            user_timestep = st.radio(
                "请选择",
                options=['hourly', 'daily'],
                index=0 if detected_timestep == 'hourly' else 1,
                horizontal=True,
                label_visibility="collapsed"
            )
        
        # ============================================================
        # 🤖 AI Agent 数据预分析
        # ============================================================
        st.divider()
        st.subheader("🧠 AI Agent 数据预分析")
        
        all_precip_arr = np.array(all_precip)
        all_evap_arr = np.array(all_evap)
        all_flow_arr = np.array(all_flow)
        
        # 为预分析准备日期（灵活解析）
        try:
            dates_for_analysis = pd.to_datetime(first_clean_df['date'], format='mixed')
        except:
            dates_for_analysis = pd.to_datetime(first_clean_df['date'], errors='coerce')
        if dates_for_analysis is None or dates_for_analysis.isna().all():
            dates_for_analysis = pd.date_range(start='2020-01-01', periods=len(all_precip), freq='D')
        elif len(dates_for_analysis) < len(all_precip):
            dates_for_analysis = pd.date_range(
                start=dates_for_analysis.iloc[0] if not pd.isna(dates_for_analysis.iloc[0]) else '2020-01-01',
                periods=len(all_precip),
                freq='H' if user_timestep == 'hourly' else 'D'
            )
        
        # 确保dates_for_analysis是有效的Series
        if len(dates_for_analysis) == 0:
            dates_for_analysis = pd.Series(pd.date_range(start='2020-01-01', periods=len(all_precip_arr)))
        
        dates_series = pd.Series(dates_for_analysis).reset_index(drop=True)
        
        preanalyzer = DataPreAnalyzer(area=catchment_area)
        preanalyzer.timestep = user_timestep
        quality = preanalyzer.evaluate_quality(all_precip_arr, all_flow_arr, dates_series)
        
        # ============================================================
        # 根据上传模式进行不同的预分析
        # ============================================================
        
        if upload_mode == "单文件（一场洪水）":
            # 模式1：整场洪水分析（不识别子场次）
            peak_idx = np.argmax(all_flow_arr)
            peak_flow = all_flow_arr[peak_idx]
            baseflow = np.percentile(all_flow_arr, 10)
            
            rise_start = max(0, peak_idx - 20)
            rise_flows = all_flow_arr[rise_start:peak_idx+1] if peak_idx > 0 else np.array([peak_flow])
            rise_rate = (peak_flow - rise_flows[0]) / len(rise_flows) if len(rise_flows) > 1 else 0
            
            fall_end = min(len(all_flow_arr), peak_idx + 30)
            fall_flows = all_flow_arr[peak_idx:fall_end]
            recession_rate = (fall_flows[0] - fall_flows[-1]) / len(fall_flows) if len(fall_flows) > 1 else 0
            
            if rise_rate > 0 and recession_rate > 0:
                peak_ratio = rise_rate / recession_rate if recession_rate > 0 else 10
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
            
            flood_volume = np.sum(all_flow_arr - baseflow) * (24 if user_timestep == 'daily' else 1) / 10000
            
            preanalysis_result = type('obj', (object,), {
                'quality': quality,
                'events': [type('obj', (object,), {
                    'event_id': 'F001',
                    'peak_flow': peak_flow,
                    'flood_volume': flood_volume,
                    'rise_rate': rise_rate,
                    'recession_rate': recession_rate,
                    'peak_type': peak_type,
                    'start_idx': 0,
                    'end_idx': len(all_flow_arr) - 1,
                    'start_date': dates_series.iloc[0] if len(dates_series) > 0 else pd.Timestamp.now(),
                    'end_date': dates_series.iloc[-1] if len(dates_series) > 0 else pd.Timestamp.now(),
                    'duration': len(all_flow_arr)
                })()],
                'selected_events': [],
                'frequency': type('obj', (object,), {
                    'n_samples': 1,
                    'mean': peak_flow,
                    'std': 0,
                    'cv': 0,
                    'cs': 0,
                    'design_values': {}
                })(),
                'area': catchment_area,
                'timestep': user_timestep
            })()
            
            st.success(f"✅ AI Agent 完成数据分析：整场洪水，峰型{peak_type}")
            
            with st.expander("📈 数据分析详情", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("数据完整率", f"{quality.completeness:.1f}%")
                col2.metric("时间连续性", f"{quality.continuity:.1f}%")
                col3.metric("降水-径流相关", f"{quality.correlation:.3f}")
                col4.metric("质量等级", quality.quality_level)
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("洪峰流量", f"{peak_flow:.1f} m³/s")
                col2.metric("洪水总量", f"{flood_volume:.1f} mm")
                col3.metric("峰型", peak_type)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("涨洪速率", f"{rise_rate:.3f}")
                col2.metric("落洪速率", f"{recession_rate:.3f}")
                col3.metric("基流", f"{baseflow:.1f} m³/s")
        
        elif upload_mode == "单文件（连续序列）":
            # 模式2：连续序列识别多场洪水（基于斜率变化）
            with st.spinner("🤖 AI Agent 正在进行洪水事件识别..."):
                detected_events = preanalyzer.detect_flood_events_by_slope(
                    all_flow_arr, 
                    dates_series,
                    precip_threshold=float(np.mean(all_precip_arr) * 2),
                    flow_threshold=float(np.percentile(all_flow_arr, 70))
                )
                
                if not detected_events:
                    st.warning("⚠️ 未识别到洪水场次，使用默认参数重新识别...")
                    detected_events = preanalyzer.detect_flood_events_by_slope(
                        all_flow_arr,
                        dates_series,
                        precip_threshold=0.5,
                        flow_threshold=float(np.percentile(all_flow_arr, 60))
                    )
            
            if detected_events:
                st.success(f"✅ AI Agent 完成洪水识别：共识别 {len(detected_events)} 场洪水")
                
                # 计算频率分析
                from scipy import stats
                peaks = np.array([e['peak_flow'] for e in detected_events])
                if len(peaks) >= 3:
                    freq_result = preanalyzer.frequency_analysis_pearson(peaks)
                else:
                    freq_result = type('obj', (object,), {
                        'n_samples': len(peaks),
                        'mean': np.mean(peaks) if len(peaks) > 0 else 0,
                        'std': np.std(peaks) if len(peaks) > 0 else 0,
                        'cv': np.std(peaks)/np.mean(peaks) if np.mean(peaks) > 0 else 0,
                        'cs': 0,
                        'design_values': {}
                    })()
                
                # 选取代表性洪水（取峰值最大的3-5场）
                n_select = min(5, max(3, len(detected_events) // 2))
                selected = sorted(detected_events, key=lambda x: x['peak_flow'], reverse=True)[:n_select]
                selected_events = []
                for i, evt in enumerate(selected):
                    evt_obj = type('obj', (object,), {
                        'event_id': f'F{i+1:03d}',
                        'peak_flow': evt['peak_flow'],
                        'flood_volume': evt.get('flood_volume', 0),
                        'rise_rate': evt.get('rise_rate', 0),
                        'recession_rate': evt.get('recession_rate', 0),
                        'peak_type': evt.get('peak_type', '未知'),
                        'start_idx': evt['start_idx'],
                        'end_idx': evt['end_idx'],
                        'start_date': evt['start_date'],
                        'end_date': evt['end_date'],
                        'duration': evt['end_idx'] - evt['start_idx'],
                        'selection_reason': f'洪峰流量第{i+1}高: {evt["peak_flow"]:.1f} m³/s'
                    })()
                    selected_events.append(evt_obj)
                
                # 构建事件列表
                events = []
                for i, evt in enumerate(detected_events):
                    evt_obj = type('obj', (object,), {
                        'event_id': f'F{i+1:03d}',
                        'peak_flow': evt['peak_flow'],
                        'flood_volume': evt.get('flood_volume', 0),
                        'rise_rate': evt.get('rise_rate', 0),
                        'recession_rate': evt.get('recession_rate', 0),
                        'peak_type': evt.get('peak_type', '未知'),
                        'start_idx': evt['start_idx'],
                        'end_idx': evt['end_idx'],
                        'start_date': evt['start_date'],
                        'end_date': evt['end_date'],
                        'duration': evt['end_idx'] - evt['start_idx']
                    })()
                    events.append(evt_obj)
                
                preanalysis_result = type('obj', (object,), {
                    'quality': quality,
                    'events': events,
                    'selected_events': selected_events,
                    'frequency': freq_result,
                    'area': catchment_area,
                    'timestep': user_timestep
                })()
                
                # 显示分析详情
                with st.expander("📈 洪水识别与频率分析详情", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("数据完整率", f"{quality.completeness:.1f}%")
                    col2.metric("时间连续性", f"{quality.continuity:.1f}%")
                    col3.metric("降水-径流相关", f"{quality.correlation:.3f}")
                    col4.metric("质量等级", quality.quality_level)
                    
                    st.divider()
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write("**🤖 AI 识别洪水场次**:")
                        events_data = []
                        for e in preanalysis_result.events:
                            start_str = e.start_date.strftime('%Y-%m-%d') if hasattr(e.start_date, 'strftime') else str(e.start_date)[:10]
                            end_str = e.end_date.strftime('%Y-%m-%d') if hasattr(e.end_date, 'strftime') else str(e.end_date)[:10]
                            events_data.append({
                                '场次': e.event_id,
                                '开始': start_str,
                                '结束': end_str,
                                '峰值(m³/s)': f"{e.peak_flow:.1f}",
                                '历时': e.duration
                            })
                        if events_data:
                            st.dataframe(pd.DataFrame(events_data), use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.write("**代表性洪水选取**:")
                        selected_data = []
                        for e in preanalysis_result.selected_events:
                            selected_data.append({
                                '场次': e.event_id,
                                '峰值(m³/s)': f"{e.peak_flow:.1f}",
                                '选取原因': e.selection_reason[:30] + '...' if len(e.selection_reason) > 30 else e.selection_reason
                            })
                        if selected_data:
                            st.dataframe(pd.DataFrame(selected_data), use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write("**频率分析参数**:")
                        freq = preanalysis_result.frequency
                        st.write(f"- 样本数: {freq.n_samples} 场")
                        st.write(f"- 均值: {freq.mean:.2f} m³/s")
                        st.write(f"- Cv: {freq.cv:.3f}")
                        st.write(f"- Cs: {freq.cs:.3f}")
                        
                        st.write("**设计洪水成果**:")
                        for rp, val in freq.design_values.items():
                            st.write(f"- {rp}: {val:.2f} m³/s")
                    
                    with col2:
                        st.write("**洪水过程线**")
                        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
                        xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
                        ax.plot(all_flow_arr, 'b-', linewidth=1, label='流量过程')
                        ax.fill_between(range(len(all_flow_arr)), 0, all_flow_arr, alpha=0.3)
                        
                        # 标记识别的洪水
                        cmap = plt.cm.get_cmap('tab10')
                        for i, evt in enumerate(selected_events):
                            color = cmap(i % 10)
                            ax.axvspan(evt.start_idx, evt.end_idx, alpha=0.2, color=color, label=f"{evt.event_id}: {evt.peak_flow:.1f} $m^3/s$")
                            ax.axvline(evt.start_idx, color=color, linestyle='--', alpha=0.5)
                            ax.axvline(evt.end_idx, color=color, linestyle='--', alpha=0.5)
                        
                        ax.set_xlabel(xlabel_text, fontsize=12)
                        ax.set_ylabel(r'流量 ($m^3/s$)', fontsize=12)
                        ax.set_title('连续序列洪水识别结果', fontsize=14)
                        ax.legend(fontsize=8, loc='upper right')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
            else:
                st.warning("⚠️ 未能识别洪水场次，切换为单场洪水模式")
                upload_mode = "单文件（一场洪水）"
                st.rerun()
        
        else:
            # 模式3：多文件模式，每文件一场洪水
            peak_idx = np.argmax(all_flow_arr)
            peak_flow = all_flow_arr[peak_idx]
            baseflow = np.percentile(all_flow_arr, 10)
            
            rise_start = max(0, peak_idx - 20)
            rise_flows = all_flow_arr[rise_start:peak_idx+1] if peak_idx > 0 else np.array([peak_flow])
            rise_rate = (peak_flow - rise_flows[0]) / len(rise_flows) if len(rise_flows) > 1 else 0
            
            fall_end = min(len(all_flow_arr), peak_idx + 30)
            fall_flows = all_flow_arr[peak_idx:fall_end]
            recession_rate = (fall_flows[0] - fall_flows[-1]) / len(fall_flows) if len(fall_flows) > 1 else 0
            
            if rise_rate > 0 and recession_rate > 0:
                peak_ratio = rise_rate / recession_rate if recession_rate > 0 else 10
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
            
            flood_volume = np.sum(all_flow_arr - baseflow) * (24 if user_timestep == 'daily' else 1) / 10000
            
            multi_events = []
            for i in range(len(file_dfs)):
                multi_events.append(type('obj', (object,), {
                    'event_id': f'F{i+1:03d}',
                    'peak_flow': peak_flow,
                    'flood_volume': flood_volume,
                    'rise_rate': rise_rate,
                    'recession_rate': recession_rate,
                    'peak_type': peak_type,
                    'start_idx': 0,
                    'end_idx': len(all_flow_arr) - 1,
                    'start_date': dates_series.iloc[0] if len(dates_series) > 0 else pd.Timestamp.now(),
                    'end_date': dates_series.iloc[-1] if len(dates_series) > 0 else pd.Timestamp.now(),
                    'duration': len(all_flow_arr)
                })())
            
            preanalysis_result = type('obj', (object,), {
                'quality': quality,
                'events': multi_events,
                'selected_events': [],
                'frequency': type('obj', (object,), {
                    'n_samples': len(file_dfs),
                    'mean': peak_flow,
                    'std': 0,
                    'cv': 0,
                    'cs': 0,
                    'design_values': {}
                })(),
                'area': catchment_area,
                'timestep': user_timestep
            })()
            
            st.success(f"✅ AI Agent 完成数据分析：{len(file_dfs)} 场洪水")
            
            with st.expander("📈 数据分析详情", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("数据完整率", f"{quality.completeness:.1f}%")
                col2.metric("时间连续性", f"{quality.continuity:.1f}%")
                col3.metric("降水-径流相关", f"{quality.correlation:.3f}")
                col4.metric("质量等级", quality.quality_level)
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("文件数量", f"{len(file_dfs)} 个")
                col2.metric("洪峰流量", f"{peak_flow:.1f} m³/s")
                col3.metric("洪水总量", f"{flood_volume:.1f} mm")
        
        st.session_state['preanalysis_result'] = preanalysis_result
        
        # ============================================================
        # 🤖 AI Agent 模型率定
        # ============================================================
        st.divider()
        st.subheader("🧠 AI Agent 模型率定")
        
        calibration_results = {}
        
        def calibrate_model(model_name, precip, evap, flow):
            try:
                spatial_data = {'area': catchment_area}
                return calibrate_model_fast(
                    model_name,
                    precip,
                    evap,
                    flow,
                    max_iter=max_iter,
                    spatial_data=spatial_data,
                    timestep=user_timestep
                )
            except Exception as e:
                st.error(f"  ⚠️ {model_name} 率定异常: {type(e).__name__}: {str(e)}")
                import traceback
                st.code(traceback.format_exc()[-600:])
                return None
        
        # 根据模式选择率定策略
        if upload_mode == "多文件（每文件一场洪水）":
            # 多文件模式：选择1/3场次率定，2/3场次验证
            st.info(f"📊 多文件模式：{len(file_dfs)} 个文件，将分为率定和验证两组")
            
            # 1. 准备所有文件数据
            file_data_list = []
            for file_idx, (file_name, raw_df) in enumerate(file_dfs):
                df = raw_df.copy()
                for std_name, orig_name in column_mapping.items():
                    if orig_name and orig_name in df.columns:
                        df = df.rename(columns={orig_name: std_name})
                
                precip_arr = np.array(df['precip'].values) if 'precip' in df.columns else np.zeros(len(df))
                flow_arr = np.array(df['flow'].values) if 'flow' in df.columns else np.zeros(len(df))
                evap_arr = np.array(df['evap'].values) if 'evap' in df.columns else np.zeros(len(df))
                
                file_data_list.append({
                    'file_name': file_name,
                    'precip': precip_arr,
                    'evap': evap_arr,
                    'flow': flow_arr,
                    'n_timesteps': len(precip_arr)
                })
            
            # 2. 随机选择1/3作为率定场次，2/3作为验证场次
            np.random.seed(42)
            n_files = len(file_data_list)
            n_calib = max(1, n_files // 3)  # 至少1个
            indices = np.random.permutation(n_files)
            calib_indices = indices[:n_calib]
            valid_indices = indices[n_calib:]
            
            calib_files = [file_data_list[i] for i in calib_indices]
            valid_files = [file_data_list[i] for i in valid_indices]
            
            st.success(f"📊 分组完成：{n_calib} 场率定 + {len(valid_files)} 场验证")
            st.write(f"**率定场次**: {[f['file_name'] for f in calib_files]}")
            st.write(f"**验证场次**: {[f['file_name'] for f in valid_files]}")
            
            # 3. 拼接率定场次数据
            calib_precip_list = []
            calib_evap_list = []
            calib_flow_list = []
            for fd in calib_files:
                calib_precip_list.extend(fd['precip'].tolist())
                calib_evap_list.extend(fd['evap'].tolist())
                calib_flow_list.extend(fd['flow'].tolist())
            
            calib_precip = np.array(calib_precip_list)
            calib_evap = np.array(calib_evap_list)
            calib_flow = np.array(calib_flow_list)
            
            st.info(f"📊 率定数据：{len(calib_precip)} 个时间步")
            
            # 4. 率定模型
            import traceback
            with st.spinner("🤖 AI Agent 正在率定模型 (率定场次)..."):
                for model_name in RECOMMENDED_MODELS:
                    if model_name in SKIP_MODELS:
                        st.write(f"  ⏭️ 跳过 {model_name}")
                        continue
                    st.write(f"  🔄 开始率定 {model_name}...")
                    result = calibrate_model(model_name, calib_precip, calib_evap, calib_flow)
                    if result:
                        params, nse, simulated = result
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(calib_flow, simulated),
                            "mae": calc_mae(calib_flow, simulated),
                            "pbias": calc_pbias(calib_flow, simulated),
                            "simulated": simulated,
                            "calib_data": (calib_precip, calib_evap, calib_flow),
                        }
                        st.write(f"  ✅ {model_name}: 率定期NSE={nse:.4f}")
                    else:
                        st.write(f"  ❌ {model_name}: 率定返回None")
            
            # 5. 用率定参数分别跑所有场次（率定+验证）
            file_simulation_results = {}
            calib_file_names = set([f['file_name'] for f in calib_files])
            
            default_xaj_params = {
                'k': 0.8, 'b': 0.3, 'im': 0.01,
                'um': 20.0, 'lm': 70.0, 'dm': 60.0, 'c': 0.15,
                'sm': 20.0, 'ex': 1.5, 'ki': 0.3, 'kg': 0.4,
                'cs': 0.8, 'l': 1, 'ci': 0.8, 'cg': 0.98,
            }
            
            for model_name, calib_result in calibration_results.items():
                params = calib_result['params']
                file_simulation_results[model_name] = {}
                
                for file_data in file_data_list:
                    file_name = file_data['file_name']
                    is_calib = file_name in calib_file_names
                    
                    spatial_data = {'area': catchment_area, 'timestep': user_timestep}
                    model = ModelRegistry.get_model(model_name)
                    
                    # 对XAJ模型使用安全的参数
                    if model_name == '新安江模型':
                        safe_params = default_xaj_params.copy()
                        # 尝试使用率定的部分参数
                        safe_params['k'] = params.get('k', 0.8)
                        safe_params['b'] = params.get('b', 0.3)
                        safe_params['im'] = params.get('im', 0.01)
                        safe_params['um'] = params.get('um', 20.0)
                        safe_params['lm'] = params.get('lm', 70.0)
                        safe_params['dm'] = params.get('dm', 60.0)
                        safe_params['c'] = params.get('c', 0.15)
                        safe_params['ex'] = params.get('ex', 1.5)
                        # 确保 ki + kg < 0.9
                        ki = min(params.get('ki', 0.3), 0.45)
                        kg = min(params.get('kg', 0.3), 0.45)
                        if ki + kg >= 0.9:
                            ki = 0.3
                            kg = 0.3
                        safe_params['ki'] = ki
                        safe_params['kg'] = kg
                        safe_params['sm'] = max(params.get('sm', 20.0), 5.0)
                        safe_params['cs'] = params.get('cs', 0.8)
                        safe_params['l'] = params.get('l', 1)
                        safe_params['ci'] = params.get('ci', 0.8)
                        safe_params['cg'] = params.get('cg', 0.98)
                    else:
                        safe_params = params.copy()
                    
                    try:
                        simulated = model.run(
                            file_data['precip'],
                            file_data['evap'],
                            safe_params,
                            spatial_data
                        )
                        file_simulation_results[model_name][file_name] = {
                            "model_name": model_name,
                            "params": safe_params,
                            "nse": calc_nse(file_data['flow'], simulated),
                            "rmse": calc_rmse(file_data['flow'], simulated),
                            "mae": calc_mae(file_data['flow'], simulated),
                            "pbias": calc_pbias(file_data['flow'], simulated),
                            "simulated": simulated,
                            "observed": file_data['flow'],
                            "precip": file_data['precip'],
                            "is_calib": is_calib,
                        }
                    except Exception as e:
                        # 如果失败，使用默认参数
                        try:
                            if model_name == '新安江模型':
                                simulated = model.run(
                                    file_data['precip'],
                                    file_data['evap'],
                                    default_xaj_params,
                                    spatial_data
                                )
                                file_simulation_results[model_name][file_name] = {
                                    "model_name": model_name,
                                    "params": default_xaj_params,
                                    "nse": calc_nse(file_data['flow'], simulated),
                                    "rmse": calc_rmse(file_data['flow'], simulated),
                                    "mae": calc_mae(file_data['flow'], simulated),
                                    "pbias": calc_pbias(file_data['flow'], simulated),
                                    "simulated": simulated,
                                    "observed": file_data['flow'],
                                    "precip": file_data['precip'],
                                    "is_calib": is_calib,
                                }
                                st.warning(f"  ⚠️ {file_name}/{model_name}: 使用默认参数")
                            else:
                                file_simulation_results[model_name][file_name] = {
                                    "model_name": model_name,
                                    "params": safe_params,
                                    "nse": -999,
                                    "rmse": -999,
                                    "mae": -999,
                                    "pbias": -999,
                                    "simulated": np.zeros_like(file_data['flow']),
                                    "observed": file_data['flow'],
                                    "precip": file_data['precip'],
                                    "is_calib": is_calib,
                                }
                                st.error(f"  ❌ {file_name}/{model_name}: {str(e)}")
                        except Exception as e2:
                            import traceback
                            st.error(f"  ❌ {file_name}/{model_name}: 完全失败")
                            st.code(f"数据长度: {len(file_data['precip'])}, 范围: [{file_data['precip'].min():.2f}, {file_data['precip'].max():.2f}]")
                            st.code(f"流量范围: [{file_data['flow'].min():.2f}, {file_data['flow'].max():.2f}]")
                            st.code(traceback.format_exc()[-800:])
            
            st.success(f"✅ 多文件模式率定完成：{len(calibration_results)} 个模型 × {len(file_data_list)} 场洪水")
            
            # ============================================================
            # 📊 绘图：每个文件一张图
            # ============================================================
            summary_data = []
            all_results = {}
            
            model_colors = {
                '新安江模型': '#e74c3c',
                '水箱模型': '#3498db',
                'HBV模型': '#2ecc71',
            }
            
            n_files = len(file_dfs)
            n_cols = min(2, n_files)
            n_rows = (n_files + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
            if n_files == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 else axes
            
            for idx, (file_name, raw_df) in enumerate(file_dfs):
                ax = axes[idx]
                
                first_model = list(file_simulation_results.keys())[0] if file_simulation_results else None
                if first_model and file_name in file_simulation_results[first_model]:
                    precip_arr = file_simulation_results[first_model][file_name].get('precip', np.zeros(len(raw_df)))
                    flow_arr = file_simulation_results[first_model][file_name]['observed']
                    is_calib = file_simulation_results[first_model][file_name].get('is_calib', False)
                else:
                    df = raw_df.copy()
                    for std_name, orig_name in column_mapping.items():
                        if orig_name and orig_name in df.columns:
                            df = df.rename(columns={orig_name: std_name})
                    precip_arr = np.array(df['precip'].values) if 'precip' in df.columns else np.zeros(len(df))
                    flow_arr = np.array(df['flow'].values) if 'flow' in df.columns else np.zeros(len(df))
                    is_calib = False
                
                event_type = "率定" if is_calib else "验证"
                ax.plot(flow_arr, "k-", label="实测", linewidth=2.5)
                
                ax2 = ax.twinx()
                ax2.bar(range(len(precip_arr)), precip_arr, color='#87CEEB', alpha=0.5, width=1, label='降水')
                ax2.set_ylabel("降水 (mm)", fontsize=14, color='black')
                ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
                ax2.invert_yaxis()
                
                xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
                
                for model_name in file_simulation_results:
                    if file_name in file_simulation_results[model_name]:
                        result = file_simulation_results[model_name][file_name]
                        color = model_colors.get(model_name, '#999999')
                        label = f"{model_name} (NSE={result['nse']:.3f})"
                        ax.plot(result['simulated'], color=color, label=label, linewidth=2, alpha=0.8)
                        
                        summary_data.append({
                            "文件": file_name,
                            "场次": "整场洪水",
                            "类型": event_type,
                            "模型": model_name,
                            "NSE": result['nse'],
                            "RMSE": result['rmse'],
                            "PBIAS": f"{result['pbias']:.1f}%"
                        })
                        
                        if file_name not in all_results:
                            all_results[file_name] = {}
                        if "整场洪水" not in all_results[file_name]:
                            all_results[file_name]["整场洪水"] = []
                        all_results[file_name]["整场洪水"].append(result)
                
                ax.set_title(f"{file_name} [{event_type}]", fontsize=14)
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(xlabel_text, fontsize=12)
                ax.set_ylabel(r"流量 ($m^3/s$)", fontsize=12)
            
            for idx in range(n_files, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 保存图像按钮
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("💾 保存图像"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"flow_comparison_{timestamp}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    st.success(f"✅ 图像已保存至: {filename}")
            
            # 汇总指标表
            if summary_data:
                st.divider()
                st.subheader("📋 率定与验证指标汇总")
                
                # 分别计算率定和验证的平均指标
                calib_data = [d for d in summary_data if d['类型'] == '率定']
                valid_data = [d for d in summary_data if d['类型'] == '验证']
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                if calib_data and valid_data:
                    st.divider()
                    st.subheader("📊 模型表现对比")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**率定场次平均**")
                        calib_nse = np.mean([d['NSE'] for d in calib_data])
                        calib_rmse = np.mean([d['RMSE'] for d in calib_data])
                        st.metric("率定NSE", f"{calib_nse:.4f}")
                        st.metric("率定RMSE", f"{calib_rmse:.4f}")
                    
                    with col2:
                        st.markdown("**验证场次平均**")
                        valid_nse = np.mean([d['NSE'] for d in valid_data])
                        valid_rmse = np.mean([d['RMSE'] for d in valid_data])
                        st.metric("验证NSE", f"{valid_nse:.4f}")
                        st.metric("验证RMSE", f"{valid_rmse:.4f}")
                
                # 统一参数表格
                st.divider()
                st.subheader("📋 模型率定参数")
                for model_name, calib_result in calibration_results.items():
                    with st.expander(f"📊 {model_name}"):
                        st.markdown(f"**率定NSE**: {calib_result['nse']:.4f}")
                        param_df = generate_param_table(model_name, calib_result['params'])
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
            
            # 多文件模式AI报告
            st.divider()
            st.subheader("🤖 AI Agent 智能分析报告")
            
            with st.spinner("🤖 AI Agent 正在生成分析报告..."):
                multifile_report = generate_multifile_report(
                    file_data_list,
                    calibration_results,
                    file_simulation_results,
                    call_minimax
                )
            st.markdown(multifile_report)
            
            if not multifile_report.startswith("[ERROR]"):
                report_filename = f"multifile_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                st.download_button(
                    "📥 导出报告 (Markdown)",
                    data=multifile_report,
                    file_name=report_filename,
                    mime="text/markdown"
                )
            
            st.stop()  # 多文件模式完成，停止后续代码
        
        elif upload_mode == "单文件（连续序列）":
            # 连续序列模式：使用代表性洪水进行率定
            file_calibration_results = {}
            
            if preanalysis_result.selected_events:
                calib_precip_list = []
                calib_evap_list = []
                calib_flow_list = []
                for event in preanalysis_result.selected_events:
                    start_idx = max(0, event.start_idx)
                    end_idx = min(len(all_precip_arr), event.end_idx + 1)
                    calib_precip_list.extend(all_precip_arr[start_idx:end_idx].tolist())
                    calib_evap_list.extend(all_evap_arr[start_idx:end_idx].tolist())
                    calib_flow_list.extend(all_flow_arr[start_idx:end_idx].tolist())
                calib_precip = np.array(calib_precip_list)
                calib_evap = np.array(calib_evap_list)
                calib_flow = np.array(calib_flow_list)
                st.info(f"🤖 使用 {len(preanalysis_result.selected_events)} 场代表性洪水率定，共 {len(calib_precip)} 个时间步")
            else:
                calib_precip = all_precip_arr
                calib_evap = all_evap_arr
                calib_flow = all_flow_arr
            
            with st.spinner("🤖 AI Agent 正在率定模型..."):
                for model_name in RECOMMENDED_MODELS:
                    if model_name in SKIP_MODELS:
                        continue
                    result = calibrate_model(model_name, calib_precip, calib_evap, calib_flow)
                    if result:
                        params, nse, simulated = result
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(calib_flow, simulated),
                            "mae": calc_mae(calib_flow, simulated),
                            "pbias": calc_pbias(calib_flow, simulated),
                            "simulated": simulated,
                            "calib_data": (calib_precip, calib_evap, calib_flow),
                        }
                        st.write(f"  ✅ {model_name}: NSE = {nse:.4f}")
        else:
            # 单文件一场洪水模式
            file_calibration_results = {}
            calib_precip = all_precip_arr
            calib_evap = all_evap_arr
            calib_flow = all_flow_arr
            
            st.info(f"📊 整场洪水数据，共 {len(calib_precip)} 个时间步")
            
            with st.spinner("🤖 AI Agent 正在率定模型..."):
                for model_name in RECOMMENDED_MODELS:
                    if model_name in SKIP_MODELS:
                        continue
                    result = calibrate_model(model_name, calib_precip, calib_evap, calib_flow)
                    if result:
                        params, nse, simulated = result
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(calib_flow, simulated),
                            "mae": calc_mae(calib_flow, simulated),
                            "pbias": calc_pbias(calib_flow, simulated),
                            "simulated": simulated,
                            "calib_data": (calib_precip, calib_evap, calib_flow),
                        }
                        st.write(f"  ✅ {model_name}: NSE = {nse:.4f}")
        
        # ============================================================
        # 📊 结果展示
        # ============================================================
        st.divider()
        st.subheader("📈 模型率定与验证结果")
        
        summary_data = []
        all_results = {}
        
        model_colors = {
            '新安江模型': '#e74c3c',
            '水箱模型': '#3498db',
            'HBV模型': '#2ecc71',
        }
        
        if upload_mode == "单文件（连续序列）":
            # 连续序列模式
            fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
            
            ax.plot(all_flow_arr, "k-", label="实测", linewidth=2.5)
            
            ax2 = ax.twinx()
            ax2.bar(range(len(all_precip_arr)), all_precip_arr, color='#87CEEB', alpha=0.5, width=1, label='降水')
            ax2.set_ylabel("降水 (mm)", fontsize=14, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
            ax2.invert_yaxis()
            
            xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
            
            for model_name, result in calibration_results.items():
                color = model_colors.get(model_name, '#999999')
                label = f"{model_name} (NSE={result['nse']:.3f})"
                ax.plot(result['simulated'], color=color, label=label, linewidth=2, alpha=0.8)
                
                summary_data.append({
                    "文件": "连续序列",
                    "场次": "整场洪水",
                    "类型": "率定",
                    "模型": model_name,
                    "NSE": result['nse'],
                    "RMSE": result['rmse'],
                    "PBIAS": f"{result['pbias']:.1f}%"
                })
            
            ax.set_title("连续序列洪水模拟结果", fontsize=14)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(xlabel_text, fontsize=12)
            ax.set_ylabel(r"流量 ($m^3/s$)", fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 标记识别的洪水场次
            if preanalysis_result and hasattr(preanalysis_result, 'selected_events') and preanalysis_result.selected_events:
                st.write("**🏊 识别的代表性洪水场次:**")
                for evt in preanalysis_result.selected_events:
                    st.write(f"  - {evt.event_id}: 峰值 {evt.peak_flow:.1f} m³/s ({evt.start_date.strftime('%Y-%m-%d') if hasattr(evt.start_date, 'strftime') else evt.start_date})")
        
        else:
            # 单文件一场洪水模式
            fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
            
            ax.plot(all_flow_arr, "k-", label="实测", linewidth=2.5)
            
            ax2 = ax.twinx()
            ax2.bar(range(len(all_precip_arr)), all_precip_arr, color='#87CEEB', alpha=0.5, width=1, label='降水')
            ax2.set_ylabel("降水 (mm)", fontsize=14, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
            ax2.invert_yaxis()
            
            xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
            
            for model_name, result in calibration_results.items():
                color = model_colors.get(model_name, '#999999')
                label = f"{model_name} (NSE={result['nse']:.3f})"
                ax.plot(result['simulated'], color=color, label=label, linewidth=2, alpha=0.8)
                
                summary_data.append({
                    "文件": "单场洪水",
                    "场次": "整场洪水",
                    "类型": "率定",
                    "模型": model_name,
                    "NSE": result['nse'],
                    "RMSE": result['rmse'],
                    "PBIAS": f"{result['pbias']:.1f}%"
                })
            
            ax.set_title("单场洪水模拟结果", fontsize=14)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(xlabel_text, fontsize=12)
            ax.set_ylabel(r"流量 ($m^3/s$)", fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
        
            # 保存图像按钮
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("💾 保存图像"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"flow_comparison_{timestamp}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    st.success(f"✅ 图像已保存至: {filename}")
            
            # 汇总指标表
            if summary_data:
                st.divider()
                st.subheader("📋 率定指标汇总")
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # 统一参数表格
                st.divider()
                st.subheader("📋 模型率定参数")
                
                for model_name, calib_result in calibration_results.items():
                    with st.expander(f"📊 {model_name}"):
                        st.markdown(f"**率定NSE**: {calib_result['nse']:.4f}")
                        param_df = generate_param_table(model_name, calib_result['params'])
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
            
            # 多文件模式AI报告
            st.divider()
            st.subheader("🤖 AI Agent 智能分析报告")
            
            with st.spinner("🤖 AI Agent 正在生成分析报告..."):
                multifile_report = generate_multifile_report(
                    file_data_list,
                    calibration_results,
                    file_simulation_results,
                    call_minimax
                )
            st.markdown(multifile_report)
            
            if not multifile_report.startswith("[ERROR]"):
                report_filename = f"multifile_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                st.download_button(
                    "📥 导出报告 (Markdown)",
                    data=multifile_report,
                    file_name=report_filename,
                    mime="text/markdown"
                )
            
            st.stop()  # 多文件模式完成

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

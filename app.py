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
    calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias, calc_kge,
    get_model_param_info, generate_param_table, muskingum_routing
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
from src.bma_ensemble import (
    calc_bma_weights, apply_bma_ensemble, calc_bma_metrics,
    format_weights_string
)
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

# 设置中文字体（兼容本地和云端）
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC", 
                 "Source Han Sans SC", "Droid Sans Fallback", "AR PL UMing CN",
                 "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
font_list = [f for f in chinese_fonts if f in available_fonts]
if not font_list:
    font_list = ["sans-serif"]
plt.rcParams["font.sans-serif"] = font_list
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 页面状态初始化
# ============================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# ============================================================
# 页面定义函数
# ============================================================
def show_models_page():
    """水文模型介绍页面"""
    st.markdown("""
    <style>
    .model-hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        padding: 50px 40px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        color: white;
    }
    .model-hero h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 15px;
        color: white;
    }
    .model-hero p {
        font-size: 1.2em;
        color: rgba(255,255,255,0.8);
    }
    .model-detail-card {
        background: white;
        border-radius: 16px;
        padding: 30px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    .model-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .model-icon {
        font-size: 3em;
        margin-right: 20px;
    }
    .model-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #1e293b;
    }
    .param-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    .param-table th {
        background: #f1f5f9;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        color: #475569;
        border-bottom: 2px solid #e2e8f0;
    }
    .param-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #f1f5f9;
        color: #64748b;
    }
    .param-table tr:hover {
        background: #f8fafc;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-hero">
        <h1>🌊 水文模型介绍</h1>
        <p>概念性水文模型是模拟流域水文过程的重要工具</p>
    </div>
    "    """, unsafe_allow_html=True)
    
    if st.button("← 返回主页", key="models_back_1", use_container_width=True):
        st.session_state.current_page = 'main'
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    <div class="model-detail-card">
        <div class="model-header">
            <span class="model-icon">🏗️</span>
            <span class="model-title">水箱模型 (Tank Model)</span>
        </div>
        <p style="color: #64748b; line-height: 1.8;">
            水箱模型是由日本学者菅原提出的一种概念性水文模型。该模型将流域的调蓄作用抽象为若干个串联或并联的水箱，模拟降雨-径流转换过程。模型结构简单，参数少，适用性广。
        </p>
        <h4 style="color: #1e293b; margin-top: 20px;">模型参数</h4>
        <table class="param-table">
            <tr><th>参数</th><th>含义</th><th>单位</th><th>典型范围</th></tr>
            <tr><td><b>k1</b></td><td>快速流调蓄系数</td><td>-</td><td>0.01 ~ 0.3</td></tr>
            <tr><td><b>k2</b></td><td>慢速流调蓄系数（基流）</td><td>-</td><td>0.001 ~ 0.05</td></tr>
            <tr><td><b>c</b></td><td>产流系数</td><td>-</td><td>0.01 ~ 0.3</td></tr>
        </table>
        <h4 style="color: #1e293b; margin-top: 20px;">适用场景</h4>
        <p style="color: #64748b;">通用流域模拟，尤其适用于数据较少或需要快速分析的流域。</p>
    </div>
    
    <div class="model-detail-card">
        <div class="model-header">
            <span class="model-icon">🏔️</span>
            <span class="model-title">HBV模型</span>
        </div>
        <p style="color: #64748b; line-height: 1.8;">
            HBV模型是由瑞典气象水文研究所(SMHI)开发的概念性水文模型。模型包含土壤含水量计算、蒸散发计算、径流生成和汇流四个模块，广泛应用于北欧和世界各地的流域模拟。
        </p>
        <h4 style="color: #1e293b; margin-top: 20px;">模型参数</h4>
        <table class="param-table">
            <tr><th>参数</th><th>含义</th><th>单位</th><th>典型范围</th></tr>
            <tr><td><b>fc</b></td><td>田间持水量</td><td>mm</td><td>50 ~ 500</td></tr>
            <tr><td><b>beta</b></td><td>形状参数</td><td>-</td><td>1.0 ~ 5.0</td></tr>
            <tr><td><b>k0</b></td><td>快速出流系数</td><td>-</td><td>0.01 ~ 0.5</td></tr>
            <tr><td><b>k1</b></td><td>慢速出流系数</td><td>-</td><td>0.001 ~ 0.1</td></tr>
            <tr><td><b>lp</b></td><td>蒸散发限制系数</td><td>-</td><td>0.3 ~ 1.0</td></tr>
        </table>
        <h4 style="color: #1e293b; margin-top: 20px;">适用场景</h4>
        <p style="color: #64748b;">湿润半湿润地区流域，尤其适用于北欧、北美等地区的流域。</p>
    </div>
    
    <div class="model-detail-card">
        <div class="model-header">
            <span class="model-icon">🌊</span>
            <span class="model-title">新安江模型 (XAJ)</span>
        </div>
        <p style="color: #64748b; line-height: 1.8;">
            新安江模型是我国水文学家赵人俊等在1980年代提出的三水源概念性水文模型。该模型基于蓄满产流机制，将径流划分为地表径流、壤中流和地下水径流三种水源，是我国湿润地区应用最广泛的流域水文模型。
        </p>
        <h4 style="color: #1e293b; margin-top: 20px;">模型参数</h4>
        <table class="param-table">
            <tr><th>参数</th><th>含义</th><th>单位</th><th>典型范围</th></tr>
            <tr><td><b>k</b></td><td>蒸散发系数</td><td>-</td><td>0.5 ~ 1.5</td></tr>
            <tr><td><b>b</b></td><td>蓄水容量曲线指数</td><td>-</td><td>0.1 ~ 0.5</td></tr>
            <tr><td><b>im</b></td><td>不透水面积比例</td><td>-</td><td>0.01 ~ 0.1</td></tr>
            <tr><td><b>um</b></td><td>上层土壤蓄水容量</td><td>mm</td><td>10 ~ 50</td></tr>
            <tr><td><b>lm</b></td><td>下层土壤蓄水容量</td><td>mm</td><td>50 ~ 150</td></tr>
            <tr><td><b>dm</b></td><td>深层土壤蓄水容量</td><td>mm</td><td>10 ~ 100</td></tr>
            <tr><td><b>c</b></td><td>深层蒸散发系数</td><td>-</td><td>0.01 ~ 0.2</td></tr>
            <tr><td><b>sm</b></td><td>自由水蓄水容量</td><td>mm</td><td>10 ~ 80</td></tr>
            <tr><td><b>ex</b></td><td>自由水容量曲线指数</td><td>-</td><td>1.0 ~ 2.0</td></tr>
            <tr><td><b>ki</b></td><td>壤中流出流系数</td><td>-</td><td>0.3 ~ 0.7</td></tr>
            <tr><td><b>kg</b></td><td>地下水出流系数</td><td>-</td><td>0.01 ~ 0.2</td></tr>
            <tr><td><b>cs</b></td><td>流域汇流系数</td><td>-</td><td>0.1 ~ 0.5</td></tr>
            <tr><td><b>l</b></td><td>滞后时间</td><td>h</td><td>0 ~ 24</td></tr>
            <tr><td><b>xg</b></td><td>地下水消退系数</td><td>-</td><td>0.9 ~ 0.999</td></tr>
        </table>
        <h4 style="color: #1e293b; margin-top: 20px;">适用场景</h4>
        <p style="color: #64748b;">中国湿润地区流域，特别是长江、珠江、淮河等流域。</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    if st.button("← 返回主页", key="models_back_2", use_container_width=True):
        st.session_state.current_page = 'main'
        st.rerun()

# ============================================================
# 标题区域
# ============================================================
st.title("HydroTune-AI - 流域水文模型智能率定系统")
st.caption("上传数据 → 智能清洗 → 多模型率定 → 自动报告")

# ============================================================
# 侧边栏
# ============================================================
RECOMMENDED_MODELS = ['新安江模型2']  # 暂时只保留新安江模型
SKIP_MODELS = []


def apply_upstream_routing(simulated, upstream_arr, k_routing, x_routing, warmup_steps=0):
    """将上游流量经过马斯京根演算后叠加到出口断面
    
    注意：此函数不处理预热期，由调用方在外部统一处理
    
    Args:
        simulated: 模型模拟的出口流量
        upstream_arr: 上游来水流量
        k_routing: Muskingum传播时间
        x_routing: Muskingum权重因子
        warmup_steps: 预热期步数（未使用，保留参数兼容性）
        
    Returns:
        叠加汇流后的流量序列
    """
    if upstream_arr is None:
        return simulated
    
    upstream_arr = np.asarray(upstream_arr)
    if upstream_arr.size == 0:
        return simulated
    
    # 检查长度一致性
    if len(upstream_arr) != len(simulated):
        return simulated
    
    # 不再在内部剔除预热期，由调用方统一处理
    # 再次检查长度
    if len(upstream_arr) != len(simulated) or upstream_arr.size == 0:
        return simulated
    
    # 马斯京根演算
    routed = muskingum_routing(upstream_arr, k_routing, x_routing)
    
    # 标记上游汇流已叠加
    st.caption(f"✓ 上游汇流已叠加 (k={k_routing}, x={x_routing})")
    
    return simulated + routed


with st.sidebar:
    st.markdown("### 🤖 AI Agent 状态")
    agent_status = st.empty()
    agent_status.success("🟢 智能Agent就绪")
    
    # 模型加载状态检查
    st.markdown("### 📊 模型状态")
    from src.models.registry import ModelRegistry
    all_models = ModelRegistry.list_models()
    check_models = ['新安江模型2']  # 暂时只检查新安江模型
    for model in check_models:
        if model in all_models:
            st.success(f"✅ {model}")
        else:
            st.error(f"❌ {model}")
    
    st.divider()
    st.markdown("### 📂 导航")
    if st.button("🏠 主页", use_container_width=True):
        st.session_state.current_page = 'main'
    if st.button("📚 水文模型介绍", use_container_width=True):
        st.session_state.current_page = 'models'
    
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

    st.header("⏱️ 预热期设置")
    has_warmup = st.radio(
        "预热期",
        options=["无", "有"],
        index=0,
        help="预热期是指模型开始响应前需要丢弃的时间段"
    )
    if has_warmup == "有":
        warmup_hours = st.number_input(
            "预热期长度(h)",
            min_value=0,
            max_value=720,
            value=24,
            step=1,
            help="需要丢弃的预热期时间长度"
        )
    else:
        warmup_hours = 0

    st.divider()
    
    # 直接导入参数模式
    st.header("📥 直接导入参数")
    use_imported_params = st.radio(
        "参数模式",
        options=["率定参数", "导入参数"],
        index=0,
        horizontal=True,
        help="选择'导入参数'时直接使用参数文件，不进行率定"
    )
    
    imported_params = {}
    
    if use_imported_params == "导入参数":
        st.success("📥 导入参数模式：直接使用参数文件")
        
        st.markdown("**Tank水箱模型参数：**")
        tank_param_file = st.file_uploader(
            "上传Tank模型参数文件",
            type=["csv"],
            key="tank_param_file"
        )
        if tank_param_file is not None:
            try:
                import pandas as pd
                tank_df = pd.read_csv(tank_param_file)
                tank_params = {col: float(tank_df[col].values[0]) for col in tank_df.columns if col != '模型'}
                imported_params['Tank水箱模型(完整版)'] = tank_params
                st.success(f"✅ Tank模型参数导入成功: {tank_params}")
            except Exception as e:
                st.error(f" Tank参数解析失败: {e}")
        
        st.markdown("**HBV模型参数：**")
        hbv_param_file = st.file_uploader(
            "上传HBV模型参数文件",
            type=["csv"],
            key="hbv_param_file"
        )
        if hbv_param_file is not None:
            try:
                import pandas as pd
                hbv_df = pd.read_csv(hbv_param_file)
                hbv_params = {col: float(hbv_df[col].values[0]) for col in hbv_df.columns if col != '模型'}
                imported_params['HBV模型(完整版)'] = hbv_params
                st.success(f"✅ HBV模型参数导入成功: {hbv_params}")
            except Exception as e:
                st.error(f" HBV参数解析失败: {e}")
        
        st.markdown("**新安江模型参数（含马斯京根）：**")
        xaj_param_file = st.file_uploader(
            "上传新安江模型参数文件",
            type=["csv"],
            key="xaj_param_file"
        )
        if xaj_param_file is not None:
            try:
                import pandas as pd
                import io
                
                # 读取文件内容到内存
                file_content = xaj_param_file.getvalue()
                
                xaj_df = None
                # 尝试多种编码
                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                    try:
                        xaj_df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                        break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue
                
                if xaj_df is None or xaj_df.empty:
                    st.warning("⚠️ CSV文件为空，请检查文件格式")
                elif len(xaj_df.columns) < 2:
                    st.warning(f"⚠️ CSV文件列数不足: {xaj_df.columns}")
                else:
                    xaj_params = {col: float(xaj_df[col].values[0]) for col in xaj_df.columns if col != '模型'}
                    imported_params['新安江模型2'] = xaj_params
                    st.success(f"✅ 新安江模型参数导入成功: {xaj_params}")
                    
                    # 读取马斯京根参数 - 使用 session_state 保存
                    if 'k_routing' in xaj_df.columns and 'x_routing' in xaj_df.columns:
                        st.session_state['imported_k_routing'] = float(xaj_df['k_routing'].values[0])
                        st.session_state['imported_x_routing'] = float(xaj_df['x_routing'].values[0])
                        st.success(f"✅ 马斯京根参数导入成功: k={st.session_state['imported_k_routing']}, x={st.session_state['imported_x_routing']}")
            except Exception as e:
                st.error(f" 新安江参数解析失败: {e}")

    # 列名配置
    st.header("📋 列名配置")
    with st.expander("配置数据列名映射"):
        st.write("请输入原始数据列名（映射到标准列名：date, precip, evap, flow, upstream）：")
        
        date_col = st.text_input("时间列名", value="date", key="date_col")
        precip_col = st.text_input("降水列名", value="precip", key="precip_col")
        evap_col = st.text_input("蒸发列名", value="evap", key="evap_col")
        flow_col = st.text_input("流量列名", value="flow", key="flow_col")
    
    column_mapping = {
        'date': date_col if date_col else 'date',
        'precip': precip_col if precip_col else 'precip',
        'evap': evap_col if evap_col else 'evap',
        'flow': flow_col if flow_col else 'flow',
        'upstream': "",
    }

    st.divider()

    st.header("🌊 上游出库汇流演算")
    enable_upstream_routing = st.checkbox(
        "启用上游出库汇流演算",
        value=False,
        help="启用后将使用马斯京根(Muskingum)方法将上游来水演算后叠加到出口断面流量"
    )
    
    if enable_upstream_routing:
        upstream_col = st.text_input(
            "上游出库列名",
            value="",
            help="上游断面流量列名（数据需在同一文件中）"
        )
        column_mapping['upstream'] = upstream_col if upstream_col else ""
        st.caption("✓ 上游汇流参数将在率定时自动优化")
    else:
        column_mapping['upstream'] = ""

    # 使用导入的参数或默认值
    if 'imported_k_routing' in st.session_state and 'imported_x_routing' in st.session_state:
        k_routing = st.session_state['imported_k_routing']
        x_routing = st.session_state['imported_x_routing']
    else:
        k_routing, x_routing = 2.5, 0.25

    st.divider()

    st.markdown("""
⚙️ 率定设置 
<span style="font-size: 14px; color: #888; cursor: help;" title="• 两阶段(推荐): 快速全局搜索+局部精细优化
• PSO: 粒子群优化算法，适合大规模问题
• SCE-UA: 洗牌复形进化，全局优化能力强
• DE: 差分进化，简单高效
• GA: 遗传算法，进化过程中保持多样性">❓</span>
""", unsafe_allow_html=True)

    algorithm = st.selectbox(
        "优化算法",
        options=["两阶段算法(推荐)", "PSO", "SCE-UA", "差分进化(DE)", "遗传算法(GA)"],
        index=0,
        help="选择率定使用的优化算法"
    )

    max_iter = st.slider(
        "迭代次数",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="优化算法迭代次数，次数越多结果越精确但耗时越长"
    )

    if algorithm == "PSO":
        st.subheader("粒子群优化 (PSO) 参数")
        n_particles = st.slider("粒子数", 10, 100, 20, help="粒子数量越多搜索能力越强，但计算时间增加")
        w = st.slider("惯性权重 w", 0.0, 1.0, 0.7, 0.01, help="控制粒子运动惯性，值越大搜索范围越广")
        c1 = st.slider("个体学习因子 c1", 0.0, 2.0, 1.5, 0.1, help="控制粒子向自身最优位置移动的能力")
        c2 = st.slider("群体学习因子 c2", 0.0, 2.0, 1.5, 0.1, help="控制粒子向群体最优位置移动的能力")
        algo_params = {"n_particles": n_particles, "w": w, "c1": c1, "c2": c2}
    elif algorithm == "遗传算法(GA)":
        st.subheader("遗传算法 (GA) 参数")
        pop_size = st.slider("种群大小", 10, 100, 20, help="种群数量越多遗传多样性越好")
        n_generations = st.slider("进化代数", 10, 100, 50, help="进化代数越多优化越充分")
        crossover_rate = st.slider("交叉率", 0.0, 1.0, 0.8, 0.05, help="染色体交叉产生新个体的概率")
        mutation_rate = st.slider("变异率", 0.0, 1.0, 0.1, 0.05, help="基因变异的概率，防止陷入局部最优")
        algo_params = {"pop_size": pop_size, "n_generations": n_generations, 
                       "crossover_rate": crossover_rate, "mutation_rate": mutation_rate}
    elif algorithm == "SCE-UA":
        st.subheader("SCE-UA 参数")
        n_complexes = st.slider("复形数量", 2, 10, 5, help="复形数量越多全局搜索能力越强")
        points_per_complex = st.slider("每复形点数", 5, 20, 10, help="每个复形包含的点数")
        algo_params = {"n_complexes": n_complexes, "points_per_complex": points_per_complex}
    elif algorithm == "差分进化(DE)":
        st.subheader("差分进化 (DE) 参数")
        mutation_factor = st.slider("变异因子 F", 0.0, 2.0, 0.8, 0.1, help="变异缩放因子，控制差异向量权重，推荐0.5-1.0")
        crossover_prob = st.slider("交叉概率 CR", 0.0, 1.0, 0.7, 0.1, help="交叉概率，值越大交叉频率越高")
        pop_size_de = st.slider("种群大小", 10, 100, 20, help="种群数量越多搜索能力越强")
        algo_params = {"mutation_factor": mutation_factor, "crossover_prob": crossover_prob, 
                       "pop_size": pop_size_de}
    else:
        algo_params = {}

    st.divider()

    st.header("📊 率定模型")
    for model_name in RECOMMENDED_MODELS:
        st.write(f"✅ {model_name}")
    
    st.caption("注：完整版模型暂不支持，使用简化版")


# ============================================================
# 欢迎页面
# ============================================================
if st.session_state.current_page == 'models':
    show_models_page()
elif not uploaded_files:
    st.markdown("""
    <style>
    .hero-section {
        background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #0f4c75 100%);
        padding: 50px 40px;
        border-radius: 20px;
        margin: 20px 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 50%);
        animation: pulse 8s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .hero-title {
        font-size: 4em;
        color: white;
        margin-bottom: 15px;
        font-weight: 800;
        letter-spacing: 2px;
        text-shadow: 0 0 30px rgba(59,130,246,0.5);
        position: relative;
    }
    .hero-subtitle {
        font-size: 1.5em;
        color: rgba(255,255,255,0.85);
        margin-bottom: 25px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59,130,246,0.8) 0%, rgba(99,102,241,0.8) 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 0.9em;
        margin: 5px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(59,130,246,0.15);
        border-color: rgba(59,130,246,0.3);
    }
    .feature-icon {
        font-size: 3em;
        margin-bottom: 20px;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
    }
    .feature-title {
        font-size: 1.3em;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .feature-desc {
        color: #64748b;
        font-size: 0.95em;
        line-height: 1.7;
    }
    .model-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 25px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .model-card:hover {
        transform: scale(1.03);
        border-color: rgba(59,130,246,0.5);
        box-shadow: 0 0 30px rgba(59,130,246,0.3);
    }
    .model-name {
        font-size: 1.2em;
        font-weight: 700;
        margin-bottom: 8px;
        color: #60a5fa;
    }
    .model-desc {
        font-size: 0.9em;
        color: rgba(255,255,255,0.7);
    }
    .nav-btn {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        font-size: 1em;
    }
    .nav-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(59,130,246,0.4);
    }
    .step-item {
        display: flex;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }
    .step-num {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 15px;
        flex-shrink: 0;
    }
    .step-text {
        color: #334155;
        font-size: 0.95em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">HydroTune-AI</div>
        <div class="hero-subtitle">流域水文模型智能率定系统</div>
        <div style="margin-top: 25px;">
            <span class="tech-badge">AI-Driven</span>
            <span class="tech-badge">Multi-Model</span>
            <span class="tech-badge">Auto-Calibration</span>
            <span class="tech-badge">Smart Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("📚 水文模型介绍", type="secondary", use_container_width=True):
            st.session_state.current_page = 'models'
            st.rerun()
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #1e293b; font-weight: 700;">核心能力</h2>
        <p style="color: #64748b;">AI-powered hydrological model calibration platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">智能数据认知</div>
            <div class="feature-desc">
                LLM 驱动的数据自动理解<br>
                • 时间尺度智能识别<br>
                • 数据质量评估<br>
                • 异常值检测
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">洪水事件识别</div>
            <div class="feature-desc">
                自动化洪水场次分析<br>
                • 峰型特征提取<br>
                • 代表性选取<br>
                • 频率曲线拟合
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">多模型率定</div>
            <div class="feature-desc">
                集成多种水文模型<br>
                • 参数自动优化<br>
                • 多目标评估<br>
                • 结果对比分析
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    col_models, col_workflow = st.columns([1, 1])
    
    with col_models:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #1e293b;">支持的水文模型</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div class="model-card">
                <div class="model-name">水箱模型</div>
                <div class="model-desc">通用流域</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div class="model-card">
                <div class="model-name">HBV模型</div>
                <div class="model-desc">湿润半湿润</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div class="model-card">
                <div class="model-name">新安江</div>
                <div class="model-desc">中国湿润区</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_workflow:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #1e293b;">使用流程</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-item">
            <span class="step-num">1</span>
            <span class="step-text">上传水文数据（CSV/Excel）</span>
        </div>
        <div class="step-item">
            <span class="step-num">2</span>
            <span class="step-text">配置列名映射</span>
        </div>
        <div class="step-item">
            <span class="step-num">3</span>
            <span class="step-text">设置流域参数</span>
        </div>
        <div class="step-item">
            <span class="step-num">4</span>
            <span class="step-text">启动智能分析</span>
        </div>
        <div class="step-item" style="border-bottom: none;">
            <span class="step-num">5</span>
            <span class="step-text">查看结果与报告</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("📋 数据格式要求", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **必需列：**
            - 时间：`date`, `日期`
            - 降水：`precip`, `降水`, `rainfall`
            - 流量：`flow`, `流量`, `discharge`
            
            **可选列：**
            - 蒸发：`evap`, `et`, `蒸发`
            """)
        with col2:
            st.markdown("""
            **支持格式：**
            - CSV 文件
            - Excel (.xlsx, .xls)
            - 自动识别列名映射
            - 多种时间格式兼容
            """)
    
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 30px 0; border-top: 1px solid #e2e8f0;">
        <small>HydroTune-AI v1.0 | AI-Powered Hydrological Model Calibration</small>
    </div>
    """, unsafe_allow_html=True)
    
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
        all_upstream_arr = None
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
                upstream_arr = np.array(clean_df['upstream'].values) if 'upstream' in clean_df.columns else None
                
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
                freq='h' if user_timestep == 'hourly' else 'D'
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
            if len(all_flow_arr) == 0:
                st.error("⚠️ 流量数据为空，请检查上传的数据文件")
                st.stop()
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
            if len(all_flow_arr) == 0:
                st.warning("⚠️ 流量数据为空，无法进行分析")
                st.stop()
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
        
        def calibrate_model(model_name, precip, evap, flow, upstream=None):
            try:
                spatial_data = {'area': catchment_area}
                return calibrate_model_fast(
                    model_name,
                    precip,
                    evap,
                    flow,
                    max_iter=max_iter,
                    spatial_data=spatial_data,
                    timestep=user_timestep,
                    algorithm=algorithm,
                    algo_params=algo_params,
                    upstream_flow=upstream,
                    enable_routing=enable_upstream_routing
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
                upstream_arr = np.array(df['upstream'].values) if 'upstream' in df.columns and enable_upstream_routing else None
                
                # 检查数据是否有效
                precip_sum = np.sum(precip_arr)
                flow_sum = np.sum(flow_arr)
                if precip_sum == 0:
                    st.warning(f"⚠️ {file_name}: 降水数据全为0，请检查列名配置")
                if flow_sum == 0:
                    st.warning(f"⚠️ {file_name}: 流量数据全为0，请检查列名配置")
                
                file_data_list.append({
                    'file_name': file_name,
                    'precip': precip_arr,
                    'evap': evap_arr,
                    'flow': flow_arr,
                    'upstream': upstream_arr,
                    'n_timesteps': len(precip_arr)
                })
            
            # 2. 随机选择1/3作为率定场次，2/3作为验证场次
            np.random.seed(42)
            n_files = len(file_data_list)
            n_calib = max(1, n_files * 3 // 4)  # 3:1比例，即3/4用于率定
            indices = np.random.permutation(n_files)
            calib_indices = indices[:n_calib]
            valid_indices = indices[n_calib:]
            
            calib_files = [file_data_list[i] for i in calib_indices]
            valid_files = [file_data_list[i] for i in valid_indices]
            
            st.success(f"📊 分组完成：{n_calib} 场率定 + {len(valid_files)} 场验证 (率定:验证=3:1)")
            st.write(f"**率定场次**: {[f['file_name'] for f in calib_files]}")
            st.write(f"**验证场次**: {[f['file_name'] for f in valid_files]}")
            
            # 计算预热期步数
            if warmup_hours > 0 and user_timestep == 'hourly':
                warmup_steps = warmup_hours
            elif warmup_hours > 0:
                warmup_steps = warmup_hours // 24
            else:
                warmup_steps = 0
            
            # 构建率定场次列表（每个场次独立）
            calib_events = []
            for fd in calib_files:
                event = {
                    'precip': fd['precip'],
                    'evap': fd['evap'],
                    'flow': fd['flow'],
                    'upstream': fd.get('upstream')
                }
                calib_events.append(event)
            
            avg_steps = sum(len(e['flow']) for e in calib_events) // len(calib_events) if calib_events else 0
            st.info(f"📊 率定{len(calib_events)}场，每场约 {avg_steps} 个时间步" + 
                   (f"，预热期 {warmup_steps} 步" if warmup_steps > 0 else ""))
            
            # 4. 率定模型（多场次模式）
            import traceback
            progress_bar = st.progress(0)
            
            # 判断是否使用导入的参数（侧边栏选择"导入参数"时）
            if use_imported_params == "导入参数" and imported_params:
                st.info("📥 使用导入的参数，跳过率定环节")
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        st.write(f"  ⏭️ 跳过 {model_name}")
                        continue
                    if model_name in imported_params:
                        st.write(f"  📥 使用导入参数 {model_name}...")
                        params = imported_params[model_name]
                        try:
                            model = ModelRegistry.get_model(model_name)
                            simulated = model.run(
                                calib_events[0]['precip'],
                                calib_events[0]['evap'],
                                params,
                                {'area': catchment_area, 'timestep': user_timestep}
                            )
                            # 剔除预热期数据计算指标
                            if warmup_steps > 0 and len(calib_events[0]['flow']) > warmup_steps:
                                obs_for_metric = calib_events[0]['flow'][warmup_steps:]
                                sim_for_metric = simulated[warmup_steps:]
                            else:
                                obs_for_metric = calib_events[0]['flow']
                                sim_for_metric = simulated
                            calibration_results[model_name] = {
                                "model_name": model_name,
                                "params": params,
                                "nse": calc_nse(obs_for_metric, sim_for_metric),
                                "kge": calc_kge(obs_for_metric, sim_for_metric),
                                "rmse": calc_rmse(obs_for_metric, sim_for_metric),
                                "pbias": calc_pbias(obs_for_metric, sim_for_metric),
                                "simulated": simulated,
                                "calib_data": (calib_events[0]['precip'], calib_events[0]['evap'], calib_events[0]['flow']),
                            }
                            st.write(f"  ✅ {model_name}: NSE={calibration_results[model_name]['nse']:.4f}")
                        except Exception as e:
                            st.error(f"  ⚠️ {model_name} 运行异常: {type(e).__name__}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc()[-600:])
                    else:
                        st.warning(f"  ⚠️ 未找到 {model_name} 的导入参数")
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
            else:
                st.info("🤖 正在进行模型率定...")
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        st.write(f"  ⏭️ 跳过 {model_name}")
                        continue
                    st.write(f"  🔄 开始率定 {model_name}...")
                    try:
                        spatial_data = {'area': catchment_area}
                        
                        total_models = len(RECOMMENDED_MODELS)
                        model_base = model_idx / total_models
                        
                        result = calibrate_model_fast(
                            model_name,
                            calib_events[0]['precip'],
                            calib_events[0]['evap'],
                            calib_events[0]['flow'],
                            max_iter=max_iter,
                            spatial_data=spatial_data,
                            timestep=user_timestep,
                            algorithm=algorithm,
                            algo_params=algo_params,
                            upstream_flow=calib_events[0].get('upstream'),
                            enable_routing=enable_upstream_routing,
                            calib_events=calib_events,
                            warmup_steps=warmup_steps,
                            progress_callback=lambda p: progress_bar.progress(model_base + p / total_models)
                        )
                    except Exception as e:
                        st.error(f"  ⚠️ {model_name} 率定异常: {type(e).__name__}: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc()[-600:])
                        result = None
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
                    if result:
                        params, nse, simulated = result
                        # 计算指标时需要剔除预热期数据
                        if warmup_steps > 0 and len(calib_events[0]['flow']) > warmup_steps:
                            obs_for_metric = calib_events[0]['flow'][warmup_steps:]
                            sim_for_metric = simulated[warmup_steps:]
                        else:
                            obs_for_metric = calib_events[0]['flow']
                            sim_for_metric = simulated
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": calc_nse(obs_for_metric, sim_for_metric),
                            "kge": calc_kge(obs_for_metric, sim_for_metric),
                            "rmse": calc_rmse(obs_for_metric, sim_for_metric),
                            "pbias": calc_pbias(obs_for_metric, sim_for_metric),
                            "simulated": simulated,
                            "calib_data": (calib_events[0]['precip'], calib_events[0]['evap'], calib_events[0]['flow']),
                        }
                        st.write(f"  ✅ {model_name}: 率定期平均NSE={calibration_results[model_name]['nse']:.4f}")
                    else:
                        st.write(f"  ❌ {model_name}: 率定返回None")
            
            # 5. 用率定参数分别跑所有场次（率定+验证）
            file_simulation_results = {}
            calib_file_names = set([f['file_name'] for f in calib_files])
            
            default_xaj_params = {
                'K': 0.8, 'B': 0.3, 'IM': 0.01,
                'WUM': 20.0, 'WLM': 70.0, 'WM': 150.0, 'C': 0.15,
                'SM': 20.0, 'EX': 1.5, 'KI': 0.3, 'KG': 0.4,
                'CS': 0.8, 'L': 1, 'CI': 0.8, 'CG': 0.98,
                'K_res': 4.88, 'X_res': 0.14, 'n': 5,
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
                    if model_name == '新安江模型2':
                        safe_params = params.copy()
                        # 确保 ki + kg < 0.9
                        ki = min(safe_params.get('KI', 0.3), 0.45)
                        kg = min(safe_params.get('KG', 0.3), 0.45)
                        if ki + kg >= 0.9:
                            ki = 0.3
                            kg = 0.3
                        safe_params['KI'] = ki
                        safe_params['KG'] = kg
                        safe_params['Area'] = catchment_area
                    else:
                        safe_params = params.copy()
                    
                    try:
                        simulated = model.run(
                            file_data['precip'],
                            file_data['evap'],
                            safe_params,
                            {'area': catchment_area}  # 简化spatial_data，只传area
                        )
                        st.caption(f"DEBUG: {file_name} - precip len={len(file_data['precip'])}, result len={len(simulated) if simulated is not None else 'None'}")
                        # 检查模拟结果是否有效
                        if simulated is None or len(simulated) == 0:
                            st.error(f"  ❌ {file_name}/{model_name}: 模型返回空结果")
                            raise ValueError(f"模型返回空结果: len={len(simulated) if simulated is not None else 'None'}")
                        
                        # 上游汇流叠加
                        upstream_arr = file_data.get('upstream')
                        if enable_upstream_routing and upstream_arr is not None and len(simulated) > 0:
                            simulated = apply_upstream_routing(
                                simulated, upstream_arr, k_routing, x_routing
                            )
                        # 应用预热期处理
                        if len(simulated) == 0:
                            sim_for_metric = np.zeros_like(file_data['flow'])
                            obs_for_metric = file_data['flow']
                        elif warmup_steps > 0 and len(file_data['flow']) > warmup_steps:
                            obs_for_metric = file_data['flow'][warmup_steps:]
                            sim_for_metric = simulated[warmup_steps:]
                        else:
                            obs_for_metric = file_data['flow']
                            sim_for_metric = simulated
                        file_simulation_results[model_name][file_name] = {
                            "model_name": model_name,
                            "params": safe_params,
                            "nse": calc_nse(obs_for_metric, sim_for_metric),
                            "kge": calc_kge(obs_for_metric, sim_for_metric),
                            "rmse": calc_rmse(obs_for_metric, sim_for_metric),
                            "pbias": calc_pbias(obs_for_metric, sim_for_metric),
                            "simulated": simulated,
                            "observed": file_data['flow'],
                            "precip": file_data['precip'],
                            "is_calib": is_calib,
                            "upstream_routed": enable_upstream_routing and upstream_arr is not None,
                        }
                    except Exception as e:
                        # 如果失败，使用默认参数
                        try:
                            if model_name == '新安江模型2':
                                simulated = model.run(
                                    file_data['precip'],
                                    file_data['evap'],
                                    default_xaj_params,
                                    spatial_data
                                )
                                # 上游汇流叠加
                                upstream_arr = file_data.get('upstream')
                                if enable_upstream_routing and upstream_arr is not None:
                                    simulated = apply_upstream_routing(
                                        simulated, upstream_arr, k_routing, x_routing
                                    )
                                # 检查模拟结果是否有效（全0或长度为0）
                                if len(simulated) == 0 or np.sum(np.abs(simulated)) == 0:
                                    sim_for_metric = np.zeros_like(file_data['flow'])
                                    obs_for_metric = file_data['flow']
                                elif warmup_steps > 0 and len(file_data['flow']) > warmup_steps:
                                    obs_for_metric = file_data['flow'][warmup_steps:]
                                    sim_for_metric = simulated[warmup_steps:]
                                else:
                                    obs_for_metric = file_data['flow']
                                    sim_for_metric = simulated
                                
                                if np.sum(np.abs(simulated)) == 0:
                                    file_simulation_results[model_name][file_name] = {
                                        "model_name": model_name,
                                        "params": default_xaj_params,
                                        "nse": -999,
                                        "kge": -999,
                                        "rmse": -999,
                                        "pbias": -999,
                                        "simulated": np.zeros_like(file_data['flow']),
                                        "observed": file_data['flow'],
                                        "precip": file_data['precip'],
                                        "is_calib": is_calib,
                                    }
                                    st.warning(f"  ⚠️ {file_name}/{model_name}: 模拟结果全0")
                                else:
                                    file_simulation_results[model_name][file_name] = {
                                        "model_name": model_name,
                                        "params": default_xaj_params,
                                        "nse": calc_nse(obs_for_metric, sim_for_metric),
                                        "kge": calc_kge(obs_for_metric, sim_for_metric),
                                        "rmse": calc_rmse(obs_for_metric, sim_for_metric),
                                        "pbias": calc_pbias(obs_for_metric, sim_for_metric),
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
                                    "kge": -999,
                                    "rmse": -999,
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
            
            # 计算各模型在所有率定场次的平均指标（场次平均，而非单场拼接数据的NSE）
            for model_name in calibration_results:
                calib_nse_list = []
                calib_kge_list = []
                calib_rmse_list = []
                calib_pbias_list = []
                for file_name in calib_file_names:
                    if file_name in file_simulation_results.get(model_name, {}):
                        result = file_simulation_results[model_name][file_name]
                        calib_nse_list.append(result['nse'])
                        calib_kge_list.append(result.get('kge', result['nse']))
                        calib_rmse_list.append(result['rmse'])
                        calib_pbias_list.append(result['pbias'])
                if calib_nse_list:
                    calibration_results[model_name]['nse'] = np.mean(calib_nse_list)
                    calibration_results[model_name]['kge'] = np.mean(calib_kge_list)
                    calibration_results[model_name]['rmse'] = np.mean(calib_rmse_list)
                    calibration_results[model_name]['pbias'] = np.mean(calib_pbias_list)
            
            # ============================================================
            # 📊 绘图：每个文件一张图
            # ============================================================
            summary_data = []
            all_results = {}
            
            model_colors = {
                '新安江模型2': '#e74c3c',
                'Tank水箱模型(完整版)': '#3498db',
                'HBV模型(完整版)': '#2ecc71',
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
                if first_model and file_name in file_simulation_results.get(first_model, {}):
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
                
                if precip_arr is None or len(precip_arr) == 0:
                    precip_arr = np.zeros(len(flow_arr)) if flow_arr is not None and len(flow_arr) > 0 else np.array([0.0])
                if flow_arr is None or len(flow_arr) == 0:
                    flow_arr = np.zeros(len(precip_arr))
                precip_arr = np.nan_to_num(precip_arr, nan=0.0, posinf=0.0, neginf=0.0)
                flow_arr = np.nan_to_num(flow_arr, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 剔除预热期数据
                if warmup_steps > 0 and len(flow_arr) > warmup_steps:
                    flow_arr_plot = flow_arr[warmup_steps:]
                    precip_arr_plot = precip_arr[warmup_steps:]
                else:
                    flow_arr_plot = flow_arr
                    precip_arr_plot = precip_arr
                
                event_type = "率定" if is_calib else "验证"
                ax.plot(flow_arr_plot, "k-", label="实测", linewidth=2.5)
                
                ax2 = ax.twinx()
                ax2.bar(range(len(precip_arr_plot)), precip_arr_plot, color='#87CEEB', alpha=0.5, width=1, label='降水')
                ax2.set_ylabel("降水 (mm)", fontsize=14, color='black')
                ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
                ax2.invert_yaxis()
                
                xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
                
                simulated_list = []
                model_names = []
                nse_list = []
                for model_name in file_simulation_results:
                    if file_name in file_simulation_results.get(model_name, {}):
                        result = file_simulation_results[model_name][file_name]
                        simulated = result.get('simulated')
                        if simulated is None or len(simulated) == 0:
                            st.warning(f"⚠️ {file_name}/{model_name}: 模拟结果为空，跳过绘制")
                            continue
                        simulated = np.nan_to_num(simulated, nan=0.0, posinf=0.0, neginf=0.0)
                        # 剔除预热期数据
                        if warmup_steps > 0 and len(simulated) > warmup_steps:
                            simulated_plot = simulated[warmup_steps:]
                        else:
                            simulated_plot = simulated
                        simulated_list.append(simulated_plot)
                        model_names.append(model_name)
                        nse_list.append(result['nse'])
                        
                        color = model_colors.get(model_name, '#999999')
                        calib_nse = calibration_results.get(model_name, {}).get('nse', result['nse'])
                        label = f"{model_name} (本场NSE={result['nse']:.3f})"
                        ax.plot(simulated_plot, color=color, label=label, linewidth=2, alpha=0.8)
                        
                        summary_data.append({
                            "文件": file_name,
                            "场次": "整场洪水",
                            "类型": event_type,
                            "模型": model_name,
                            "NSE": result['nse'],
                            "KGE": result.get('kge', result['nse']),
                            "RMSE": result['rmse'],
                            "PBIAS": result['pbias']
                        })
                        
                        if file_name not in all_results:
                            all_results[file_name] = {}
                        if "整场洪水" not in all_results[file_name]:
                            all_results[file_name]["整场洪水"] = []
                        all_results[file_name]["整场洪水"].append(result)
                
                if len(simulated_list) >= 2 and len(flow_arr_plot) == len(simulated_list[0]):
                    weights = calc_bma_weights(nse_list)
                    bma_result = apply_bma_ensemble(simulated_list, weights)
                    bma_nse = calc_nse(flow_arr_plot, bma_result)
                    bma_rmse = calc_rmse(flow_arr_plot, bma_result)
                    bma_pbias = calc_pbias(flow_arr_plot, bma_result)
                    weights_str = format_weights_string(model_names, weights)
                    label = f'BMA集成 (NSE={bma_nse:.3f}) [{weights_str}]'
                    ax.plot(bma_result, color='#9b59b6', linestyle='--', 
                            label=label, linewidth=2.5, alpha=0.9)
                    
                    summary_data.append({
                        "文件": file_name,
                        "场次": "整场洪水",
                        "类型": event_type,
                        "模型": "BMA集成",
                        "NSE": bma_nse,
                        "KGE": bma_nse,
                        "RMSE": bma_rmse,
                        "PBIAS": bma_pbias
                    })
                
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
                
                # 创建率定场次汇总表
                if calib_data:
                    st.subheader("📊 率定场次指标汇总")
                    calib_df = pd.DataFrame(calib_data)
                    st.dataframe(calib_df, use_container_width=True, hide_index=True)
                
                # 创建验证场次汇总表
                if valid_data:
                    st.subheader("📊 验证场次指标汇总")
                    valid_df = pd.DataFrame(valid_data)
                    st.dataframe(valid_df, use_container_width=True, hide_index=True)
                
                if calib_data and valid_data:
                    st.divider()
                    st.subheader("📊 模型表现对比")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**率定场次平均**")
                        calib_nse = np.mean([d['NSE'] for d in calib_data])
                        calib_kge = np.mean([d['KGE'] for d in calib_data])
                        calib_rmse = np.mean([d['RMSE'] for d in calib_data])
                        calib_pbias = np.mean([d['PBIAS'] for d in calib_data])
                        st.metric("NSE", f"{calib_nse:.4f}")
                        st.metric("KGE", f"{calib_kge:.4f}")
                        st.metric("RMSE", f"{calib_rmse:.4f}")
                        st.metric("PBIAS", f"{calib_pbias:.2f}%")
                    
                    with col2:
                        st.markdown("**验证场次平均**")
                        valid_nse = np.mean([d['NSE'] for d in valid_data])
                        valid_kge = np.mean([d['KGE'] for d in valid_data])
                        valid_rmse = np.mean([d['RMSE'] for d in valid_data])
                        valid_pbias = np.mean([d['PBIAS'] for d in valid_data])
                        st.metric("NSE", f"{valid_nse:.4f}")
                        st.metric("KGE", f"{valid_kge:.4f}")
                        st.metric("RMSE", f"{valid_rmse:.4f}")
                        st.metric("PBIAS", f"{valid_pbias:.2f}%")
                
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
                    call_minimax,
                    warmup_hours
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
            
            # 计算预热期步数
            if warmup_hours > 0 and user_timestep == 'hourly':
                warmup_steps = warmup_hours
            elif warmup_hours > 0:
                warmup_steps = warmup_hours // 24
            else:
                warmup_steps = 0
            
            if preanalysis_result.selected_events:
                # 构建率定场次列表
                calib_events = []
                _all_upstream_arr = all_upstream_arr if 'all_upstream_arr' in globals() else None
                for event in preanalysis_result.selected_events:
                    start_idx = max(0, event.start_idx)
                    end_idx = min(len(all_precip_arr), event.end_idx + 1)
                    calib_events.append({
                        'precip': all_precip_arr[start_idx:end_idx],
                        'evap': all_evap_arr[start_idx:end_idx],
                        'flow': all_flow_arr[start_idx:end_idx],
                        'upstream': _all_upstream_arr[start_idx:end_idx] if enable_upstream_routing and _all_upstream_arr is not None else None
                    })
                st.info(f"🤖 使用 {len(calib_events)} 场代表性洪水率定，共 {sum(len(e['flow']) for e in calib_events)} 个时间步" +
                       (f"，预热期 {warmup_steps} 步" if warmup_steps > 0 else ""))
            else:
                _all_upstream_arr = all_upstream_arr if 'all_upstream_arr' in globals() else None
                calib_events = [{
                    'precip': all_precip_arr,
                    'evap': all_evap_arr,
                    'flow': all_flow_arr,
                    'upstream': _all_upstream_arr if enable_upstream_routing and _all_upstream_arr is not None else None
                }]
                st.info(f"📊 整场洪水数据，共 {len(calib_events[0]['flow'])} 个时间步" +
                       (f"，预热期 {warmup_steps} 步" if warmup_steps > 0 else ""))
            
            progress_bar = st.progress(0)
            with st.spinner("🤖 AI Agent 正在率定模型..."):
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        continue
                    try:
                        spatial_data = {'area': catchment_area}
                        
                        total_models = len(RECOMMENDED_MODELS)
                        model_base = model_idx / total_models
                        
                        result = calibrate_model_fast(
                            model_name,
                            calib_events[0]['precip'],
                            calib_events[0]['evap'],
                            calib_events[0]['flow'],
                            max_iter=max_iter,
                            spatial_data=spatial_data,
                            timestep=user_timestep,
                            algorithm=algorithm,
                            algo_params=algo_params,
                            upstream_flow=calib_events[0].get('upstream'),
                            enable_routing=enable_upstream_routing,
                            calib_events=calib_events,
                            warmup_steps=warmup_steps,
                            progress_callback=lambda p: progress_bar.progress(model_base + p / total_models)
                        )
                    except Exception as e:
                        st.error(f"  ⚠️ {model_name} 率定异常: {type(e).__name__}: {str(e)}")
                        result = None
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
                    if result:
                        params, nse, simulated = result
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(calib_events[0]['flow'], simulated),
                            "mae": calc_mae(calib_events[0]['flow'], simulated),
                            "pbias": calc_pbias(calib_events[0]['flow'], simulated),
                            "simulated": simulated,
                            "calib_data": (calib_events[0]['precip'], calib_events[0]['evap'], calib_events[0]['flow']),
                        }
                        st.write(f"  ✅ {model_name}: 平均NSE = {nse:.4f}")
        else:
            # 单文件一场洪水模式
            file_calibration_results = {}
            
            # 计算预热期步数
            if warmup_hours > 0 and user_timestep == 'hourly':
                warmup_steps = warmup_hours
            elif warmup_hours > 0:
                warmup_steps = warmup_hours // 24
            else:
                warmup_steps = 0
            
            # 构建单场次列表
            _all_upstream = all_upstream_arr if 'all_upstream_arr' in globals() and all_upstream_arr is not None else None
            calib_events = [{
                'precip': all_precip_arr,
                'evap': all_evap_arr,
                'flow': all_flow_arr,
                'upstream': _all_upstream
            }]
            
            st.info(f"📊 整场洪水数据，共 {len(calib_events[0]['flow'])} 个时间步" +
                   (f"，预热期 {warmup_steps} 步" if warmup_steps > 0 else ""))
            
            progress_bar = st.progress(0)
            with st.spinner("🤖 AI Agent 正在率定模型..."):
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        continue
                    try:
                        spatial_data = {'area': catchment_area}
                        
                        total_models = len(RECOMMENDED_MODELS)
                        model_base = model_idx / total_models
                        
                        result = calibrate_model_fast(
                            model_name,
                            calib_events[0]['precip'],
                            calib_events[0]['evap'],
                            calib_events[0]['flow'],
                            max_iter=max_iter,
                            spatial_data=spatial_data,
                            timestep=user_timestep,
                            algorithm=algorithm,
                            algo_params=algo_params,
                            upstream_flow=calib_events[0].get('upstream'),
                            enable_routing=enable_upstream_routing,
                            calib_events=calib_events,
                            warmup_steps=warmup_steps,
                            progress_callback=lambda p: progress_bar.progress(model_base + p / total_models)
                        )
                    except Exception as e:
                        st.error(f"  ⚠️ {model_name} 率定异常: {type(e).__name__}: {str(e)}")
                        result = None
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
                    if result:
                        params, nse, simulated = result
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(calib_events[0]['flow'], simulated),
                            "mae": calc_mae(calib_events[0]['flow'], simulated),
                            "pbias": calc_pbias(calib_events[0]['flow'], simulated),
                            "simulated": simulated,
                            "calib_data": (calib_events[0]['precip'], calib_events[0]['evap'], calib_events[0]['flow']),
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
            '新安江模型2': '#e74c3c',
            'Tank水箱模型(完整版)': '#3498db',
            'HBV模型(完整版)': '#2ecc71',
        }
        
        if upload_mode == "单文件（连续序列）":
            # 连续序列模式
            # 剔除预热期数据
            if warmup_steps > 0 and len(all_flow_arr) > warmup_steps:
                all_flow_arr_plot = all_flow_arr[warmup_steps:]
                all_precip_arr_plot = all_precip_arr[warmup_steps:]
            else:
                all_flow_arr_plot = all_flow_arr
                all_precip_arr_plot = all_precip_arr
            
            fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
            
            ax.plot(all_flow_arr_plot, "k-", label="实测", linewidth=2.5)
            
            ax2 = ax.twinx()
            ax2.bar(range(len(all_precip_arr_plot)), all_precip_arr_plot, color='#87CEEB', alpha=0.5, width=1, label='降水')
            ax2.set_ylabel("降水 (mm)", fontsize=14, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
            ax2.invert_yaxis()
            
            xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
            
            simulated_list = []
            model_names = []
            nse_list = []
            for model_name, result in calibration_results.items():
                simulated = result['simulated']
                # 剔除预热期数据
                if warmup_steps > 0 and len(simulated) > warmup_steps:
                    simulated_plot = simulated[warmup_steps:]
                else:
                    simulated_plot = simulated
                color = model_colors.get(model_name, '#999999')
                label = f"{model_name} (NSE={result['nse']:.3f})"
                ax.plot(simulated_plot, color=color, label=label, linewidth=2, alpha=0.8)
                simulated_list.append(simulated_plot)
                model_names.append(model_name)
                nse_list.append(result['nse'])
                
                summary_data.append({
                    "文件": "连续序列",
                    "场次": "整场洪水",
                    "类型": "率定",
                    "模型": model_name,
                    "NSE": result['nse'],
                    "KGE": result.get('kge', result['nse']),
                    "RMSE": result['rmse'],
                    "PBIAS": result['pbias']
                })
            
            if len(simulated_list) >= 2:
                weights = calc_bma_weights(nse_list)
                bma_result = apply_bma_ensemble(simulated_list, weights)
                bma_nse = calc_nse(all_flow_arr, bma_result)
                bma_rmse = calc_rmse(all_flow_arr, bma_result)
                bma_pbias = calc_pbias(all_flow_arr, bma_result)
                weights_str = format_weights_string(model_names, weights)
                label = f'BMA集成 (NSE={bma_nse:.3f}) [{weights_str}]'
                ax.plot(bma_result, color='#9b59b6', linestyle='--', 
                        label=label, linewidth=2.5, alpha=0.9)
                
                summary_data.append({
                    "文件": "连续序列",
                    "场次": "整场洪水",
                    "类型": "率定",
                    "模型": "BMA集成",
                    "NSE": bma_nse,
                    "KGE": bma_nse,
                    "RMSE": bma_rmse,
                    "PBIAS": bma_pbias
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
            # 剔除预热期数据
            if warmup_steps > 0 and len(all_flow_arr) > warmup_steps:
                all_flow_arr_plot = all_flow_arr[warmup_steps:]
                all_precip_arr_plot = all_precip_arr[warmup_steps:]
            else:
                all_flow_arr_plot = all_flow_arr
                all_precip_arr_plot = all_precip_arr
            
            fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
            
            ax.plot(all_flow_arr_plot, "k-", label="实测", linewidth=2.5)
            
            ax2 = ax.twinx()
            ax2.bar(range(len(all_precip_arr_plot)), all_precip_arr_plot, color='#87CEEB', alpha=0.5, width=1, label='降水')
            ax2.set_ylabel("降水 (mm)", fontsize=14, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
            ax2.invert_yaxis()
            
            xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
            
            simulated_list = []
            model_names = []
            nse_list = []
            for model_name, result in calibration_results.items():
                simulated = result['simulated']
                # 剔除预热期数据
                if warmup_steps > 0 and len(simulated) > warmup_steps:
                    simulated_plot = simulated[warmup_steps:]
                else:
                    simulated_plot = simulated
                color = model_colors.get(model_name, '#999999')
                label = f"{model_name} (NSE={result['nse']:.3f})"
                ax.plot(simulated_plot, color=color, label=label, linewidth=2, alpha=0.8)
                simulated_list.append(simulated_plot)
                model_names.append(model_name)
                nse_list.append(result['nse'])
                
                summary_data.append({
                    "文件": "单场洪水",
                    "场次": "整场洪水",
                    "类型": "率定",
                    "模型": model_name,
                    "NSE": result['nse'],
                    "KGE": result.get('kge', result['nse']),
                    "RMSE": result['rmse'],
                    "PBIAS": result['pbias']
                })
            
            if len(simulated_list) >= 2:
                weights = calc_bma_weights(nse_list)
                bma_result = apply_bma_ensemble(simulated_list, weights)
                bma_nse = calc_nse(all_flow_arr, bma_result)
                bma_rmse = calc_rmse(all_flow_arr, bma_result)
                bma_pbias = calc_pbias(all_flow_arr, bma_result)
                weights_str = format_weights_string(model_names, weights)
                label = f'BMA集成 (NSE={bma_nse:.3f}) [{weights_str}]'
                ax.plot(bma_result, color='#9b59b6', linestyle='--', 
                        label=label, linewidth=2.5, alpha=0.9)
                
                summary_data.append({
                    "文件": "单场洪水",
                    "场次": "整场洪水",
                    "类型": "率定",
                    "模型": "BMA集成",
                    "NSE": bma_nse,
                    "KGE": bma_nse,
                    "RMSE": bma_rmse,
                    "PBIAS": bma_pbias
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
            
            # AI报告（仅多文件模式）
            if upload_mode == "多文件（每文件一场洪水）":
                try:
                    st.divider()
                    st.subheader("🤖 AI Agent 智能分析报告")
                    
                    with st.spinner("🤖 AI Agent 正在生成分析报告..."):
                        multifile_report = generate_multifile_report(
                            file_data_list,
                            calibration_results,
                            file_simulation_results,
                            call_minimax,
                            warmup_hours
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
                except NameError:
                    pass
                st.markdown(multifile_report)
                
                if not multifile_report.startswith("[ERROR]"):
                    report_filename = f"multifile_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                    st.download_button(
                        "📥 导出报告 (Markdown)",
                        data=multifile_report,
                        file_name=report_filename,
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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("支持模型", f"{len(RECOMMENDED_MODELS)} 个")
    with col2:
        st.metric("支持格式", "CSV / Excel")
    with col3:
        st.metric("率定算法", "5种可选")
    with col4:
        st.metric("汇流演算", "马斯京根")
    
    st.divider()
    
    st.subheader("📖 使用说明")
    st.markdown("""
    1. **上传数据**：上传一个或多个水文数据文件
    2. **洪水场次**：系统自动识别每场洪水
    3. **自动率定**：对每场洪水分别进行多模型率定
    4. **结果对比**：查看性能指标(NSE/KGE/RMSE/PBIAS)、参数表格、流量过程线
    5. **智能报告**：自动生成分析报告并支持导出
    """)
    
    st.subheader("🔧 率定算法选项")
    st.markdown("""
    - **两阶段算法(推荐)**：dual_annealing + L-BFGS-B，快速全局搜索+局部精细优化
    - **PSO粒子群**：粒子群优化算法，适合大规模问题
    - **SCE-UA**：洗牌复形进化算法，全局优化能力强
    - **DE差分进化**：简单高效，适合连续参数优化
    - **GA遗传算法**：进化过程中保持多样性
    """)
    
    st.subheader("🌊 上游出库汇流演算")
    st.markdown("""
    支持启用上游出库马斯京根河道汇流演算功能：
    - 在侧边栏启用"上游出库汇流演算"
    - 指定上游出库列名（数据需在同一文件中）
    - Muskingum参数(k, x)与水文模型参数一起率定
    """)
    
    st.subheader("📊 数据格式要求")
    st.markdown("""
    数据应包含以下列（列名支持中英文）：
    - **时间列**: date / time / 日期 / 时间
    - **降水列**: precip / rainfall / p / 降水 / 降雨
    - **蒸发列**: evap / et / 蒸发 (可选)
    - **流量列**: flow / discharge / q / 流量 / 径流
    - **上游出库列**: upstream (可选，启用汇流演算时需要)
    """)

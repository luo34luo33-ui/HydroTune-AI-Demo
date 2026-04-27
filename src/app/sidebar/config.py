# -*- coding: utf-8 -*-
"""
侧边栏配置模块
从 app.py 侧边栏部分提取
"""
import streamlit as st
from typing import Dict, Any, Tuple, Optional


def render_sidebar(
    RECOMMENDED_MODELS: list = None,
) -> Dict[str, Any]:
    """渲染侧边栏并返回配置参数
    
    Returns:
        包含所有侧边栏配置的字典
    """
    if RECOMMENDED_MODELS is None:
        RECOMMENDED_MODELS = ['HBV模型', '新安江模型2', 'tank水箱模型']
    
    # 使用 st.sidebar 渲染到侧边栏
    st.sidebar.markdown("### 🤖 AI Agent 状态")
    st.sidebar.success("🟢 智能Agent就绪")
    
    st.sidebar.markdown("### 📊 模型状态")
    from src.models.registry import ModelRegistry
    all_models = ModelRegistry.list_models()
    check_models = RECOMMENDED_MODELS
    for model in check_models:
        if model in all_models:
            st.sidebar.success(f"✅ {model}")
        else:
            st.sidebar.error(f"❌ {model}")
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📂 导航")
    if st.sidebar.button("🏠 主页", use_container_width=True):
        st.session_state.current_page = 'main'
        st.rerun()
    if st.sidebar.button("📚 水文模型介绍", use_container_width=True):
        st.session_state.current_page = 'models'
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.header("📁 数据上传")
    
    uploaded_files = st.sidebar.file_uploader(
        "上传水文数据文件",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="支持多文件上传",
    )
    
    catchment_area = st.sidebar.number_input(
        "流域面积 (km²)",
        min_value=0.1,
        max_value=100000.0,
        value=150.7944,
        step=1.0,
        format="%.4f",
    )
    
    st.sidebar.header("⏱️ 时间尺度")
    timestep = st.sidebar.radio(
        "时间尺度",
        options=["hourly", "daily"],
        index=1,
        horizontal=True,
        help="选择数据的时间尺度：小时或日"
    )

    upload_mode = st.sidebar.radio(
        "📂 上传模式",
        options=[
            "单文件（一场洪水）",
            "单文件（连续序列）",
            "多文件（每文件一场洪水）"
        ],
        index=0,
        horizontal=False,
    )

    st.sidebar.header("⏱️ 预热期设置")
    has_warmup = st.sidebar.radio(
        "预热期",
        options=["无", "有"],
        index=0,
        help="预热期是指模型开始响应前需要丢弃的时间段"
    )
    warmup_hours = 0
    if has_warmup == "有":
        warmup_hours = st.sidebar.number_input(
            "预热期长度(h)",
            min_value=0,
            max_value=720,
            value=24,
            step=1,
            help="需要丢弃的预热期时间长度"
        )

    st.sidebar.divider()
    
    st.sidebar.header("📥 直接导入参数")
    use_imported_params = st.sidebar.radio(
        "参数模式",
        options=["率定参数", "导入参数"],
        index=0,
        horizontal=True,
        help="选择'导入参数'时直接使用参数文件，不进行率定"
    )
    
    imported_params = {}
    
    if use_imported_params == "导入参数":
        st.sidebar.success("📥 导入参数模式")
        
        st.sidebar.markdown("**Tank水箱模型参数：**")
        tank_param_file = st.sidebar.file_uploader(
            "上传Tank模型参数",
            type=["csv"],
            key="tank_param_file"
        )
        if tank_param_file is not None:
            imported_params.update(_parse_tank_params(tank_param_file))
        
        st.sidebar.markdown("**HBV模型参数：**")
        hbv_param_file = st.sidebar.file_uploader(
            "上传HBV模型参数",
            type=["csv"],
            key="hbv_param_file"
        )
        if hbv_param_file is not None:
            imported_params.update(_parse_hbv_params(hbv_param_file))
        
        st.sidebar.markdown("**新安江模型参数：**")
        xaj_param_file = st.sidebar.file_uploader(
            "上传新安江模型参数",
            type=["csv"],
            key="xaj_param_file"
        )
        if xaj_param_file is not None:
            result = _parse_xaj_params(xaj_param_file)
            imported_params.update(result.get('params', {}))
            if 'routing' in result:
                st.session_state['imported_k_routing'] = result['routing'].get('k')
                st.session_state['imported_x_routing'] = result['routing'].get('x')

    st.sidebar.divider()

    st.sidebar.header("📋 列名配置")
    with st.sidebar.expander("配置数据列名映射"):
        st.sidebar.write("请输入原始数据列名：")
        
        date_col = st.sidebar.text_input("时间列名", value="date", key="date_col")
        precip_col = st.sidebar.text_input("降水列名", value="precip", key="precip_col")
        evap_col = st.sidebar.text_input("蒸发列名", value="evap", key="evap_col")
        flow_col = st.sidebar.text_input("流量列名", value="flow", key="flow_col")
    
    column_mapping = {
        'date': date_col if date_col else 'date',
        'precip': precip_col if precip_col else 'precip',
        'evap': evap_col if evap_col else 'evap',
        'flow': flow_col if flow_col else 'flow',
        'upstream': "",
    }

    st.sidebar.divider()

    st.sidebar.header("🌊 上游汇流演算")
    enable_upstream_routing = st.sidebar.checkbox(
        "启用上游出库汇流演算",
        value=False,
        help="启用后将使用马斯京根方法将上游来水演算后叠加"
    )
    
    k_routing, x_routing = 2.5, 0.25
    if enable_upstream_routing:
        upstream_col = st.sidebar.text_input(
            "上游出库列名",
            value="",
            help="上游断面流量列名"
        )
        column_mapping['upstream'] = upstream_col if upstream_col else ""
        
        st.sidebar.markdown("**马斯京根参数：**")
        col_k, col_x = st.sidebar.columns(2)
        with col_k:
            k_routing = st.sidebar.number_input(
                "k (河道传播时间)",
                min_value=0.5,
                max_value=10.0,
                value=2.5,
                step=0.1,
            )
        with col_x:
            x_routing = st.sidebar.number_input(
                "x (洪水坦化系数)",
                min_value=0.0,
                max_value=0.5,
                value=0.25,
                step=0.01,
            )
    else:
        column_mapping['upstream'] = ""

    st.sidebar.divider()

    st.sidebar.markdown("""
⚙️ **率定设置**
""", unsafe_allow_html=True)

    algorithm = st.sidebar.selectbox(
        "优化算法",
        options=["两阶段算法(推荐)", "PSO", "SCE-UA", "差分进化(DE)", "遗传算法(GA)"],
        index=0,
        help="选择率定使用的优化算法"
    )

    max_iter = st.sidebar.slider(
        "迭代次数",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="优化算法迭代次数"
    )

    algo_params = {}
    if algorithm == "PSO":
        st.sidebar.markdown("**PSO 参数**")
        n_particles = st.sidebar.slider("粒子数", 10, 100, 100)
        w = st.sidebar.slider("惯性权重 w", 0.0, 1.0, 0.7, 0.01)
        c1 = st.sidebar.slider("个体学习因子 c1", 0.0, 2.0, 1.5, 0.1)
        c2 = st.sidebar.slider("群体学习因子 c2", 0.0, 2.0, 1.5, 0.1)
        algo_params = {"n_particles": n_particles, "w": w, "c1": c1, "c2": c2}
    elif algorithm == "遗传算法(GA)":
        st.sidebar.markdown("**GA 参数**")
        pop_size = st.sidebar.slider("种群大小", 10, 100, 50)
        crossover_rate = st.sidebar.slider("交叉率", 0.0, 1.0, 0.8, 0.05)
        mutation_rate = st.sidebar.slider("变异率", 0.0, 1.0, 0.1, 0.05)
        algo_params = {"pop_size": pop_size, "crossover_rate": crossover_rate, "mutation_rate": mutation_rate}
    elif algorithm == "SCE-UA":
        st.sidebar.markdown("**SCE-UA 参数**")
        n_complexes = st.sidebar.slider("复形数量", 2, 10, 5)
        points_per_complex = st.sidebar.slider("每复形点数", 5, 20, 10)
        algo_params = {"n_complexes": n_complexes, "points_per_complex": points_per_complex}
    elif algorithm == "差分进化(DE)":
        st.sidebar.markdown("**DE 参数**")
        mutation_factor = st.sidebar.slider("变异因子 F", 0.0, 2.0, 0.8, 0.1)
        crossover_prob = st.sidebar.slider("交叉概率 CR", 0.0, 1.0, 0.7, 0.1)
        pop_size_de = st.sidebar.slider("种群大小", 10, 100, 50)
        algo_params = {"mutation_factor": mutation_factor, "crossover_prob": crossover_prob, 
                       "pop_size": pop_size_de}

    st.sidebar.divider()

    st.sidebar.header("📊 率定模型")
    for model_name in RECOMMENDED_MODELS:
        st.sidebar.write(f"✅ {model_name}")

    return {
        'uploaded_files': uploaded_files,
        'catchment_area': catchment_area,
        'timestep': timestep,
        'upload_mode': upload_mode,
        'warmup_hours': warmup_hours,
        'use_imported_params': use_imported_params,
        'imported_params': imported_params,
        'column_mapping': column_mapping,
        'enable_upstream_routing': enable_upstream_routing,
        'k_routing': k_routing,
        'x_routing': x_routing,
        'algorithm': algorithm,
        'max_iter': max_iter,
        'algo_params': algo_params,
        'recommended_models': RECOMMENDED_MODELS,
    }


def _parse_tank_params(file) -> Dict:
    """解析Tank模型参数文件"""
    import pandas as pd
    import io
    
    try:
        file_content = file.getvalue()
        if not file_content or len(file_content.strip()) == 0:
            raise ValueError("文件为空，请检查上传的文件")
        
        tank_df = None
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                tank_df = pd.read_csv(io.BytesIO(file_content), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        
        if tank_df is None:
            raise ValueError("无法解码文件，请保存为UTF-8编码")
        if len(tank_df.columns) == 0:
            raise ValueError("CSV文件没有列，请检查文件格式")
        if len(tank_df) == 0:
            raise ValueError("CSV文件没有数据行，请检查文件内容")
        
        tank_params = {
            col: float(tank_df[col].values[0])
            for col in tank_df.columns
            if col != '模型' and col not in ['k_routing', 'x_routing']
        }
        st.success(f"✅ Tank模型参数导入成功: {tank_params}")
        return {'tank水箱模型': tank_params}
    except Exception as e:
        st.error(f" Tank参数解析失败: {e}")
        return {}


def _parse_hbv_params(file) -> Dict:
    """解析HBV模型参数文件"""
    import pandas as pd
    
    try:
        hbv_df = pd.read_csv(file)
        hbv_params = {col: float(hbv_df[col].values[0]) for col in hbv_df.columns if col != '模型'}
        st.success(f"✅ HBV模型参数导入成功: {hbv_params}")
        return {'HBV模型': hbv_params}
    except Exception as e:
        st.error(f" HBV参数解析失败: {e}")
        return {}


def _parse_xaj_params(file) -> Dict:
    """解析新安江模型参数文件"""
    import pandas as pd
    import io
    
    try:
        file_content = file.getvalue()
        
        xaj_df = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                xaj_df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        
        if xaj_df is None or xaj_df.empty:
            st.warning("⚠️ CSV文件为空，请检查文件格式")
            return {'params': {}, 'routing': None}
        elif len(xaj_df.columns) < 2:
            st.warning(f"⚠️ CSV文件列数不足: {xaj_df.columns}")
            return {'params': {}, 'routing': None}
        
        xaj_params = {col: float(xaj_df[col].values[0]) for col in xaj_df.columns if col != '模型'}
        st.success(f"✅ 新安江模型参数导入成功: {xaj_params}")
        
        result = {'params': {'新安江模型2': xaj_params}}
        
        if 'k_routing' in xaj_df.columns and 'x_routing' in xaj_df.columns:
            result['routing'] = {
                'k': float(xaj_df['k_routing'].values[0]),
                'x': float(xaj_df['x_routing'].values[0])
            }
            st.success(f"✅ 马斯京根参数导入成功: k={result['routing']['k']}, x={result['routing']['x']}")
        
        return result
    except Exception as e:
        st.error(f" 新安江参数解析失败: {e}")
        return {'params': {}, 'routing': None}
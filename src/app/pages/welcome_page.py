# -*- coding: utf-8 -*-
"""
欢迎主页模块
从 app.py 欢迎主页部分提取
"""
import streamlit as st


def render_welcome_page():
    """渲染欢迎主页"""
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
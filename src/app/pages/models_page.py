# -*- coding: utf-8 -*-
"""
模型介绍页面模块
从 app.py show_models_page 函数提取
"""
import streamlit as st


def render_models_page():
    """渲染水文模型介绍页面"""
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
    """, unsafe_allow_html=True)
    
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
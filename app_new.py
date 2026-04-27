# -*- coding: utf-8 -*-
"""
HydroTune-AI - 流域水文模型智能率定系统
Streamlit 主入口 (新版，使用模块化结构)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.app.sidebar import render_sidebar
from src.app.pages import render_models_page, render_welcome_page
from src.app.handlers import DataHandler, CalibrationHandler, VisualizationHandler
from src.app.reports import ReportGenerator
from src.models.registry import ModelRegistry
from src.hydro_calc import (
    calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias, calc_kge,
    muskingum_routing
)
from src.data_preanalysis import DataPreAnalyzer
from src.llm_api import call_minimax
from src.data_agent import detect_flood_events, FloodEvent, infer_timestep_by_llm
from src.llm_reporter import generate_calibration_report, generate_comprehensive_report

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

# 页面状态初始化
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# ============================================================
# 工具函数
# ============================================================
def apply_upstream_routing(simulated, upstream_arr, k_routing, x_routing, warmup_steps=0):
    """将上游流量经过马斯京根演算后叠加到出口断面"""
    if upstream_arr is None or len(upstream_arr) == 0:
        return simulated
    if len(upstream_arr) != len(simulated):
        return simulated
    routed = muskingum_routing(upstream_arr, k_routing, x_routing)
    return simulated + routed


# ============================================================
# 页面路由
# ============================================================
def main():
    """主函数"""
    RECOMMENDED_MODELS = ['HBV模型', '新安江模型2', 'tank水箱模型']
    
    # 渲染侧边栏（使用 st.sidebar）
    sidebar_config = render_sidebar(RECOMMENDED_MODELS)
    
    uploaded_files = sidebar_config.get('uploaded_files')
    catchment_area = sidebar_config.get('catchment_area')
    user_timestep = sidebar_config.get('timestep', 'daily')
    upload_mode = sidebar_config.get('upload_mode')
    warmup_hours = sidebar_config.get('warmup_hours')
    use_imported_params = sidebar_config.get('use_imported_params')
    imported_params = sidebar_config.get('imported_params')
    column_mapping = sidebar_config.get('column_mapping')
    enable_upstream_routing = sidebar_config.get('enable_upstream_routing')
    k_routing = sidebar_config.get('k_routing')
    x_routing = sidebar_config.get('x_routing')
    algorithm = sidebar_config.get('algorithm')
    max_iter = sidebar_config.get('max_iter')
    algo_params = sidebar_config.get('algo_params')
    
    # 页面路由
    if st.session_state.current_page == 'models':
        render_models_page()
        return
    
    if not uploaded_files:
        render_welcome_page()
        return
    
    # 渲染标题
    st.title("HydroTune-AI - 流域水文模型智能率定系统")
    st.caption("上传数据 → 智能清洗 → 多模型率定 → 自动报告")
    
    # 数据处理
    data_handler = DataHandler()
    file_dfs = data_handler.read_files(uploaded_files)
    
    if not file_dfs:
        st.error("没有可用的文件")
        st.stop()
    
    first_file_name, first_df = file_dfs[0]
    st.divider()
    st.subheader("📋 配置确认")
    st.write(f"**{first_file_name}** 的列名: `{list(first_df.columns)}`")
    
    col_mapping_accepted = st.checkbox(
        f"使用列名映射: {column_mapping['date']}→date, {column_mapping['precip']}→precip, {column_mapping['evap']}→evap, {column_mapping['flow']}→flow",
        value=True
    )
    
    if st.button("🚀 开始分析", type="primary", disabled=not col_mapping_accepted):
        with st.expander("📥 数据文件处理详情", expanded=False):
            for file_idx, (file_name, raw_df) in enumerate(file_dfs):
                st.write(f"**📄 {file_name}** - {raw_df.shape[0]} 行")
        
        # 根据上传模式处理数据
        file_data_list = []
        
        for file_idx, (file_name, raw_df) in enumerate(file_dfs):
            df = raw_df.copy()
            for std_name, orig_name in column_mapping.items():
                if orig_name and orig_name in df.columns:
                    df = df.rename(columns={orig_name: std_name})
            
            if 'precip' not in df.columns or 'flow' not in df.columns:
                st.error(f"缺少必要列。当前列: {list(df.columns)}")
                continue
            
            if 'evap' not in df.columns:
                df['evap'] = 0.0
            if 'date' not in df.columns:
                df['date'] = range(len(df))
            
            df = df.fillna(0)
            file_data_list.append({
                'name': file_name,
                'precip': df['precip'].values,
                'evap': df['evap'].values,
                'flow': df['flow'].values,
            })
        
        # 根据上传模式决定如何处理
        if upload_mode == "多文件（每文件一场洪水）":
            st.divider()
            st.subheader("🌊 多文件洪水场次率定")
            st.info(f"📊 检测到 {len(file_data_list)} 个文件")
            
            # 划分率定集和验证集 (75% 率定, 25% 验证)
            n_calib = int(len(file_data_list) * 0.75)
            calib_files = file_data_list[:n_calib]
            valid_files = file_data_list[n_calib:]
            
            st.write(f"📈 **率定集**: {len(calib_files)} 个文件 (75%)")
            st.write(f"📉 **验证集**: {len(valid_files)} 个文件 (25%)")
            
            # 合并率定集数据
            calib_precip = []
            calib_evap = []
            calib_flow = []
            
            for file_data in calib_files:
                calib_precip.extend(file_data['precip'].tolist())
                calib_evap.extend(file_data['evap'].tolist())
                calib_flow.extend(file_data['flow'].tolist())
            
            calib_precip_arr = np.array(calib_precip)
            calib_evap_arr = np.array(calib_evap)
            calib_flow_arr = np.array(calib_flow)
            
            st.write(f"📊 率定数据总量: {len(calib_flow_arr)} 个时间步")
            
            # 率定模型
            st.divider()
            st.subheader("🧠 模型率定")
            
            spatial_data = {'area': catchment_area, 'timestep': user_timestep}
            calibration_results = {}
            progress_bar = st.progress(0)
            
            for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                st.write(f"🔄 开始率定 {model_name}...")
                
                try:
                    result = calibrate_model_fast(
                        model_name=model_name,
                        precip=calib_precip_arr,
                        evap=calib_evap_arr,
                        observed_flow=calib_flow_arr,
                        max_iter=max_iter,
                        spatial_data=spatial_data,
                        timestep=user_timestep,
                        algorithm=algorithm,
                        algo_params=algo_params,
                        progress_callback=lambda p: progress_bar.progress((model_idx + p) / len(RECOMMENDED_MODELS)),
                    )
                    
                    params, nse, simulated = result
                    calibration_results[model_name] = {
                        "params": params,
                        "nse": nse,
                        "simulated": simulated,
                        "calib_data": (calib_precip_arr, calib_evap_arr, calib_flow_arr),
                    }
                    st.write(f"  ✅ {model_name}: 率定期NSE={nse:.4f}")
                except Exception as e:
                    st.error(f"  ⚠️ {model_name} 率定异常: {e}")
                
                progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
            
            # 验证集评估
            st.divider()
            st.subheader("📈 验证集评估")
            
            from src.hydro_calc import calc_nse
            
            valid_results = {}
            
            for model_name, calib_result in calibration_results.items():
                if calib_result is None:
                    continue
                
                params = calib_result['params']
                st.write(f"### 📊 {model_name}")
                
                model_nses = []
                
                for file_data in valid_files:
                    precip_arr = file_data['precip']
                    evap_arr = file_data['evap']
                    flow_arr = file_data['flow']
                    
                    model = ModelRegistry.get_model(model_name)
                    simulated = model.run(precip_arr, evap_arr, params, spatial_data)
                    
                    valid_nse = calc_nse(flow_arr, simulated)
                    model_nses.append(valid_nse)
                    st.write(f"  📄 {file_data['name']}: NSE={valid_nse:.4f}")
                
                avg_nse = np.mean(model_nses) if model_nses else 0
                st.metric("验证集平均NSE", f"{avg_nse:.4f}")
                valid_results[model_name] = {
                    "calib_nse": calib_result['nse'],
                    "valid_nse": avg_nse,
                    "valid_nses": model_nses,
                }
                st.divider()
            
            # 汇总表格
            st.subheader("📊 结果汇总")
            summary_data = []
            for model_name, result in valid_results.items():
                summary_data.append({
                    '模型': model_name,
                    '率定期NSE': f"{result['calib_nse']:.4f}",
                    '验证集NSE': f"{result['valid_nse']:.4f}",
                })
            
            if summary_data:
                import pandas as pd
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)
            
            st.stop()
        
        # 单文件模式：合并所有数据
        all_precip = []
        all_evap = []
        all_flow = []
        
        for file_data in file_data_list:
            all_precip.extend(file_data['precip'].tolist())
            all_evap.extend(file_data['evap'].tolist())
            all_flow.extend(file_data['flow'].tolist())
        
        all_precip_arr = np.array(all_precip)
        all_evap_arr = np.array(all_evap)
        all_flow_arr = np.array(all_flow)
        
        # AI预分析
        st.divider()
        st.subheader("🧠 AI Agent 数据预分析")
        
        import pandas as pd
        dates = pd.date_range(start='2020-01-01', periods=len(all_flow_arr), freq='D')
        
        preanalyzer = DataPreAnalyzer(area=catchment_area)
        preanalyzer.timestep = user_timestep
        
        # 洪水事件检测
        with st.spinner("检测洪水事件..."):
            flood_events = detect_flood_events(dates, all_precip_arr, all_flow_arr, all_evap_arr)
        
        # 完整预分析
        with st.spinner("进行数据预分析..."):
            preanalysis_result = preanalyzer.analyze(
                dates, all_precip_arr, all_flow_arr, 
                timestep=user_timestep, n_select=5
            )
        
        quality = preanalysis_result.quality
        
        if len(all_flow_arr) == 0:
            st.error("⚠️ 流量数据为空，请检查上传的数据文件")
            st.stop()
        
        peak_idx = np.argmax(all_flow_arr)
        peak_flow = all_flow_arr[peak_idx]
        baseflow = np.percentile(all_flow_arr, 10)
        
        st.success(f"✅ AI Agent 完成数据分析")
        
        # 洪水事件信息
        st.write(f"**🏊 识别到 {len(flood_events)} 个洪水事件，选取 {len(preanalysis_result.selected_events)} 个代表性场次**")
        
        with st.expander("📈 数据分析详情", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("数据完整率", f"{quality.completeness:.1f}%")
            col2.metric("时间连续性", f"{quality.continuity:.1f}%")
            col3.metric("降水-径流相关", f"{quality.correlation:.3f}")
            col4.metric("质量等级", quality.quality_level)
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("洪峰流量", f"{peak_flow:.1f} m³/s")
            col2.metric("基流", f"{baseflow:.1f} m³/s")
            col3.metric("洪水场次", f"{len(flood_events)} 场")
        
        # 生成预分析报告
        from src.llm_reporter import generate_preanalysis_report
        with st.spinner("生成智能报告..."):
            preanalysis_report = generate_preanalysis_report(preanalysis_result)
        
        with st.expander("📝 AI 预分析报告", expanded=False):
            st.markdown(preanalysis_report)
        
        # 模型率定
        st.divider()
        st.subheader("🧠 AI Agent 模型率定")
        
        calibration_results = {}
        progress_bar = st.progress(0)
        
        spatial_data = {'area': catchment_area, 'timestep': user_timestep}
        
        for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
            st.write(f"  🔄 开始率定 {model_name}...")
            
            try:
                result = calibrate_model_fast(
                    model_name=model_name,
                    precip=all_precip_arr,
                    evap=all_evap_arr,
                    observed_flow=all_flow_arr,
                    max_iter=max_iter,
                    spatial_data=spatial_data,
                    timestep=user_timestep,
                    algorithm=algorithm,
                    algo_params=algo_params,
                    progress_callback=lambda p: progress_bar.progress((model_idx + p) / len(RECOMMENDED_MODELS)),
                )
                
                params, nse, simulated = result
                calibration_results[model_name] = {
                    "model_name": model_name,
                    "params": params,
                    "nse": nse,
                    "rmse": calc_rmse(all_flow_arr, simulated),
                    "kge": calc_kge(all_flow_arr, simulated),
                    "pbias": calc_pbias(all_flow_arr, simulated),
                    "simulated": simulated,
                    "calib_data": (all_precip_arr, all_evap_arr, all_flow_arr),
                }
                st.write(f"  ✅ {model_name}: NSE={nse:.4f}")
                
            except Exception as e:
                st.error(f"  ⚠️ {model_name} 率定异常: {e}")
                import traceback
                st.code(traceback.format_exc()[-600:])
            
            progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
        
        # 结果展示
        st.divider()
        st.subheader("📈 模型率定结果")
        
        viz = VisualizationHandler()
        
        # 绘制所有模型的流量过程对比图
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        
        x = np.arange(len(all_flow_arr))
        xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
        
        ax.plot(x, all_flow_arr, 'k-', linewidth=2, label='实测流量', alpha=0.8)
        
        colors = {'HBV模型': '#e74c3c', '新安江模型2': '#3498db', 'tank水箱模型': '#2ecc71'}
        
        for model_name, result in calibration_results.items():
            if result is None:
                continue
            color = colors.get(model_name, '#999999')
            ax.plot(x, result['simulated'], color=color, linewidth=1.5, 
                   label=f"{model_name} (NSE={result.get('nse', 0):.3f})", alpha=0.7)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel(r'流量 ($m^3/s$)', fontsize=12)
        ax.set_title("流量过程对比", fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # 各模型详细结果
        for model_name, result in calibration_results.items():
            if result is None:
                continue
            
            with st.expander(f"📊 {model_name} 详细结果"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("NSE", f"{result.get('nse', 0):.4f}")
                with col2:
                    st.metric("RMSE", f"{result.get('rmse', 0):.2f}")
                with col3:
                    st.metric("KGE", f"{result.get('kge', 0):.4f}")
                with col4:
                    st.metric("Pbias", f"{result.get('pbias', 0):.2f}%")
                
                # 流量过程对比图
                fig1 = viz.plot_flow_comparison(
                    all_flow_arr, 
                    result['simulated'],
                    title=f"{model_name} 流量过程对比",
                    timestep=user_timestep
                )
                st.pyplot(fig1)
                
                # 散点图
                fig2 = viz.plot_scatter(
                    all_flow_arr,
                    result['simulated'],
                    title=f"{model_name} 观测-模拟散点图"
                )
                st.pyplot(fig2)
        
        # 指标对比
        st.divider()
        st.subheader("📊 模型表现对比")
        
        viz.plot_metrics_table(calibration_results)
        
        # XGBoost误差校正
        st.divider()
        st.subheader("🔧 XGBoost 误差校正")
        
        try:
            from src.app.error_correction import (
                ErrorCorrector, select_best_model, apply_error_correction
            )
            
            with st.spinner("选择最优模型并训练误差校正器..."):
                # 选择最优模型
                best_name, best_result = select_best_model(calibration_results)
                
                if best_name and best_result is not None:
                    st.success(f"✅ 最优模型: {best_name} (NSE={best_result['nse']:.4f})")
                    
                    # 应用误差校正
                    correction_result = apply_error_correction(
                        best_name,
                        best_result,
                        all_precip_arr,
                        all_flow_arr,
                    )
                    
                    # 显示校正效果
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "校正前NSE",
                            f"{correction_result['nse_before']:.4f}"
                        )
                    with col2:
                        st.metric(
                            "校正后NSE",
                            f"{correction_result['nse_after']:.4f}"
                        )
                    with col3:
                        st.metric(
                            "NSE提升",
                            f"{correction_result['nse_improvement']:+.4f}",
                            delta=correction_result['nse_improvement']
                        )
                    
                    # 绘制校正前后对比图
                    x = np.arange(len(all_flow_arr))
                    xlabel_text = "时间(天)" if user_timestep == 'daily' else "时间(h)"
                    
                    fig3, ax3 = plt.subplots(figsize=(14, 5), dpi=150)
                    ax3.plot(x, all_flow_arr, 'k-', linewidth=2, label='实测流量', alpha=0.8)
                    ax3.plot(x, correction_result['simulated'], 'b--', linewidth=1.5,
                            label=f"原始 ({best_name})", alpha=0.6)
                    ax3.plot(x, correction_result['corrected'], 'r-', linewidth=1.5,
                            label="XGB校正后", alpha=0.8)
                    
                    ax3.set_xlabel(xlabel_text, fontsize=12)
                    ax3.set_ylabel(r'流量 ($m^3/s$)', fontsize=12)
                    ax3.set_title(f"{best_name} + XGB误差校正 流量过程对比", fontsize=14)
                    ax3.legend(fontsize=10, loc='upper right')
                    ax3.grid(True, alpha=0.3)
                    
                    st.pyplot(fig3)
                    
                    # 散点图对比
                    fig4, ax4 = plt.subplots(figsize=(8, 8), dpi=150)
                    
                    max_val = max(np.max(all_flow_arr), np.max(correction_result['corrected'])) * 1.1
                    ax4.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1线')
                    ax4.scatter(all_flow_arr, correction_result['simulated'], 
                               alpha=0.4, s=15, c='blue', label='原始模拟')
                    ax4.scatter(all_flow_arr, correction_result['corrected'], 
                               alpha=0.4, s=15, c='red', label='XGB校正后')
                    
                    ax4.set_xlabel('实测流量 ($m^3/s$)', fontsize=12)
                    ax4.set_ylabel('模拟流量 ($m^3/s$)', fontsize=12)
                    ax4.set_title(f"{best_name} 观测-模拟散点图", fontsize=14)
                    ax4.set_xlim(0, max_val)
                    ax4.set_ylim(0, max_val)
                    ax4.set_aspect('equal')
                    ax4.grid(True, alpha=0.3)
                    ax4.legend()
                    
                    st.pyplot(fig4)
                    
                else:
                    st.warning("未能找到有效的率定结果")
                    
        except ImportError as e:
            st.warning(f"XGBoost误差校正模块不可用: {e}")
        except Exception as e:
            st.error(f"误差校正失败: {e}")
        
        # 报告生成
        st.divider()
        st.subheader("🤖 AI Agent 智能分析报告")
        
        report_gen = ReportGenerator()
        report_content = report_gen.generate_calibration_report(
            calibration_results=calibration_results,
            catchment_area=catchment_area,
        )
        st.markdown(report_content)
        report_gen.download_report(report_content, "calibration_report.md")


if __name__ == "__main__":
    main()
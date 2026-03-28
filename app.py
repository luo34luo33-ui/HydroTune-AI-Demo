"""
Hydromind-Demo 流域水文模型智能率定系统
Streamlit 主入口
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from src.llm_api import call_minimax
from src.data_agent import clean_data_with_sandbox, infer_timestep, get_timestep_info
from src.hydro_calc import compare_all_models, calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias
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
RECOMMENDED_MODELS = ['Tank水箱模型', 'HBV模型', '新安江模型']
SKIP_MODELS = ['HBV模型(完整版)']  # 完整版HBV模型暂不支持小时尺度，使用简化版

with st.sidebar:
    st.header("📁 数据上传")
    uploaded_file = st.file_uploader(
        "上传水文数据文件",
        type=["csv", "xlsx", "xls"],
        help="支持 CSV、Excel 格式，系统将自动识别列名",
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
    
    st.caption("注：完整版HBV模型(需日尺度)，简化版支持小时尺度")

    st.divider()

    st.header("📐 未来扩展")
    st.info(
        """
    **预留接口支持：**
    - DEM数据 (GeoTIFF)
    - 土地利用数据
    - 流域边界 (Shapefile)
    - 分布式模型
    """
    )


# ============================================================
# 主流程
# ============================================================
if uploaded_file is not None:

    # ---- 阶段1：读取数据 ----
    with st.status("📥 正在读取上传数据...", expanded=True) as status:
        st.write("解析文件格式...")

        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.write(f"✅ 读取成功，共 {raw_df.shape[0]} 行，{raw_df.shape[1]} 列")
            st.write(f"列名: {list(raw_df.columns)}")

            status.update(label="✅ 数据读取完成", state="complete")
        except Exception as e:
            st.error(f"数据读取失败: {e}")
            st.stop()

    # 显示原始数据预览
    with st.expander("📊 查看原始数据"):
        st.dataframe(raw_df.head(20))

    # ---- 阶段2：智能数据清洗 ----
    with st.status("🧠 数据智能体正在分析数据格式...", expanded=True) as status:
        st.write("提取数据指纹...")
        time.sleep(0.5)  # 仪式感

        st.write("调用 LLM 生成清洗规则...")
        time.sleep(0.5)

        # 执行数据清洗
        try:
            clean_df, detected_timestep = clean_data_with_sandbox(raw_df, call_minimax)
            st.write("✅ 数据清洗完成，已标准化为: date, precip, evap, flow")
        except Exception as e:
            st.error(f"数据清洗失败: {e}")
            st.stop()

        status.update(label="✅ 数据清洗完成", state="complete")

    # 显示清洗后的数据
    with st.expander("📊 查看清洗后数据"):
        st.dataframe(clean_df.head(20))

    # 检查数据有效性
    if "precip" not in clean_df.columns or "flow" not in clean_df.columns:
        st.error("数据清洗后仍缺少必要列 (precip/flow)，请检查原始数据")
        st.stop()

    # 填充缺失值
    clean_df = clean_df.fillna(0)

    # ---- 阶段2.5：时间尺度确认 ----
    st.divider()
    st.subheader("⏱️ 时间尺度确认")

    timestep_info = get_timestep_info(detected_timestep)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"🔍 检测到数据为 **{timestep_info['label']}** ({timestep_info['description']})")
    with col2:
        user_timestep = st.radio(
            "请确认时间尺度：",
            options=['hourly', 'daily'],
            index=0 if detected_timestep == 'hourly' else 1,
            horizontal=True,
            label_visibility="collapsed"
        )

    if user_timestep != detected_timestep:
        st.warning(f"⚠️ 已切换为 **{'小时' if user_timestep == 'hourly' else '日'}尺度**，单位转换已调整")

    # 提取数据数组
    precip = np.array(clean_df["precip"].values)
    evap = np.array(clean_df["evap"].values)
    observed = np.array(clean_df["flow"].values)

    # 确定需要率定的模型列表
    models_to_calibrate = []
    models_skipped = []

    for model_name in RECOMMENDED_MODELS:
        if model_name in SKIP_MODELS and user_timestep == 'hourly':
            models_skipped.append(model_name)
            continue
        models_to_calibrate.append(model_name)

    if models_skipped:
        st.warning(f"⚠️ 以下模型暂不支持小时尺度，已跳过: {', '.join(models_skipped)}")

    # ---- 阶段3：多模型率定 ----
    st.divider()
    st.subheader("🌊 多模型自动率定")

    st.write(f"数据长度: {len(precip)} 个时间步 ({'小时' if user_timestep == 'hourly' else '日'}尺度)")
    st.write(f"正在率定 {len(models_to_calibrate)} 个模型（两阶段快速算法）...")

    def calibrate_single_model(model_name):
        try:
            return calibrate_model_fast(
                model_name, precip, evap, observed,
                max_iter=max_iter,
                timestep=user_timestep
            )
        except Exception as e:
            return None

    with st.status("🌊 启动多模型并发率定...", expanded=True) as status:
        start_time = time.time()

        results = []
        with ThreadPoolExecutor(max_workers=len(models_to_calibrate)) as executor:
            futures = {executor.submit(calibrate_single_model, name): name for name in models_to_calibrate}
            progress_bar = st.progress(0)
            
            for i, future in enumerate(as_completed(futures)):
                model_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        params, nse, simulated = result
                        results.append({
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(observed, simulated),
                            "mae": calc_mae(observed, simulated),
                            "pbias": calc_pbias(observed, simulated),
                            "simulated": simulated,
                        })
                        st.write(f"  ✅ {model_name}: NSE = {nse:.4f}")
                except Exception as e:
                    st.write(f"  ❌ {model_name}: {e}")
                
                progress_bar.progress((i + 1) / len(RECOMMENDED_MODELS))
        
        results.sort(key=lambda x: x["nse"], reverse=True)
        
        elapsed = time.time() - start_time
        st.write(f"⏱️ 率定完成，耗时 {elapsed:.1f} 秒")

        status.update(label="✅ 率定完成", state="complete")

    # ---- 阶段4：结果展示 ----
    st.divider()
    st.subheader("📈 模拟结果对比")

    # 指标对比表
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**性能指标对比**")
        metrics_df = pd.DataFrame(
            [
                {
                    "模型": r["model_name"],
                    "NSE": f"{r['nse']:.4f}",
                    "RMSE": f"{r['rmse']:.4f}",
                    "MAE": f"{r['mae']:.4f}",
                    "PBIAS": f"{r['pbias']:.2f}%",
                }
                for r in results
            ]
        )
        st.dataframe(metrics_df, width='stretch', hide_index=True)

    with col2:
        st.write("**最优模型**")
        best = results[0]
        st.success(f"🏆 {best['model_name']} (NSE = {best['nse']:.4f})")

        st.write("**最优参数**")
        for param, value in best["params"].items():
            st.write(f"- {param}: {value:.6f}")

    # 流量过程线图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(observed, "k-", label="实测流量", linewidth=1.5, alpha=0.8)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, r in enumerate(results):
        ax.plot(
            r["simulated"],
            color=colors[i % len(colors)],
            label=f"{r['model_name']} (NSE={r['nse']:.3f})",
            linewidth=1,
            alpha=0.7,
        )

    ax.set_xlabel("时间步")
    ax.set_ylabel("流量 (m3/s)")
    ax.set_title("流量模拟结果对比")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # 降水过程线
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.bar(range(len(precip)), precip, color="#3498db", alpha=0.7, width=1)
    ax1.set_ylabel("降水 (mm)")
    ax1.set_title("降水过程")
    ax1.invert_yaxis()  # 降水向下
    ax1.grid(True, alpha=0.3)

    ax2.plot(observed, "k-", label="实测", linewidth=1.5)
    ax2.plot(
        results[0]["simulated"],
        "r-",
        label=f"最优模型 ({results[0]['model_name']})",
        linewidth=1,
    )
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("流量 (m3/s)")
    ax2.set_title("流量过程")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

    # ---- 阶段5：LLM 生成报告 ----
    st.divider()

    with st.status("📝 正在生成智能分析报告...", expanded=True) as status:
        # 构造结果摘要
        results_summary = "\n".join(
            [
                f"- {r['model_name']}: NSE={r['nse']:.4f}, RMSE={r['rmse']:.4f}, "
                f"MAE={r['mae']:.4f}, PBIAS={r['pbias']:.2f}%"
                for r in results
            ]
        )

        best_params_str = "\n".join(
            [f"  - {k}: {v:.6f}" for k, v in results[0]["params"].items()]
        )

        report_prompt = f"""你是一位资深水文专家。请基于以下率定结果，撰写一份专业的模型比较报告。

**数据概况：**
- 数据长度: {len(precip)} 个时间步
- 平均降水: {np.nanmean(precip):.2f} mm
- 平均流量: {np.nanmean(observed):.2f} m3/s

**率定结果：**
{results_summary}

**最优模型 ({results[0]['model_name']}) 的参数：**
{best_params_str}

**要求：**
1. 只基于上述数据说话，不要编造不存在的数据
2. 分析各模型的适用性，解释NSE值的含义
3. 给出推荐结论，说明哪个模型最适合该流域
4. 对最优模型的参数进行物理解释
5. 使用 Markdown 格式输出，包含适当的标题和列表
6. 报告要专业但通俗易懂
"""

        report = call_minimax(report_prompt)

        # 检查是否出错
        if report.startswith("[ERROR]"):
            st.warning(f"LLM报告生成失败: {report}")
            # 使用备用报告
            report = f"""
## 模型比较分析报告

### 一、数据概况
- 数据长度: {len(precip)} 个时间步
- 平均降水: {np.nanmean(precip):.2f} mm
- 平均流量: {np.nanmean(observed):.2f} m3/s

### 二、率定结果

| 模型 | NSE | RMSE | MAE | PBIAS |
|------|-----|------|-----|-------|
"""
            for r in results:
                report += f"| {r['model_name']} | {r['nse']:.4f} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['pbias']:.2f}% |\n"

            report += f"""
### 三、结论

**推荐模型: {results[0]['model_name']}**

该模型在本次率定中表现最优，NSE值为 {results[0]['nse']:.4f}，表明模型能够解释 {results[0]['nse']*100:.1f}% 的流量变化。

**最优参数:**
"""
            for k, v in results[0]["params"].items():
                report += f"- {k}: {v:.6f}\n"

        status.update(label="✅ 报告生成完成", state="complete")

    st.subheader("📋 智能分析报告")
    st.markdown(report)

else:
    # 欢迎页面
    st.info("👈 请在左侧上传水文数据文件开始分析")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("率定模型", f"{len(RECOMMENDED_MODELS)} 个")

    with col2:
        st.metric("支持格式", "CSV / Excel")

    with col3:
        st.metric("率定算法", "两阶段(并行)")

    st.divider()

    st.subheader("📖 使用说明")
    st.markdown(
        """
    1. **上传数据**：在左侧上传 CSV 或 Excel 格式的水文数据
    2. **智能清洗**：系统自动识别列名并标准化数据格式
    3. **自动率定**：对所有注册模型进行参数率定
    4. **结果对比**：查看各模型性能指标和流量过程线
    5. **智能报告**：自动生成专业的模型比较分析报告
    """
    )

    st.subheader("📊 数据格式要求")
    st.markdown(
        """
    数据应包含以下列（列名支持中英文）：
    - **时间列**: date / time / 日期 / 时间
    - **降水列**: precip / rainfall / p / 降水 / 降雨
    - **蒸发列**: evap / et / 蒸发 (可选)
    - **流量列**: flow / discharge / q / 流量 / 径流
    """
    )

"""
HydroTune-AI Intelligent Watershed Hydrological Model Calibration System
Streamlit Main Entry (English Version)
"""
import streamlit as st
from src.app.routing import render_routing_settings, validate_upstream_columns, simulate_imported_model
from src.app.plotting import PLOT_DPI, EVENTS_PER_FIGURE, create_event_figure, render_figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from typing import List, Dict, Tuple
from io import StringIO

from src.llm_api import call_minimax, set_language
# Set English language for reports, plots, and LLM prompts
set_language('en')
from src.data_agent import (
    clean_data_with_sandbox, infer_timestep, infer_timestep_by_llm, get_timestep_info,
    detect_flood_events, FloodEvent
)
from src.hydro_calc import (
    calibrate_model_fast, calc_nse, calc_rmse, calc_mae, calc_pbias, calc_kge,
    get_model_param_info, generate_param_table, apply_configured_routing
)
from src.data_preanalysis import (
    DataPreAnalyzer, PreAnalysisResult, FloodEvent as PAFloodEvent,
    DataQualityResult, FrequencyAnalysisResult
)
from src.flood_semantic_encoder import FloodSemanticCode, FloodSemanticEncoder
from src.llm_preanalyzer import SemanticPreAnalyzer, DataQualityMetrics
from src.date_parser import parse_dates, infer_timestep as infer_timestep_from_dates
from src.llm_reporter import (
    generate_preanalysis_report, generate_calibration_report,
    generate_comprehensive_report, generate_multifile_report
)
from src.models.registry import ModelRegistry
from src.bma_ensemble import (
    calc_bma_weights, apply_bma_ensemble, calc_bma_metrics,
    format_weights_string
)
from src.app.error_correction import (
    ErrorCorrector, select_best_model, apply_error_correction
)
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="HydroTune",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up font support (for Chinese data display compatibility)
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# Try to load simsun.ttc font from project directory
font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simsun.ttc")

if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    chinese_font = prop.get_name()
    print(f"[HydroTune-AI] Loaded font: {chinese_font}")
else:
    # Fallback to system fonts
    import platform
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chinese_fonts = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "PingFang SC"]
    chinese_font = next((f for f in chinese_fonts if f in available_fonts), "sans-serif")
    print(f"[HydroTune-AI] Using system font: {chinese_font}")

plt.rcParams["font.sans-serif"] = [chinese_font]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# Page State Initialization
# ============================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'



def render_copyable_report(report_text: str, key: str, height: int = 420):
    """Render a report as a copyable Markdown text box."""
    st.text_area(
        "Copyable Markdown Report",
        value=report_text or "",
        height=height,
        key=key,
    )


def format_quality_level_en(quality) -> str:
    """Return an English quality label from quality metrics."""
    raw_level = str(getattr(quality, "quality_level", "")).strip()
    level_map = {
        "Excellent": "Excellent",
        "Good": "Good",
        "Fair": "Fair",
        "Poor": "Poor",
        "优秀": "Excellent",
        "良好": "Good",
        "一般": "Fair",
        "较差": "Poor",
    }
    if raw_level in level_map:
        return level_map[raw_level]

    if quality.completeness >= 95 and quality.continuity >= 95 and quality.correlation >= 0.5:
        return "Excellent"
    if quality.completeness >= 90 and quality.continuity >= 90 and quality.correlation >= 0.3:
        return "Good"
    if quality.completeness >= 80:
        return "Fair"
    return "Poor"


def mean_valid_metric(values) -> float:
    """Average metrics while excluding NaN/inf and failure sentinels."""
    valid_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value) and numeric_value > -900:
            valid_values.append(numeric_value)
    return float(np.mean(valid_values)) if valid_values else np.nan

# ============================================================
# Page definition functions
# ============================================================
def show_models_page():
    """Hydrological Model Introduction Page"""
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
        <h1>🌊 Hydrological Models</h1>
        <p>Conceptual hydrological models are essential tools for simulating watershed hydrological processes</p>
    </div>
    "    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="models_back_1", width="stretch"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    <div class="model-detail-card">
        <div class="model-header">
            <span class="model-icon">🏗️</span>
            <span class="model-title">Tank Model</span>
        </div>
        <p style="color: #64748b; line-height: 1.8;">
            The Tank Model is a conceptual hydrological model proposed by Japanese scholar Sugawara. It abstracts the watershed's storage and regulation effects into multiple series or parallel tanks to simulate the rainfall-runoff transformation process. The model has a simple structure, few parameters, and wide applicability.
        </p>
        <h4 style="color: #1e293b; margin-top: 20px;">Model Parameters</h4>
        <table class="param-table">
            <tr><th>Parameter</th><th>Description</th><th>Unit</th><th>Typical Range</th></tr>
            <tr><td><b>k1</b></td><td>Fast flow storage coefficient</td><td>-</td><td>0.01 ~ 0.3</td></tr>
            <tr><td><b>k2</b></td><td>Slow flow storage coefficient (baseflow)</td><td>-</td><td>0.001 ~ 0.05</td></tr>
            <tr><td><b>c</b></td><td>Runoff coefficient</td><td>-</td><td>0.01 ~ 0.3</td></tr>
        </table>
        <h4 style="color: #1e293b; margin-top: 20px;">Applicable Scenarios</h4>
        <p style="color: #64748b;">General watershed simulation, especially suitable for data-scarce regions or rapid analysis needs.</p>
    </div>
    
    <div class="model-detail-card">
        <div class="model-header">
            <span class="model-icon">🏔️</span>
            <span class="model-title">HBV Model</span>
        </div>
        <p style="color: #64748b; line-height: 1.8;">
            The HBV model is a conceptual hydrological model developed by the Swedish Meteorological and Hydrological Institute (SMHI). It consists of four modules: soil moisture accounting, evapotranspiration, runoff generation, and routing. It is widely used in Nordic countries and watersheds around the world.
        </p>
        <h4 style="color: #1e293b; margin-top: 20px;">Model Parameters</h4>
        <table class="param-table">
            <tr><th>Parameter</th><th>Description</th><th>Unit</th><th>Typical Range</th></tr>
            <tr><td><b>fc</b></td><td>Field capacity</td><td>mm</td><td>50 ~ 500</td></tr>
            <tr><td><b>beta</b></td><td>Shape parameter</td><td>-</td><td>1.0 ~ 5.0</td></tr>
            <tr><td><b>k0</b></td><td>Fast runoff coefficient</td><td>-</td><td>0.01 ~ 0.5</td></tr>
            <tr><td><b>k1</b></td><td>Slow runoff coefficient</td><td>-</td><td>0.001 ~ 0.1</td></tr>
            <tr><td><b>lp</b></td><td>Evapotranspiration limit coefficient</td><td>-</td><td>0.3 ~ 1.0</td></tr>
        </table>
        <h4 style="color: #1e293b; margin-top: 20px;">Applicable Scenarios</h4>
        <p style="color: #64748b;">Humid and semi-humid watersheds, particularly suitable for Nordic, North American and similar regions.</p>
    </div>
    
    <div class="model-detail-card">
        <div class="model-header">
            <span class="model-icon">🌊</span>
            <span class="model-title">Xinanjiang Model (XAJ)</span>
        </div>
        <p style="color: #64748b; line-height: 1.8;">
            The Xinanjiang (XAJ) model is a three-source conceptual hydrological model proposed by Chinese hydrologist Zhao Renjun et al. in the 1980s. Based on the saturation-excess runoff mechanism, it divides runoff into surface flow, interflow, and groundwater flow. It is the most widely applied watershed hydrological model in China's humid regions.
        </p>
        <h4 style="color: #1e293b; margin-top: 20px;">Model Parameters</h4>
        <table class="param-table">
            <tr><th>Parameter</th><th>Description</th><th>Unit</th><th>Typical Range</th></tr>
            <tr><td><b>k</b></td><td>Evapotranspiration coefficient</td><td>-</td><td>0.5 ~ 1.5</td></tr>
            <tr><td><b>b</b></td><td>Water storage capacity curve exponent</td><td>-</td><td>0.1 ~ 0.5</td></tr>
            <tr><td><b>im</b></td><td>Impervious area ratio</td><td>-</td><td>0.01 ~ 0.1</td></tr>
            <tr><td><b>um</b></td><td>Upper soil water storage capacity</td><td>mm</td><td>10 ~ 50</td></tr>
            <tr><td><b>lm</b></td><td>Lower soil water storage capacity</td><td>mm</td><td>50 ~ 150</td></tr>
            <tr><td><b>dm</b></td><td>Deep soil water storage capacity</td><td>mm</td><td>10 ~ 100</td></tr>
            <tr><td><b>c</b></td><td>Deep evapotranspiration coefficient</td><td>-</td><td>0.01 ~ 0.2</td></tr>
            <tr><td><b>sm</b></td><td>Free water storage capacity</td><td>mm</td><td>10 ~ 80</td></tr>
            <tr><td><b>ex</b></td><td>Free water capacity curve exponent</td><td>-</td><td>1.0 ~ 2.0</td></tr>
            <tr><td><b>ki</b></td><td>Interflow outflow coefficient</td><td>-</td><td>0.3 ~ 0.7</td></tr>
            <tr><td><b>kg</b></td><td>Groundwater outflow coefficient</td><td>-</td><td>0.01 ~ 0.2</td></tr>
            <tr><td><b>cs</b></td><td>Channel routing coefficient</td><td>-</td><td>0.1 ~ 0.5</td></tr>
            <tr><td><b>l</b></td><td>Lag time</td><td>h</td><td>0 ~ 24</td></tr>
            <tr><td><b>xg</b></td><td>Groundwater recession coefficient</td><td>-</td><td>0.9 ~ 0.999</td></tr>
        </table>
        <h4 style="color: #1e293b; margin-top: 20px;">Applicable Scenarios</h4>
        <p style="color: #64748b;">Humid regions of China, particularly the Yangtze, Pearl, and Huai River basins.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    if st.button("← Back to Home", key="models_back_2", width="stretch"):
        st.session_state.current_page = 'main'
        st.rerun()

# ============================================================
# Title Area
# ============================================================
st.title("HydroTune")
st.caption("an LLM-agent platform for accurate hydrological simulation through multi-model calibration and physics-informed error correction")

st.markdown("""
### 🌟 Features

**HydroTune** is an LLM-agent platform for accurate hydrological simulation through multi-model calibration and physics-informed error correction that supports:

- 📊 **Multi-Model Integration**: HBV, Xinanjiang (XAJ), Tank Model
- 🔄 **Smart Calibration**: Automatic model parameter calibration with multiple optimization algorithms
- 🎯 **XGBoost Error Correction**: Post-processing correction of model simulation results for improved accuracy
- 📈 **Data Pre-Analysis**: Automatic flood event identification and data quality analysis
- 🤖 **AI-Powered Reports**: Automated generation of analysis reports

### 📋 Workflow

1. **Upload Data**: Select upload mode in the sidebar (single file / multi-file)
2. **Configure Parameters**: Set warm-up period, calibration parameters, XGB options
3. **Start Analysis**: Click the "🚀 Start Analysis" button
4. **View Results**: Review calibration results, error correction, and AI reports
""")

# ============================================================
# 侧边栏
# ============================================================
RECOMMENDED_MODELS = ['HBV Model', 'XAJ Model', 'Tank Model']
SKIP_MODELS = []




with st.sidebar:
    st.markdown("### 🤖 AI Agent Status")
    agent_status = st.empty()
    agent_status.success("🟢 AI Agent Ready")
    
    # Model loading status check
    st.markdown("### 📊 Model Status")
    from src.models.registry import ModelRegistry
    all_models = ModelRegistry.list_models()
    check_models = ['HBV Model', 'XAJ Model', 'Tank Model']
    for model in check_models:
        if model in all_models:
            st.success(f"✅ {model}")
        else:
            st.error(f"❌ {model}")
    
    st.divider()
    st.markdown("### 📂 Navigation")
    if st.button("🏠 Home", width="stretch"):
        st.session_state.current_page = 'main'
    if st.button("📚 Hydrological Models", width="stretch"):
        st.session_state.current_page = 'models'
    
    st.divider()
    st.header("📁 Data Upload")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Hydrological Data File",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Supports multiple file uploads",
        )
    with col2:
        catchment_area = st.number_input(
            "Catchment Area",
            min_value=0.1,
            max_value=100000.0,
            value=150.7944,
            step=1.0,
            format="%.4f",
            help="Unit: km²",
        )

    upload_mode = st.radio(
        "📂 Upload Mode",
        options=[
            "Single File (One Flood Event)",      # single event, full analysis
            "Single File (Continuous Series)",    # continuous series, detect multiple events
            "Multi-File (One Event Per File)"     # one event per file
        ],
        index=0,
        horizontal=False,
    )

    st.header("⏱️ Warm-up Period")
    has_warmup = st.radio(
        "Warm-up Period",
        options=["None", "Yes"],
        index=0,
        help="Warm-up period is the time window to discard before the model starts responding"
    )
    if has_warmup == "Yes":
        warmup_hours = st.number_input(
            "Warm-up Length (h)",
            min_value=0,
            max_value=720,
            value=24,
            step=1,
            help="Length of warm-up period to discard"
        )
    else:
        warmup_hours = 0

    st.divider()
    
    # Import parameters mode
    st.header("📥 Import Parameters")
    use_imported_params = st.radio(
        "Parameter Mode",
        options=["Calibrate Parameters", "Import Parameters"],
        index=0,
        horizontal=True,
        help="Select 'Import Parameters' to use parameter files directly without calibration"
    )
    
    imported_params = {}
    
    if use_imported_params == "Import Parameters":
        st.session_state.pop('imported_k_routing', None)
        st.session_state.pop('imported_x_routing', None)
        st.success("📥 Import Mode: Using parameter files directly")
        
        st.markdown("**Tank Model Parameters:**")
        tank_param_file = st.file_uploader(
            "Upload Tank Model Parameter File",
            type=["csv"],
            key="tank_param_file"
        )
        if tank_param_file is not None:
            try:
                import pandas as pd
                import io
                
                file_content = tank_param_file.getvalue()
                if not file_content or len(file_content.strip()) == 0:
                    raise ValueError("File is empty, please check the uploaded file")
                
                tank_df = None
                for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                    try:
                        tank_df = pd.read_csv(io.BytesIO(file_content), encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if tank_df is None:
                    raise ValueError("Unable to decode file, please save as UTF-8 encoding")
                if len(tank_df.columns) == 0:
                    raise ValueError("CSV file has no columns, please check file format")
                if len(tank_df) == 0:
                    raise ValueError("CSV file has no data rows, please check file content")
                
                tank_params = {
                    col: float(tank_df[col].values[0])
                    for col in tank_df.columns
                    if col != '模型'
                }
                imported_params['Tank Model'] = tank_params
                if 'k_routing' in tank_params and 'x_routing' in tank_params:
                    st.session_state['imported_k_routing'] = tank_params['k_routing']
                    st.session_state['imported_x_routing'] = tank_params['x_routing']
                st.success(f"✅ Tank model parameters imported: {tank_params}")
            except Exception as e:
                st.error(f" Tank parameter parse failed: {e}")
        
        st.markdown("**HBV Model Parameters:**")
        hbv_param_file = st.file_uploader(
            "Upload HBV Model Parameter File",
            type=["csv"],
            key="hbv_param_file"
        )
        if hbv_param_file is not None:
            try:
                import pandas as pd
                hbv_df = pd.read_csv(hbv_param_file)
                hbv_params = {col: float(hbv_df[col].values[0]) for col in hbv_df.columns if col != '模型'}
                imported_params['HBV Model'] = hbv_params
                if 'k_routing' in hbv_params and 'x_routing' in hbv_params:
                    st.session_state['imported_k_routing'] = hbv_params['k_routing']
                    st.session_state['imported_x_routing'] = hbv_params['x_routing']
                st.success(f"✅ HBV model parameters imported: {hbv_params}")
            except Exception as e:
                st.error(f" HBV parameter parse failed: {e}")
        
        st.markdown("**Xinanjiang (XAJ) Model Parameters (incl. Muskingum):**")
        xaj_param_file = st.file_uploader(
            "Upload XAJ Model Parameter File",
            type=["csv"],
            key="xaj_param_file"
        )
        if xaj_param_file is not None:
            try:
                import pandas as pd
                import io
                
                file_content = xaj_param_file.getvalue()
                
                xaj_df = None
                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                    try:
                        xaj_df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                        break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue
                
                if xaj_df is None or xaj_df.empty:
                    st.warning("⚠️ CSV file is empty, please check file format")
                elif len(xaj_df.columns) < 2:
                    st.warning(f"⚠️ CSV file has insufficient columns: {xaj_df.columns}")
                else:
                    xaj_params = {col: float(xaj_df[col].values[0]) for col in xaj_df.columns if col != '模型'}
                    imported_params['XAJ Model'] = xaj_params
                    st.success(f"✅ XAJ model parameters imported: {xaj_params}")
                    
                    if 'k_routing' in xaj_df.columns and 'x_routing' in xaj_df.columns:
                        st.session_state['imported_k_routing'] = float(xaj_df['k_routing'].values[0])
                        st.session_state['imported_x_routing'] = float(xaj_df['x_routing'].values[0])
                        st.success(f"✅ Muskingum parameters imported: k={st.session_state['imported_k_routing']}, x={st.session_state['imported_x_routing']}")
            except Exception as e:
                st.error(f" XAJ parameter parse failed: {e}")

    # Column name configuration
    st.header("📋 Column Mapping")
    with st.expander("Configure Column Name Mapping"):
        st.write("Enter original data column names (mapped to standard names: date, precip, evap, flow, upstream):")
        
        date_col = st.text_input("Date Column", value="date", key="date_col")
        precip_col = st.text_input("Precipitation Column", value="precip", key="precip_col")
        evap_col = st.text_input("Evaporation Column", value="evap", key="evap_col")
        flow_col = st.text_input("Flow Column", value="flow", key="flow_col")
    
    column_mapping = {
        'date': date_col if date_col else 'date',
        'precip': precip_col if precip_col else 'precip',
        'evap': evap_col if evap_col else 'evap',
        'flow': flow_col if flow_col else 'flow',
        'upstream': "",
    }

    st.divider()

    st.header("🌊 Upstream Routing")
    enable_upstream_routing = st.checkbox(
        "Enable Upstream Routing",
        value=False,
        help="When enabled, uses the Muskingum method to route upstream inflow and add it to the outlet discharge"
    )
    
    if enable_upstream_routing:
        upstream_col = st.text_input(
            "Upstream Outflow Column Name",
            value="",
            help="Column name for upstream discharge (data must be in the same file)"
        )
        column_mapping['upstream'] = upstream_col if upstream_col else ""
        
        manual_routing_params = render_routing_settings(
            import_mode=use_imported_params == "Import Parameters", english=True,
        )
    else:
        column_mapping['upstream'] = ""
        manual_routing_params = None

    st.divider()
    
    st.header("🔧 XGBoost Error Correction")
    enable_xgb_correction = st.checkbox(
        "Enable XGBoost Error Correction",
        value=True,
        help="When enabled, applies error correction to hydrological model simulation results"
    )
    
    if enable_xgb_correction:
        with st.expander("XGBoost Parameter Settings"):
            xgb_max_depth = st.slider(
                "max_depth (Tree Depth)",
                min_value=3,
                max_value=15,
                value=10,
                help="Maximum depth of XGBoost decision trees"
            )
            xgb_learning_rate = st.number_input(
                "learning_rate",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="XGBoost learning rate"
            )
            xgb_n_estimators = st.slider(
                "n_estimators (Number of Trees)",
                min_value=50,
                max_value=500,
                value=300,
                step=50,
                help="Number of XGBoost decision trees"
            )
            xgb_random_seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=9999,
                value=2025,
                step=1,
                help="XGBoost random seed for reproducible results"
            )
            xgb_enable_error_clipping = st.checkbox(
                "Limit Predicted Error Range",
                value=True,
                help="Clip XGB-predicted residuals using training residual bounds to avoid extreme correction spikes"
            )
            if xgb_enable_error_clipping:
                xgb_clip_quantile = st.slider(
                    "Residual Quantile Range",
                    min_value=0.80,
                    max_value=0.98,
                    value=0.90,
                    step=0.02,
                    help="Central fraction of training residuals used as prediction-error bounds"
                )
                xgb_max_correction_ratio = st.slider(
                    "Max Correction Ratio",
                    min_value=0.10,
                    max_value=1.00,
                    value=0.45,
                    step=0.05,
                    help="Maximum absolute correction as a fraction of the current simulated flow"
                )
            else:
                xgb_clip_quantile = 1.0
                xgb_max_correction_ratio = 1.0
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': xgb_max_depth,
            'learning_rate': xgb_learning_rate,
            'n_estimators': xgb_n_estimators,
            'random_state': int(xgb_random_seed),
        }
    else:
        xgb_params = None
        xgb_random_seed = 2025
        xgb_enable_error_clipping = True
        xgb_clip_quantile = 0.90
        xgb_max_correction_ratio = 0.45

    st.divider()

    st.markdown("""
⚙️ Calibration Settings 
<span style="font-size: 14px; color: #888; cursor: help;" title="• Two-Stage (Recommended): Fast global search + local fine optimization
• PSO: Particle Swarm Optimization, suitable for large-scale problems
• SCE-UA: Shuffled Complex Evolution, strong global optimization
• DE: Differential Evolution, simple and efficient
• GA: Genetic Algorithm, maintains diversity during evolution">❓</span>
""", unsafe_allow_html=True)

    algorithm = st.selectbox(
        "Optimization Algorithm",
        options=["Two-Stage (Recommended)", "PSO", "SCE-UA", "Differential Evolution (DE)", "Genetic Algorithm (GA)"],
        index=0,
        help="Select the optimization algorithm for calibration"
    )

    max_iter = st.slider(
        "Max Iterations",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Number of optimization iterations; more iterations yield better results but take longer"
    )

    if algorithm == "PSO":
        st.subheader("Particle Swarm Optimization (PSO) Parameters")
        n_particles = st.slider("Number of Particles", 10, 100, 100, help="More particles improve search capability but increase computation time")
        w = st.slider("Inertia Weight w", 0.0, 1.0, 0.7, 0.01, help="Controls particle inertia; larger values expand search range")
        c1 = st.slider("Cognitive Coefficient c1", 0.0, 2.0, 1.5, 0.1, help="Controls movement toward personal best position")
        c2 = st.slider("Social Coefficient c2", 0.0, 2.0, 1.5, 0.1, help="Controls movement toward global best position")
        algo_params = {"n_particles": n_particles, "w": w, "c1": c1, "c2": c2}
    elif algorithm == "Genetic Algorithm (GA)":
        st.subheader("Genetic Algorithm (GA) Parameters")
        pop_size = st.slider("Population Size", 10, 100, 50, help="Larger population improves genetic diversity")
        n_generations = st.slider("Number of Generations", 10, 100, 50, help="More generations lead to more thorough optimization")
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8, 0.05, help="Probability of chromosome crossover to produce new individuals")
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1, 0.05, help="Probability of gene mutation to prevent local optima")
        algo_params = {"pop_size": pop_size, "n_generations": n_generations, 
                       "crossover_rate": crossover_rate, "mutation_rate": mutation_rate}
    elif algorithm == "SCE-UA":
        st.subheader("SCE-UA Parameters")
        n_complexes = st.slider("Number of Complexes", 2, 10, 5, help="More complexes improve global search capability")
        points_per_complex = st.slider("Points per Complex", 5, 20, 10, help="Number of points in each complex")
        algo_params = {"n_complexes": n_complexes, "points_per_complex": points_per_complex}
    elif algorithm == "Differential Evolution (DE)":
        st.subheader("Differential Evolution (DE) Parameters")
        mutation_factor = st.slider("Mutation Factor F", 0.0, 2.0, 0.8, 0.1, help="Scaling factor for differential mutation, recommended 0.5-1.0")
        crossover_prob = st.slider("Crossover Probability CR", 0.0, 1.0, 0.7, 0.1, help="Crossover probability; higher values increase crossover frequency")
        pop_size_de = st.slider("Population Size", 10, 100, 50, help="Larger population improves search capability")
        algo_params = {"mutation_factor": mutation_factor, "crossover_prob": crossover_prob, 
                       "pop_size": pop_size_de}
    else:
        algo_params = {}

    st.divider()

    st.header("📊 Calibration Models")
    for model_name in RECOMMENDED_MODELS:
        st.write(f"✅ {model_name}")
    


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
        <div class="hero-title">HydroTune</div>
        <div class="hero-subtitle">an LLM-agent platform for accurate hydrological simulation through multi-model calibration and physics-informed error correction</div>
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
        if st.button("📚 Hydrological Models", type="secondary", width="stretch"):
            st.session_state.current_page = 'models'
            st.rerun()
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #1e293b; font-weight: 700;">Core Capabilities</h2>
        <p style="color: #64748b;">an LLM-agent platform for accurate hydrological simulation through multi-model calibration and physics-informed error correction</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Intelligent Data Understanding</div>
            <div class="feature-desc">
                LLM-driven automatic data comprehension<br>
                • Intelligent timestep recognition<br>
                • Data quality assessment<br>
                • Anomaly detection
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Flood Event Detection</div>
            <div class="feature-desc">
                Automated flood event analysis<br>
                • Peak shape feature extraction<br>
                • Representative event selection<br>
                • Frequency curve fitting
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Multi-Model Calibration</div>
            <div class="feature-desc">
                Integrated multiple hydrological models<br>
                • Automatic parameter optimization<br>
                • Multi-objective evaluation<br>
                • Comparative result analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    col_models, col_workflow = st.columns([1, 1])
    
    with col_models:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #1e293b;">Supported Hydrological Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("""
            <div class="model-card">
                <div class="model-name">Tank Model</div>
                <div class="model-desc">General Watershed</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div class="model-card">
                <div class="model-name">HBV Model</div>
                <div class="model-desc">Humid/Semi-humid</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div class="model-card">
                <div class="model-name">Xinanjiang</div>
                <div class="model-desc">Humid Regions (China)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_workflow:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #1e293b;">Workflow</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-item">
            <span class="step-num">1</span>
            <span class="step-text">Upload hydrological data (CSV/Excel)</span>
        </div>
        <div class="step-item">
            <span class="step-num">2</span>
            <span class="step-text">Configure column name mapping</span>
        </div>
        <div class="step-item">
            <span class="step-num">3</span>
            <span class="step-text">Set catchment parameters</span>
        </div>
        <div class="step-item">
            <span class="step-num">4</span>
            <span class="step-text">Start intelligent analysis</span>
        </div>
        <div class="step-item" style="border-bottom: none;">
            <span class="step-num">5</span>
            <span class="step-text">View results and reports</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("📋 Data Format Requirements", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **Required Columns:**
            - Time: `date`, `time`
            - Precipitation: `precip`, `rainfall`
            - Flow: `flow`, `discharge`
            
            **Optional Columns:**
            - Evaporation: `evap`, `et`
            """)
        with col2:
            st.markdown("""
            **Supported Formats:**
            - CSV files
            - Excel (.xlsx, .xls)
            - Auto-detect column mapping
            - Multi-format time compatibility
            """)
    
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 30px 0; border-top: 1px solid #e2e8f0;">
        <small>HydroTune v1.0 | LLM-Agent Hydrological Simulation Platform</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload hydrological data files on the left to start using HydroTune")

# ============================================================
# Main Flow
# ============================================================
if uploaded_files and len(uploaded_files) > 0:

    all_results = {}
    all_flood_events = []
    report_sections = []
    
    # ---- Read and cache all file data ----
    file_dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            file_dfs.append((uploaded_file.name, df))
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
            continue
    
    if not file_dfs:
        st.error("No available files")
        st.stop()
    
    # ---- Column name confirmation (first file) ----
    st.divider()
    st.subheader("📋 Configuration Confirmation")
    
    first_file_name, first_df = file_dfs[0]
    st.write(f"**{first_file_name}** columns: `{list(first_df.columns)}`")
    
    col_mapping_accepted = st.checkbox(
        f"Use column mapping: {column_mapping['date']}→date, {column_mapping['precip']}→precip, {column_mapping['evap']}→evap, {column_mapping['flow']}→flow",
        value=True
    )
    
    # ============================================================
    # 📥 Data File Processing (executed on button click)
    # ============================================================
    if st.button("🚀 Start Analysis", type="primary", disabled=not col_mapping_accepted):
        if enable_upstream_routing:
            try:
                validate_upstream_columns(file_dfs, column_mapping['upstream'], english=True)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        
        # ---- Process each file ----
        all_precip = []
        all_evap = []
        all_flow = []
        all_dates = []
        all_file_events = []
        all_upstream_arr = None
        all_upstream = []
        detected_timestep = 'daily'
        user_timestep = 'daily'
        file_summary = []
        preanalysis_file_boundaries = []
        preanalysis_offset = 0
        
        with st.expander("📥 Data Processing Details", expanded=False):
            for file_idx, (file_name, raw_df) in enumerate(file_dfs):
                st.write(f"**📄 {file_name}** - {raw_df.shape[0]} rows")
                
                rename_map = {}
                for std_name, orig_name in column_mapping.items():
                    if orig_name and orig_name in raw_df.columns:
                        rename_map[orig_name] = std_name
                
                if rename_map:
                    raw_df = raw_df.rename(columns=rename_map)
                
                if 'precip' not in raw_df.columns or 'flow' not in raw_df.columns:
                    st.error(f"Missing required columns. Current columns: {list(raw_df.columns)}")
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
                        name="All Data",
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
                if enable_upstream_routing and upstream_arr is not None:
                    all_upstream.extend(upstream_arr.tolist())
                if len(flow_arr) > 0:
                    preanalysis_file_boundaries.append({
                        'start_idx': preanalysis_offset,
                        'end_idx': preanalysis_offset + len(flow_arr) - 1,
                        'peak_idx': preanalysis_offset + int(np.argmax(flow_arr)),
                        'peak_flow': float(np.max(flow_arr)),
                        'file_name': file_name,
                    })
                preanalysis_offset += len(flow_arr)
                
                st.write(f"  ✅ Detected **{len(flood_events)}** flood event(s)")
        
        # File summary
        total_events = sum(f['events'] for f in file_summary)
        total_rows = sum(f['rows'] for f in file_summary)
        st.success(f"📊 Processed {len(file_summary)} file(s), {total_rows} total rows, detected {total_events} flood event(s)")
        
        # Timestep confirmation
        st.divider()
        st.subheader("⏱️ Timestep Confirmation")
        
        first_file_name, first_raw_df = file_dfs[0]
        rename_map_first = {}
        for std_name, orig_name in column_mapping.items():
            if orig_name and orig_name in first_raw_df.columns:
                rename_map_first[orig_name] = std_name
        first_clean_df = first_raw_df.rename(columns=rename_map_first)
        if 'date' not in first_clean_df.columns:
            first_clean_df['date'] = range(len(first_raw_df))
        first_clean_df = first_clean_df.fillna(0)
        
        with st.spinner("🤖 AI Agent is analyzing data timestep..."):
            detected_timestep = infer_timestep_by_llm(first_clean_df['date'], call_minimax)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"🤖 AI Agent detected **hourly** scale" if detected_timestep == 'hourly' else f"🤖 AI Agent detected **daily** scale")
        with col2:
            user_timestep = st.radio(
                "Select",
                options=['hourly', 'daily'],
                index=0 if detected_timestep == 'hourly' else 1,
                horizontal=True,
                label_visibility="collapsed"
            )
        
# ============================================================
        # 🤖 AI Agent Data Pre-Analysis (Semantic Encoding + LLM)
        # ============================================================
        st.divider()
        st.subheader("🧠 AI Agent Data Pre-Analysis")
        
        all_precip_arr = np.array(all_precip)
        all_evap_arr = np.array(all_evap)
        all_flow_arr = np.array(all_flow)
        all_upstream_arr = np.asarray(all_upstream, dtype=float) if enable_upstream_routing else None
        
        dates_for_analysis = parse_dates(first_clean_df['date'])
        
        if len(dates_for_analysis) < len(all_precip):
            dates_for_analysis = pd.date_range(
                start=dates_for_analysis.iloc[0] if not pd.isna(dates_for_analysis.iloc[0]) else '2020-01-01',
                periods=len(all_precip),
                freq='h' if user_timestep == 'hourly' else 'D'
            )
        
        dates_series = pd.Series(dates_for_analysis).reset_index(drop=True)
        
        detected_timestep = infer_timestep_from_dates(dates_series)
        
        preanalyzer = SemanticPreAnalyzer(area=catchment_area, timestep=user_timestep)
        
        is_multi_file_mode = upload_mode == "Multi-File (One Event Per File)"
        analysis_event_boundaries = preanalysis_file_boundaries if is_multi_file_mode else None
        
        with st.spinner("🤖 AI is analyzing flood event semantic encoding..."):
            analysis_result = preanalyzer.analyze(
                all_flow_arr, all_precip_arr, dates_series, call_minimax, 
                multi_file_mode=is_multi_file_mode,
                event_boundaries=analysis_event_boundaries
            )
        
        semantic_codes = analysis_result['semantic_codes']
        quality_metrics = analysis_result['quality']
        quality_level_en = format_quality_level_en(quality_metrics)
        
        st.success(f"✅ AI Agent completed semantic encoding: {len(semantic_codes)} flood event(s)")
        
        with st.expander("📈 Flood Semantic Encoding Details", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Data Completeness", f"{quality_metrics.completeness:.1f}%")
            col2.metric("Time Continuity", f"{quality_metrics.continuity:.1f}%")
            col3.metric("Precip-Runoff Correlation", f"{quality_metrics.correlation:.3f}")
            col4.metric("Quality Level", quality_level_en)
            
            if quality_metrics.issues:
                st.warning("Data issues: " + "; ".join(quality_metrics.issues))
            
            st.divider()
            
            if semantic_codes:
                st.markdown("### Flood Event Semantic Encoding Table")
                codes_data = []
                for code in semantic_codes:
                    codes_data.append({
                        'Event ID': code.event_id,
                        'Magnitude': code.magnitude,
                        'Peak (m³/s)': f"{code.peak_flow:.1f}",
                        'Volume (10⁴m³)': f"{code.flood_volume:.1f}",
                        'Duration': code.duration,
                        'Rise Type': code.rise_type,
                        'Recession Type': code.recession_type,
                        'Peak Shape': code.peak_shape,
                        'Rainfall Driven': code.rainfall_driven,
                        'Representative Score': f"{code.representative_score:.1f}"
                    })
                if codes_data:
                    st.dataframe(pd.DataFrame(codes_data), width="stretch", hide_index=True)
        
        st.divider()
        with st.expander("🤖 LLM Analysis Report", expanded=True):
            if analysis_result['base64_plot']:
                st.markdown("![Flood Analysis Plot](data:image/png;base64," + analysis_result['base64_plot'] + ")")
            
            st.markdown(analysis_result['llm_report'])
        
        preanalysis_result = type('obj', (object,), {
            'quality': quality_metrics,
            'events': semantic_codes,
            'selected_events': semantic_codes[:5] if len(semantic_codes) >= 5 else semantic_codes,
            'frequency': type('obj', (object,), {
                'n_samples': len(semantic_codes),
                'mean': np.mean([c.peak_flow for c in semantic_codes]) if semantic_codes else 0,
                'std': np.std([c.peak_flow for c in semantic_codes]) if semantic_codes else 0,
                'cv': np.std([c.peak_flow for c in semantic_codes]) / np.mean([c.peak_flow for c in semantic_codes]) if semantic_codes and np.mean([c.peak_flow for c in semantic_codes]) > 0 else 0,
                'cs': 0,
                'design_values': {}
            })(),
            'area': catchment_area,
            'timestep': user_timestep
        })()
        
        st.session_state['preanalysis_result'] = preanalysis_result
        
        # ============================================================
        # 🤖 AI Agent Model Calibration
        # ============================================================
        st.divider()
        st.subheader("🧠 AI Agent Model Calibration")
        
        calibration_results = {}
        
        def run_configured_model(model_name, precip, evap, flow, **options):
            if use_imported_params == "Import Parameters":
                if model_name not in imported_params:
                    raise ValueError(f"Missing imported parameters: {model_name}")
                return simulate_imported_model(
                    model_name, precip, evap, flow,
                    params=imported_params[model_name], **options,
                )
            return calibrate_model_fast(model_name, precip, evap, flow, **options)

        def calibrate_model(model_name, precip, evap, flow, upstream=None):
            try:
                spatial_data = {'area': catchment_area}
                manual_routing = manual_routing_params if enable_upstream_routing else None
                return run_configured_model(
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
                    enable_routing=enable_upstream_routing,
                    manual_routing_params=manual_routing
                )
            except Exception as e:
                st.error(f"  ⚠️ {model_name} calibration error: {type(e).__name__}: {str(e)}")
                import traceback
                st.code(traceback.format_exc()[-600:])
                return None
        
# Select calibration strategy based on mode
        if upload_mode == "Multi-File (One Event Per File)":
            use_imported = (use_imported_params == "Import Parameters" and imported_params)
            
            if use_imported:
                st.info(f"📊 Uploaded flood events: {len(file_dfs)} events")
            else:
                st.info(f"📊 Multi-file mode: {len(file_dfs)} files, will be split into calibration and validation sets")
            
            # 1. Prepare all file data
            file_data_list = []
            for file_idx, (file_name, raw_df) in enumerate(file_dfs):
                df = raw_df.copy()
                for std_name, orig_name in column_mapping.items():
                    if orig_name and orig_name in df.columns:
                        df = df.rename(columns={orig_name: std_name})
                
                df['precip'] = df['precip'].fillna(0) if 'precip' in df.columns else pd.Series(np.zeros(len(df)))
                df['flow'] = df['flow'].fillna(0) if 'flow' in df.columns else pd.Series(np.zeros(len(df)))
                df['evap'] = df['evap'].fillna(0) if 'evap' in df.columns else pd.Series(np.zeros(len(df)))
                precip_arr = np.array(df['precip'].values)
                flow_arr = np.array(df['flow'].values)
                evap_arr = np.array(df['evap'].values)
                upstream_arr = np.array(df['upstream'].fillna(0).values) if 'upstream' in df.columns and enable_upstream_routing else None
                
                # Check if data is valid
                precip_sum = np.sum(precip_arr)
                flow_sum = np.sum(flow_arr)
                if precip_sum == 0:
                    st.warning(f"⚠️ {file_name}: Precipitation data is all zeros, please check column name configuration")
                if flow_sum == 0:
                    st.warning(f"⚠️ {file_name}: Flow data is all zeros, please check column name configuration")
                
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
            n_calib = max(1, n_files * 3 // 4)  # 3:1 ratio, i.e., 3/4 for calibration
            indices = np.random.permutation(n_files)
            calib_indices = indices[:n_calib]
            valid_indices = indices[n_calib:]
            
            calib_files = [file_data_list[i] for i in calib_indices]
            valid_files = [file_data_list[i] for i in valid_indices]
            
            if use_imported:
                st.success(f"📊 Uploaded flood events: {len(file_data_list)} events")
                st.write(f"**Flood Events**: {[f['file_name'] for f in file_data_list]}")
            else:
                st.success(f"📊 Grouping complete: {n_calib} calibration + {len(valid_files)} validation (Calib:Valid=3:1)")
                st.write(f"**Calibration Events**: {[f['file_name'] for f in calib_files]}")
                st.write(f"**Validation Events**: {[f['file_name'] for f in valid_files]}")
            
            # Calculate warm-up period steps
            if warmup_hours > 0 and user_timestep == 'hourly':
                warmup_steps = warmup_hours
            elif warmup_hours > 0:
                warmup_steps = warmup_hours // 24
            else:
                warmup_steps = 0
            
            # Build calibration event list (each event independent)
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
            if use_imported:
                st.info(f"📊 Uploaded {len(calib_events)} flood events, ~{avg_steps} time steps each" + 
                       (f", warm-up {warmup_steps} steps" if warmup_steps > 0 else ""))
            else:
                st.info(f"📊 Calibrating {len(calib_events)} events, ~{avg_steps} time steps each" + 
                       (f", warm-up {warmup_steps} steps" if warmup_steps > 0 else ""))
            
            # 4. Calibrate models (multi-event mode)
            import traceback
            progress_bar = st.progress(0)
            
            # Check whether to use imported parameters
            if use_imported_params == "Import Parameters" and imported_params:
                st.info("📥 Using imported parameters, skipping calibration")
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        st.write(f"  ⏭️ Skipping {model_name}")
                        continue
                    if model_name in imported_params:
                        st.write(f"  📥 Using imported parameters for {model_name}...")
                        params = imported_params[model_name].copy()
                        try:
                            params, _, simulated = run_configured_model(
                                model_name, calib_events[0]['precip'], calib_events[0]['evap'],
                                calib_events[0]['flow'], spatial_data={'area': catchment_area},
                                timestep=user_timestep, warmup_steps=warmup_steps,
                                upstream_flow=calib_events[0].get('upstream'),
                                enable_routing=enable_upstream_routing,
                                manual_routing_params=manual_routing_params,
                            )
                            # Exclude warm-up period data for metrics
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
                            st.error(f"  ⚠️ {model_name} runtime error: {type(e).__name__}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc()[-600:])
                    else:
                        st.warning(f"  ⚠️ No imported parameters found for {model_name}")
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
            else:
                st.info("🤖 Model calibration in progress...")
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        st.write(f"  ⏭️ Skipping {model_name}")
                        continue
                    st.write(f"  🔄 Starting calibration for {model_name}...")
                    try:
                        spatial_data = {'area': catchment_area}
                        
                        total_models = len(RECOMMENDED_MODELS)
                        model_base = model_idx / total_models
                        
                        manual_routing = manual_routing_params if enable_upstream_routing else None
                        
                        result = run_configured_model(
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
                            progress_callback=lambda p: progress_bar.progress(model_base + p / total_models),
                            manual_routing_params=manual_routing
                        )
                    except Exception as e:
                        st.error(f"  ⚠️ {model_name} calibration error: {type(e).__name__}: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc()[-600:])
                        result = None
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
                    if result:
                        params, nse, simulated = result
                        # Exclude warm-up period data when computing metrics
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
                        st.write(f"  ✅ {model_name}: Calibration complete")
                    else:
                        st.write(f"  ❌ {model_name}: Calibration returned None")
            
            # 5. Run all events (calibration + validation) with calibrated parameters
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
                    
                    # Use safe parameters for XAJ model
                    if model_name == 'XAJ Model':
                        safe_params = params.copy()
                        # Ensure ki + kg < 0.9
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
                            {'area': catchment_area}  # Simplified spatial_data, only pass area
                        )
                        # Check simulation results are valid
                        if simulated is None or len(simulated) == 0:
                            st.error(f"  ❌ {file_name}/{model_name}: Model returned empty result")
                            raise ValueError(f"Model returned empty result: len={len(simulated) if simulated is not None else 'None'}")
                        
                        # 上游汇流叠加
                        upstream_arr = file_data.get('upstream')
                        if enable_upstream_routing and upstream_arr is not None and len(simulated) > 0:
                            simulated = apply_configured_routing(
                                simulated, upstream_arr, safe_params
                            )
                        # Apply warm-up period handling
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
                            if model_name == 'XAJ Model':
                                fallback_params = default_xaj_params.copy()
                                if enable_upstream_routing:
                                    fallback_params.update({key: safe_params[key] for key in ('k_routing', 'x_routing')})
                                simulated = model.run(
                                    file_data['precip'],
                                    file_data['evap'],
                                    fallback_params,
                                    spatial_data
                                )
                        # Apply upstream routing superposition
                                upstream_arr = file_data.get('upstream')
                                if enable_upstream_routing and upstream_arr is not None:
                                    simulated = apply_configured_routing(
                                        simulated, upstream_arr, safe_params
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
                                        "params": fallback_params,
                                        "nse": -999,
                                        "kge": -999,
                                        "rmse": -999,
                                        "pbias": -999,
                                        "simulated": np.zeros_like(file_data['flow']),
                                        "observed": file_data['flow'],
                                        "precip": file_data['precip'],
                                        "is_calib": is_calib,
                                    }
                                    st.warning(f"  ⚠️ {file_name}/{model_name}: Simulation results all zeros")
                                else:
                                    file_simulation_results[model_name][file_name] = {
                                        "model_name": model_name,
                                        "params": fallback_params,
                                        "nse": calc_nse(obs_for_metric, sim_for_metric),
                                        "kge": calc_kge(obs_for_metric, sim_for_metric),
                                        "rmse": calc_rmse(obs_for_metric, sim_for_metric),
                                        "pbias": calc_pbias(obs_for_metric, sim_for_metric),
                                        "simulated": simulated,
                                        "observed": file_data['flow'],
                                        "precip": file_data['precip'],
                                        "is_calib": is_calib,
                                    }
                                    st.warning(f"  ⚠️ {file_name}/{model_name}: Using default parameters")
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
                            st.error(f"  ❌ {file_name}/{model_name}: Complete failure")
                            st.code(f"Data length: {len(file_data['precip'])}, range: [{file_data['precip'].min():.2f}, {file_data['precip'].max():.2f}]")
                            st.code(f"Flow range: [{file_data['flow'].min():.2f}, {file_data['flow'].max():.2f}]")
                            st.code(traceback.format_exc()[-800:])
            
            st.success(f"✅ Multi-file calibration complete: {len(calibration_results)} models × {len(file_data_list)} flood events")
            
            # Calculate average metrics across calibration events for each model (event-average, not concatenated NSE)
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
                    calibration_results[model_name]['nse'] = mean_valid_metric(calib_nse_list)
                    calibration_results[model_name]['kge'] = mean_valid_metric(calib_kge_list)
                    calibration_results[model_name]['rmse'] = mean_valid_metric(calib_rmse_list)
                    calibration_results[model_name]['pbias'] = mean_valid_metric(calib_pbias_list)
            
            # Initialize XGB correction data dictionary (for subsequent plotting)
            xgb_correction_data = {}
            
            # XGBoost误差校正（多文件模式）
            if enable_xgb_correction:
                st.divider()
                st.subheader("🔧 XGBoost Error Correction")
                
                try:
                    from src.app.error_correction import (
                        ErrorCorrector, select_best_model, apply_error_correction
                    )
                    from src.hydro_calc import calc_nse
                    
                    with st.spinner("Selecting best model and training error corrector..."):
                        best_name, best_result = select_best_model(calibration_results)
                    
                    if best_name and best_result is not None:
                        st.success(f"✅ Best model: {best_name} (NSE={best_result['nse']:.4f})")
                        
                        model = ModelRegistry.get_model(best_name)
                        best_params = best_result['params']
                        
                        spatial_data = {'area': catchment_area}
                        if best_name == 'XAJ Model':
                            safe_params = best_params.copy()
                            ki = min(safe_params.get('KI', 0.3), 0.45)
                            kg = min(safe_params.get('KG', 0.3), 0.45)
                            if ki + kg >= 0.9:
                                ki = 0.3
                                kg = 0.3
                            safe_params['KI'] = ki
                            safe_params['KG'] = kg
                            safe_params['Area'] = catchment_area
                        else:
                            safe_params = best_params.copy()
                        
                        st.info(f"📊 Applying XGB error correction to all events")
                        
                        train_files_data = calib_files
                        test_files_data = file_data_list  # Apply XGB correction to all events
                        
                        sim_cache = {}
                        debug_failed = []
                        
                        train_precip_events = []
                        train_flow_events = []
                        train_sim_events = []
                        
                        for f in train_files_data:
                            try:
                                sim = model.run(f['precip'], f['evap'], safe_params, spatial_data)
                                if sim is not None and len(sim) > 0:
                                    if enable_upstream_routing and f.get('upstream') is not None:
                                        sim = apply_configured_routing(sim, f['upstream'], safe_params)
                                    
                                    if warmup_steps > 0 and len(f['precip']) > warmup_steps:
                                        train_precip_events.append(f['precip'][warmup_steps:])
                                        train_flow_events.append(f['flow'][warmup_steps:])
                                        train_sim_events.append(sim[warmup_steps:])
                                    else:
                                        train_precip_events.append(f['precip'])
                                        train_flow_events.append(f['flow'])
                                        train_sim_events.append(sim)
                                else:
                                    debug_failed.append((f['file_name'], 'sim is None or empty'))
                            except Exception as e:
                                debug_failed.append((f['file_name'], str(e)))
                        
                        if debug_failed:
                            st.warning(f"⚠️ {len(debug_failed)} calibration events failed: {debug_failed[:5]}...")
                        
                        if len(train_precip_events) < 3:
                            st.warning("⚠️ Fewer than 3 calibration events, insufficient data to train XGB error correction model.")
                            xgb_correction_data = {}
                        else:
                            lower_q = (1.0 - xgb_clip_quantile) / 2.0
                            upper_q = 1.0 - lower_q
                            corrector = ErrorCorrector(
                                xgb_params=xgb_params,
                                enable_error_clipping=xgb_enable_error_clipping,
                                error_clip_quantiles=(lower_q, upper_q),
                                max_correction_ratio=xgb_max_correction_ratio,
                            )
                            corrector.train_by_events(
                                train_precip_events, train_flow_events, train_sim_events, test_event_ratio=0.0
                            )
                            st.success(f"✅ XGB model training complete: trained on {len(train_precip_events)} calibration events.")
                            
                            xgb_correction_data = {}
                            correction_results_list = []
                            
                            for idx, sf in enumerate(test_files_data):
                                try:
                                    simulated = model.run(
                                        sf['precip'],
                                        sf['evap'],
                                        safe_params,
                                        spatial_data
                                    )
                                    
                                    if simulated is None or len(simulated) == 0:
                                        continue
                                    
                                    if enable_upstream_routing and sf.get('upstream') is not None:
                                        simulated = apply_configured_routing(
                                            simulated, sf['upstream'], safe_params
                                        )
                                    
                                    if warmup_steps > 0 and len(sf['flow']) > warmup_steps:
                                        flow_plot = sf['flow'][warmup_steps:]
                                        simulated_plot = simulated[warmup_steps:]
                                        precip_plot = sf['precip'][warmup_steps:]
                                    else:
                                        flow_plot = sf['flow']
                                        simulated_plot = simulated
                                        precip_plot = sf['precip']
                                    
                                    corrected_plot = corrector.correct_with_true_error_lags(
                                        precip_plot, flow_plot, simulated_plot
                                    )
                                    
                                    nse_before = calc_nse(flow_plot, simulated_plot)
                                    nse_after = calc_nse(flow_plot, corrected_plot)
                                    kge_after = calc_kge(flow_plot, corrected_plot)
                                    rmse_after = calc_rmse(flow_plot, corrected_plot)
                                    pbias_after = calc_pbias(flow_plot, corrected_plot)
                                    
                                    xgb_correction_data[sf['file_name']] = {
                                        'corrected': corrected_plot,
                                        'flow': flow_plot,
                                        'nse': nse_after,
                                        'kge': kge_after,
                                        'rmse': rmse_after,
                                        'pbias': pbias_after,
                                        'best_model': best_name,
                                    }
                                    
                                    correction_results_list.append({
                                        'file_name': sf['file_name'],
                                        'nse_before': nse_before,
                                        'nse_after': nse_after,
                                        'nse_improvement': nse_after - nse_before,
                                    })
                                except Exception as e:
                                    st.error(f"⚠️ {sf['file_name']}: Processing failed - {str(e)}")
                    else:
                        st.warning("Unable to select the best model for error correction")
                except Exception as e:
                    st.warning(f"Error correction function temporarily unavailable: {str(e)}")
            else:
                st.info("ℹ️ XGBoost error correction disabled")
            
            # ============================================================
            # 📊 Plot: one figure per file
            # ============================================================
            summary_data = []
            all_results = {}
            
            # Ensure xgb_correction_data is always defined
            if 'xgb_correction_data' not in locals():
                xgb_correction_data = {}
            
            model_colors = {
                'XAJ Model': '#e74c3c',
                'Tank Model': '#3498db',
                'HBV Model': '#2ecc71',
            }
            
            n_files = len(file_dfs)
            for idx, (file_name, raw_df) in enumerate(file_dfs):
                panel_index = idx % EVENTS_PER_FIGURE
                if panel_index == 0:
                    batch_size = min(EVENTS_PER_FIGURE, n_files - idx)
                    fig, axes = create_event_figure(batch_size)
                ax = axes[panel_index]

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
                    df['precip'] = df['precip'].fillna(0) if 'precip' in df.columns else pd.Series(np.zeros(len(df)))
                    df['flow'] = df['flow'].fillna(0) if 'flow' in df.columns else pd.Series(np.zeros(len(df)))
                    precip_arr = np.array(df['precip'].values)
                    flow_arr = np.array(df['flow'].values)
                    is_calib = False
                
                if precip_arr is None or len(precip_arr) == 0:
                    precip_arr = np.zeros(len(flow_arr)) if flow_arr is not None and len(flow_arr) > 0 else np.array([0.0])
                if flow_arr is None or len(flow_arr) == 0:
                    flow_arr = np.zeros(len(precip_arr))
                precip_arr = np.nan_to_num(precip_arr, nan=0.0, posinf=0.0, neginf=0.0)
                flow_arr = np.nan_to_num(flow_arr, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Exclude warm-up period data
                if warmup_steps > 0 and len(flow_arr) > warmup_steps:
                    flow_arr_plot = flow_arr[warmup_steps:]
                    precip_arr_plot = precip_arr[warmup_steps:]
                else:
                    flow_arr_plot = flow_arr
                    precip_arr_plot = precip_arr
                
                event_type = "Calibration" if is_calib else "Validation"
                ax.plot(flow_arr_plot, "k-", label="Observed", linewidth=2.5)
                
                ax2 = ax.twinx()
                ax2.bar(range(len(precip_arr_plot)), precip_arr_plot, color='#87CEEB', alpha=0.5, width=1, label='Precipitation')
                ax2.set_ylabel("Precipitation (mm)", fontsize=14, color='black')
                ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
                ax2.invert_yaxis()
                
                xlabel_text = "Time (days)" if user_timestep == 'daily' else "Time (h)"
                
                simulated_list = []
                model_names = []
                nse_list = []
                for model_name in file_simulation_results:
                    if file_name in file_simulation_results.get(model_name, {}):
                        result = file_simulation_results[model_name][file_name]
                        simulated = result.get('simulated')
                        if simulated is None or len(simulated) == 0:
                            st.warning(f"⚠️ {file_name}/{model_name}: Simulation result is empty, skipping plot")
                            continue
                        simulated = np.nan_to_num(simulated, nan=0.0, posinf=0.0, neginf=0.0)
                        # Exclude warm-up period data
                        if warmup_steps > 0 and len(simulated) > warmup_steps:
                            simulated_plot = simulated[warmup_steps:]
                        else:
                            simulated_plot = simulated
                        simulated_list.append(simulated_plot)
                        model_names.append(model_name)
                        nse_list.append(result['nse'])
                        
                        color = model_colors.get(model_name, '#999999')
                        label = f"{model_name} (NSE={result['nse']:.3f})"
                        ax.plot(simulated_plot, color=color, label=label, linewidth=2, alpha=0.85)
                        
                        summary_data.append({
                            "File": file_name,
                            "Event": "Full Event",
                            "Type": event_type,
                            "Model": model_name,
                            "NSE": result['nse'],
                            "KGE": result.get('kge', result['nse']),
                            "RMSE": result['rmse'],
                            "PBIAS": result['pbias']
                        })
                            
                        if file_name not in all_results:
                            all_results[file_name] = {}
                        if "Full Event" not in all_results[file_name]:
                            all_results[file_name]["Full Event"] = []
                        all_results[file_name]["Full Event"].append(result)
                    
                if len(simulated_list) >= 2 and len(flow_arr_plot) == len(simulated_list[0]):
                    weights = calc_bma_weights(nse_list)
                    bma_result = apply_bma_ensemble(simulated_list, weights)
                    bma_nse = calc_nse(flow_arr_plot, bma_result)
                    bma_kge = calc_kge(flow_arr_plot, bma_result)
                    bma_rmse = calc_rmse(flow_arr_plot, bma_result)
                    bma_pbias = calc_pbias(flow_arr_plot, bma_result)
                    label = f'BMA Ensemble (NSE={bma_nse:.3f})'
                    ax.plot(bma_result, color='#9b59b6', linestyle='--', 
                            label=label, linewidth=2.4, alpha=0.9)
                    
                    summary_data.append({
                        "File": file_name,
                        "Event": "Full Event",
                        "Type": event_type,
                        "Model": "BMA Ensemble",
                        "NSE": bma_nse,
                        "KGE": bma_kge,
                        "RMSE": bma_rmse,
                        "PBIAS": bma_pbias
                    })
                
                # Add XGB error correction curve
                xgb_in_data = xgb_correction_data and file_name in xgb_correction_data
                if xgb_in_data:
                    xgb_data = xgb_correction_data[file_name]
                    corrected_plot = xgb_data['corrected']
                    
                    if len(corrected_plot) == len(flow_arr_plot):
                        xgb_nse = xgb_data['nse']
                        xgb_rmse = xgb_data['rmse']
                        xgb_pbias = xgb_data['pbias']
                        xgb_model = xgb_data['best_model']
                        label = f'XGB Corrected ({xgb_model}, NSE={xgb_nse:.3f})'
                        ax.plot(corrected_plot, color='#f39c12', linestyle='-.', 
                                label=label, linewidth=2.4, alpha=0.95)
                        
                        summary_data.append({
                            "File": file_name,
                            "Event": "Full Event",
                            "Type": event_type,
                            "Model": f"{xgb_model}-XGB",
                            "NSE": xgb_nse,
                            "KGE": xgb_data['kge'],
                            "RMSE": xgb_rmse,
                            "PBIAS": xgb_pbias
                        })
                
                ax.set_title(f"{file_name} [{event_type}]", fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(xlabel_text, fontsize=12)
                ax.set_ylabel(r"Flow ($m^3/s$)", fontsize=12)
                
                handles, labels = ax.get_legend_handles_labels()
                precip_handles, precip_labels = ax2.get_legend_handles_labels()
                ax.legend(
                    handles + precip_handles,
                    labels + precip_labels,
                    fontsize=8,
                    loc='upper right',
                    frameon=True,
                    framealpha=0.85
                )
            
                if panel_index + 1 == batch_size:
                    fig.tight_layout()
                    batch_number = idx // EVENTS_PER_FIGURE + 1
                    render_figure(
                        fig, label="💾 Download Image",
                        file_name=f"flow_comparison_batch_{batch_number}.png",
                        key=f"download_plots_batch_{batch_number}",
                    )

            # Summary metrics table
            if summary_data:
                st.divider()
                st.subheader("📋 Calibration & Validation Metrics Summary")
                
                # Calculate average metrics for calibration and validation separately
                calib_data = [d for d in summary_data if d['Type'] == 'Calibration']
                valid_data = [d for d in summary_data if d['Type'] == 'Validation']
                
                # Create calibration metrics summary table
                if calib_data:
                    st.subheader("📊 Calibration Event Metrics")
                    calib_df = pd.DataFrame(calib_data)
                    st.dataframe(calib_df, width="stretch", hide_index=True)
                
                # Create validation metrics summary table
                if valid_data:
                    st.subheader("📊 Validation Event Metrics")
                    valid_df = pd.DataFrame(valid_data)
                    st.dataframe(valid_df, width="stretch", hide_index=True)
                
                if calib_data and valid_data:
                    st.divider()
                    st.subheader("📊 Model Performance Comparison")
                    
                    model_names_for_comparison = ['HBV Model', 'XAJ Model', 'Tank Model']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Calibration Summary**")
                        for model_name in model_names_for_comparison:
                            model_calib_data = [d for d in calib_data if d['Model'] == model_name]
                            if model_calib_data:
                                calib_nse = mean_valid_metric([d['NSE'] for d in model_calib_data])
                                calib_kge = mean_valid_metric([d['KGE'] for d in model_calib_data])
                                calib_rmse = mean_valid_metric([d['RMSE'] for d in model_calib_data])
                                calib_pbias = mean_valid_metric([d['PBIAS'] for d in model_calib_data])
                                st.markdown(f"**{model_name}**")
                                st.metric("NSE", f"{calib_nse:.4f}")
                                st.metric("KGE", f"{calib_kge:.4f}")
                                st.metric("RMSE", f"{calib_rmse:.4f}")
                                st.metric("PBIAS", f"{calib_pbias:.2f}%")
                                st.divider()
                    
                    with col2:
                        st.markdown("**Validation Summary**")
                        for model_name in model_names_for_comparison:
                            model_valid_data = [d for d in valid_data if d['Model'] == model_name]
                            if model_valid_data:
                                valid_nse = mean_valid_metric([d['NSE'] for d in model_valid_data])
                                valid_kge = mean_valid_metric([d['KGE'] for d in model_valid_data])
                                valid_rmse = mean_valid_metric([d['RMSE'] for d in model_valid_data])
                                valid_pbias = mean_valid_metric([d['PBIAS'] for d in model_valid_data])
                                st.markdown(f"**{model_name}**")
                                st.metric("NSE", f"{valid_nse:.4f}")
                                st.metric("KGE", f"{valid_kge:.4f}")
                                st.metric("RMSE", f"{valid_rmse:.4f}")
                                st.metric("PBIAS", f"{valid_pbias:.2f}%")
                                st.divider()
                    
                    with col3:
                        xgb_data = [d for d in summary_data if '-XGB' in d['Model']]
                        if xgb_data:
                            xgb_groups = [
                                ("XGB Calibration Summary", [d for d in xgb_data if d['Type'] == 'Calibration']),
                                ("XGB Validation Summary", [d for d in xgb_data if d['Type'] == 'Validation']),
                                ("XGB Overall Summary", xgb_data),
                            ]
                            for title, group_data in xgb_groups:
                                if not group_data:
                                    continue
                                xgb_nse = mean_valid_metric([d['NSE'] for d in group_data])
                                xgb_kge = mean_valid_metric([d['KGE'] for d in group_data])
                                xgb_rmse = mean_valid_metric([d['RMSE'] for d in group_data])
                                xgb_pbias = mean_valid_metric([d['PBIAS'] for d in group_data])
                                st.markdown(f"**{title} (n={len(group_data)})**")
                                st.metric("NSE", f"{xgb_nse:.4f}")
                                st.metric("KGE", f"{xgb_kge:.4f}")
                                st.metric("RMSE", f"{xgb_rmse:.4f}")
                                st.metric("PBIAS", f"{xgb_pbias:.2f}%")
                                st.divider()
                        else:
                            st.info("No XGB data")
                
                # Unified parameter table
                st.divider()
                st.subheader("📋 Model Calibration Parameters")
                for model_name, calib_result in calibration_results.items():
                    with st.expander(f"📊 {model_name}"):
                        st.markdown(f"**Calibration NSE**: {calib_result['nse']:.4f}")
                        param_df = generate_param_table(model_name, calib_result['params'])
                        st.dataframe(param_df, width="stretch", hide_index=True)
            
            # Multi-file AI report
            st.divider()
            st.subheader("🤖 AI Agent Intelligent Analysis Report")
            
            with st.spinner("🤖 AI Agent is generating the analysis report..."):
                xgb_results = correction_results_list if 'correction_results_list' in locals() and correction_results_list else None
                multifile_report = generate_multifile_report(
                    file_data_list,
                    calibration_results,
                    file_simulation_results,
                    call_minimax,
                    warmup_hours,
                    xgb_results
                )
            render_copyable_report(multifile_report, key="multifile_ai_report_copyable")
            
            if not multifile_report.startswith("[ERROR]"):
                report_filename = f"multifile_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                st.download_button(
                    "📥 Export Report (Markdown)",
                    data=multifile_report,
                    file_name=report_filename,
                    mime="text/markdown"
                )
            
            st.stop()  # Multi-file mode complete, stop subsequent code
        
        elif upload_mode == "Single File (Continuous Series)":
            # Continuous series mode: use representative flood events for calibration
            file_calibration_results = {}
            
            # Calculate warm-up period steps
            if warmup_hours > 0 and user_timestep == 'hourly':
                warmup_steps = warmup_hours
            elif warmup_hours > 0:
                warmup_steps = warmup_hours // 24
            else:
                warmup_steps = 0
            
            if preanalysis_result.selected_events:
                # Build calibration event list
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
                st.info(f"🤖 Using {len(calib_events)} representative flood events for calibration, {sum(len(e['flow']) for e in calib_events)} total time steps" +
                       (f", warm-up {warmup_steps} steps" if warmup_steps > 0 else ""))
            else:
                _all_upstream_arr = all_upstream_arr if 'all_upstream_arr' in globals() else None
                calib_events = [{
                    'precip': all_precip_arr,
                    'evap': all_evap_arr,
                    'flow': all_flow_arr,
                    'upstream': _all_upstream_arr if enable_upstream_routing and _all_upstream_arr is not None else None
                }]
                st.info(f"📊 Full flood event data, {len(calib_events[0]['flow'])} total time steps" +
                       (f", warm-up {warmup_steps} steps" if warmup_steps > 0 else ""))
            
            progress_bar = st.progress(0)
            with st.spinner("🤖 AI Agent is calibrating models..."):
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        continue
                    try:
                        spatial_data = {'area': catchment_area}
                        
                        total_models = len(RECOMMENDED_MODELS)
                        model_base = model_idx / total_models
                        
                        manual_routing = manual_routing_params if enable_upstream_routing else None
                        
                        result = run_configured_model(
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
                            progress_callback=lambda p: progress_bar.progress(model_base + p / total_models),
                            manual_routing_params=manual_routing
                        )
                    except Exception as e:
                        st.error(f"  ⚠️ {model_name} calibration error: {type(e).__name__}: {str(e)}")
                        result = None
                    progress_bar.progress((model_idx + 1) / len(RECOMMENDED_MODELS))
                    if result:
                        params, nse, simulated = result
                        try:
                            full_simulated = ModelRegistry.get_model(model_name).run(
                                all_precip_arr, all_evap_arr,
                                {key: value for key, value in params.items() if key not in ('k_routing', 'x_routing')},
                                {'area': catchment_area, 'timestep': user_timestep},
                                warmup_steps=warmup_steps,
                            )
                            if enable_upstream_routing:
                                full_simulated = apply_configured_routing(full_simulated, all_upstream_arr, params)
                        except Exception as exc:
                            st.error(f"{model_name}: {exc}")
                            continue
                        calibration_results[model_name] = {
                            "model_name": model_name,
                            "params": params,
                            "nse": nse,
                            "rmse": calc_rmse(calib_events[0]['flow'], simulated),
                            "mae": calc_mae(calib_events[0]['flow'], simulated),
                            "pbias": calc_pbias(calib_events[0]['flow'], simulated),
                            "simulated": simulated,
                            "simulated_full": full_simulated,
                            "calib_data": (calib_events[0]['precip'], calib_events[0]['evap'], calib_events[0]['flow']),
                        }
                        st.write(f"  ✅ {model_name}: Avg NSE = {nse:.4f}")
        else:
            # Single file, one flood event mode
            file_calibration_results = {}
            
            # Calculate warm-up period steps
            if warmup_hours > 0 and user_timestep == 'hourly':
                warmup_steps = warmup_hours
            elif warmup_hours > 0:
                warmup_steps = warmup_hours // 24
            else:
                warmup_steps = 0
            
            # Build single event list
            _all_upstream = all_upstream_arr if 'all_upstream_arr' in globals() and all_upstream_arr is not None else None
            calib_events = [{
                'precip': all_precip_arr,
                'evap': all_evap_arr,
                'flow': all_flow_arr,
                'upstream': _all_upstream
            }]
            
            st.info(f"📊 Full flood event data, {len(calib_events[0]['flow'])} total time steps" +
                   (f", warm-up {warmup_steps} steps" if warmup_steps > 0 else ""))
            
            progress_bar = st.progress(0)
            with st.spinner("🤖 AI Agent is calibrating models..."):
                for model_idx, model_name in enumerate(RECOMMENDED_MODELS):
                    if model_name in SKIP_MODELS:
                        continue
                    try:
                        spatial_data = {'area': catchment_area}
                        
                        total_models = len(RECOMMENDED_MODELS)
                        model_base = model_idx / total_models
                        
                        manual_routing = manual_routing_params if enable_upstream_routing else None
                        
                        result = run_configured_model(
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
                            progress_callback=lambda p: progress_bar.progress(model_base + p / total_models),
                            manual_routing_params=manual_routing
                        )
                    except Exception as e:
                        st.error(f"  ⚠️ {model_name} calibration error: {type(e).__name__}: {str(e)}")
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
        # 📊 Results display
        # ============================================================
        st.divider()
        st.subheader("📈 Model Calibration & Validation Results")
        
        summary_data = []
        all_results = {}
        
        model_colors = {
            'XAJ Model': '#e74c3c',
            'Tank Model': '#3498db',
            'HBV Model': '#2ecc71',
        }
        
        if upload_mode == "Single File (Continuous Series)":
            # Continuous series mode
            # Exclude warm-up period data
            if warmup_steps > 0 and len(all_flow_arr) > warmup_steps:
                all_flow_arr_plot = all_flow_arr[warmup_steps:]
                all_precip_arr_plot = all_precip_arr[warmup_steps:]
            else:
                all_flow_arr_plot = all_flow_arr
                all_precip_arr_plot = all_precip_arr
            
            fig, ax = plt.subplots(figsize=(16, 5.8), dpi=PLOT_DPI)
            
            ax.plot(all_flow_arr_plot, "k-", label="Observed", linewidth=2.5)
            
            ax2 = ax.twinx()
            ax2.bar(range(len(all_precip_arr_plot)), all_precip_arr_plot, color='#87CEEB', alpha=0.5, width=1, label='Precipitation')
            ax2.set_ylabel("Precipitation (mm)", fontsize=14, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
            ax2.invert_yaxis()
            
            xlabel_text = "Time (days)" if user_timestep == 'daily' else "Time (h)"
            
            simulated_list = []
            model_names = []
            nse_list = []
            for model_name, result in calibration_results.items():
                simulated = result.get('simulated_full', result['simulated'])
                # Exclude warm-up period data
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
                    "File": "Continuous Series",
                    "Event": "Full Event",
                    "Type": "Calibration",
                    "Model": model_name,
                    "NSE": result['nse'],
                    "KGE": result.get('kge', result['nse']),
                    "RMSE": result['rmse'],
                    "PBIAS": result['pbias']
                })
            
            if len(simulated_list) >= 2 and len(all_flow_arr_plot) == len(simulated_list[0]):
                weights = calc_bma_weights(nse_list)
                bma_result = apply_bma_ensemble(simulated_list, weights)
                bma_nse = calc_nse(all_flow_arr_plot, bma_result)
                bma_kge = calc_kge(all_flow_arr_plot, bma_result)
                bma_rmse = calc_rmse(all_flow_arr_plot, bma_result)
                bma_pbias = calc_pbias(all_flow_arr_plot, bma_result)
                weights_str = format_weights_string(model_names, weights)
                label = f'BMA Ensemble (NSE={bma_nse:.3f}) [{weights_str}]'
                ax.plot(bma_result, color='#9b59b6', linestyle='--', 
                        label=label, linewidth=2.5, alpha=0.9)
                
                summary_data.append({
                    "File": "Continuous Series",
                    "Event": "Full Event",
                    "Type": "Calibration",
                    "Model": "BMA Ensemble",
                    "NSE": bma_nse,
                    "KGE": bma_kge,
                    "RMSE": bma_rmse,
                    "PBIAS": bma_pbias
                })
            
            ax.set_title("Continuous Series Flood Simulation Results", fontsize=14)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(xlabel_text, fontsize=12)
            ax.set_ylabel(r"Flow ($m^3/s$)", fontsize=12)
            
            plt.tight_layout()
            render_figure(fig)
            
            # Mark identified flood events
            if preanalysis_result and hasattr(preanalysis_result, 'selected_events') and preanalysis_result.selected_events:
                st.write("**🏊 Identified Representative Flood Events:**")
                for evt in preanalysis_result.selected_events:
                    st.write(f"  - {evt.event_id}: Peak {evt.peak_flow:.1f} m³/s ({evt.start_date.strftime('%Y-%m-%d') if hasattr(evt.start_date, 'strftime') else evt.start_date})")
        
        else:
            # Single file, one flood event mode
            # Exclude warm-up period data
            if warmup_steps > 0 and len(all_flow_arr) > warmup_steps:
                all_flow_arr_plot = all_flow_arr[warmup_steps:]
                all_precip_arr_plot = all_precip_arr[warmup_steps:]
            else:
                all_flow_arr_plot = all_flow_arr
                all_precip_arr_plot = all_precip_arr
            
            fig, ax = plt.subplots(figsize=(16, 5.8), dpi=PLOT_DPI)
            
            ax.plot(all_flow_arr_plot, "k-", label="Observed", linewidth=2.5)
            
            ax2 = ax.twinx()
            ax2.bar(range(len(all_precip_arr_plot)), all_precip_arr_plot, color='#87CEEB', alpha=0.5, width=1, label='Precipitation')
            ax2.set_ylabel("Precipitation (mm)", fontsize=14, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
            ax2.invert_yaxis()
            
            xlabel_text = "Time (days)" if user_timestep == 'daily' else "Time (h)"
            
            simulated_list = []
            model_names = []
            nse_list = []
            for model_name, result in calibration_results.items():
                simulated = result['simulated']
                # Exclude warm-up period data
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
                    "File": "Single Event",
                    "Event": "Full Event",
                    "Type": "Calibration",
                    "Model": model_name,
                    "NSE": result['nse'],
                    "KGE": result.get('kge', result['nse']),
                    "RMSE": result['rmse'],
                    "PBIAS": result['pbias']
                })
            
            if len(simulated_list) >= 2 and len(all_flow_arr_plot) == len(simulated_list[0]):
                weights = calc_bma_weights(nse_list)
                bma_result = apply_bma_ensemble(simulated_list, weights)
                bma_nse = calc_nse(all_flow_arr_plot, bma_result)
                bma_kge = calc_kge(all_flow_arr_plot, bma_result)
                bma_rmse = calc_rmse(all_flow_arr_plot, bma_result)
                bma_pbias = calc_pbias(all_flow_arr_plot, bma_result)
                weights_str = format_weights_string(model_names, weights)
                label = f'BMA Ensemble (NSE={bma_nse:.3f}) [{weights_str}]'
                ax.plot(bma_result, color='#9b59b6', linestyle='--', 
                        label=label, linewidth=2.5, alpha=0.9)
                
                summary_data.append({
                    "File": "Single Event",
                    "Event": "Full Event",
                    "Type": "Calibration",
                    "Model": "BMA Ensemble",
                    "NSE": bma_nse,
                    "KGE": bma_kge,
                    "RMSE": bma_rmse,
                    "PBIAS": bma_pbias
                })
            
            ax.set_title("Single Flood Event Simulation Results", fontsize=14)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(xlabel_text, fontsize=12)
            ax.set_ylabel(r"Flow ($m^3/s$)", fontsize=12)
            
            plt.tight_layout()
            render_figure(
                fig, label="💾 Download Image",
                file_name="flow_comparison_single.png",
                key="download_single_plot_btn",
            )

            # Summary metrics table
            if summary_data:
                st.divider()
                st.subheader("📋 Calibration Metrics Summary")
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width="stretch", hide_index=True)
                
                # Unified parameter table
                st.divider()
                st.subheader("📋 Model Calibration Parameters")
                
                for model_name, calib_result in calibration_results.items():
                    with st.expander(f"📊 {model_name}"):
                        st.markdown(f"**Calibration NSE**: {calib_result['nse']:.4f}")
                        param_df = generate_param_table(model_name, calib_result['params'])
                        st.dataframe(param_df, width="stretch", hide_index=True)
            
            # XGBoost error correction
            st.divider()
            st.subheader("🔧 XGBoost Error Correction")
            
            try:
                from src.app.error_correction import (
                    ErrorCorrector, select_best_model, apply_error_correction
                )
                
                with st.spinner("Selecting best model and training error corrector..."):
                    best_name, best_result = select_best_model(calibration_results)
                    
                    if best_name and best_result is not None:
                        st.success(f"✅ Best model: {best_name} (NSE={best_result['nse']:.4f})")
                        
                        correction_result = apply_error_correction(
                            best_name,
                            best_result,
                            all_precip_arr,
                            all_flow_arr,
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("NSE Before Correction", f"{correction_result['nse_before']:.4f}")
                        with col2:
                            st.metric("NSE After Correction", f"{correction_result['nse_after']:.4f}")
                        with col3:
                            st.metric("NSE Improvement", f"{correction_result['nse_improvement']:+.4f}", 
                                      delta=correction_result['nse_improvement'])
                        
                        x = np.arange(len(all_flow_arr))
                        xlabel_text = "Time (days)" if user_timestep == 'daily' else "Time (h)"
                        
                        fig_ec, ax_ec = plt.subplots(figsize=(16, 5.8), dpi=PLOT_DPI)
                        ax_ec.plot(x, all_flow_arr, 'k-', linewidth=2, label='Observed Flow', alpha=0.8)
                        ax_ec.plot(x, correction_result['simulated'], 'b--', linewidth=1.5,
                                 label=f"Original ({best_name})", alpha=0.6)
                        ax_ec.plot(x, correction_result['corrected'], 'r-', linewidth=1.5,
                                 label="XGB Corrected", alpha=0.8)
                        ax_ec.set_xlabel(xlabel_text, fontsize=12)
                        ax_ec.set_ylabel(r'Flow ($m^3/s$)', fontsize=12)
                        ax_ec.set_title(f"{best_name} + XGB Error Correction - Flow Comparison", fontsize=14)
                        ax_ec.legend(fontsize=10, loc='upper right')
                        ax_ec.grid(True, alpha=0.3)
                        render_figure(fig_ec)
                        
                        corrected_df = pd.DataFrame({
                            'Time Step': x,
                            'Observed Flow': all_flow_arr,
                            'Original Simulation': correction_result['simulated'],
                            'Corrected Flow': correction_result['corrected']
                        })
                        st.download_button(
                            "📥 Export Correction Results (CSV)",
                            data=corrected_df.to_csv(index=False),
                            file_name=f"corrected_flow_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Unable to select best model for error correction")
            except Exception as e:
                st.warning(f"Error correction temporarily unavailable: {str(e)}")
            
            # AI report (multi-file mode only)
            if upload_mode == "Multi-File (One Event Per File)":
                try:
                    st.divider()
                    st.subheader("🤖 AI Agent Intelligent Analysis Report")
                    
                    with st.spinner("🤖 AI Agent is generating the analysis report..."):
                        multifile_report = generate_multifile_report(
                            file_data_list,
                            calibration_results,
                            file_simulation_results,
                            call_minimax,
                            warmup_hours
                        )
                    render_copyable_report(multifile_report, key="single_mode_multifile_ai_report_copyable")
                    
                    if not multifile_report.startswith("[ERROR]"):
                        report_filename = f"multifile_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                        st.download_button(
                            "📥 Export Report (Markdown)",
                            data=multifile_report,
                            file_name=report_filename,
                            mime="text/markdown"
                        )
                except NameError:
                    pass
                render_copyable_report(multifile_report, key="single_mode_multifile_ai_report_copyable_fallback")
                
                if not multifile_report.startswith("[ERROR]"):
                    report_filename = f"multifile_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                    st.download_button(
                        "📥 Export Report (Markdown)",
                        data=multifile_report,
                        file_name=report_filename,
                        mime="text/markdown"
                    )

else:
# Welcome page
    st.info("👈 Please upload hydrological data files from the left sidebar to start analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Supported Models", f"{len(RECOMMENDED_MODELS)}")
    with col2:
        st.metric("Supported Formats", "CSV / Excel")
    with col3:
        st.metric("Calibration Algorithm", "Two-Stage (Parallel)")
    
    st.divider()
    
    st.subheader("📖 Usage Instructions")
    st.markdown("""
    1. **Upload Data**: Upload one or more hydrological data files
    2. **Flood Events**: System automatically identifies each flood event
    3. **Auto Calibration**: Multi-model calibration for each flood event
    4. **Result Comparison**: View performance metrics, parameter tables, flow hydrographs
    5. **Intelligent Report**: Auto-generated analysis report with export support
    """)
    
    st.subheader("📊 Data Format Requirements")
    st.markdown("""
    Data should include the following columns (column names support both Chinese and English):
    - **Date Column**: date / time / 日期 / 时间
    - **Precipitation Column**: precip / rainfall / p / 降水 / 降雨
    - **Evaporation Column**: evap / et / 蒸发 (optional)
    - **Flow Column**: flow / discharge / q / 流量 / 径流
    """)

# Welcome page (shown when no files uploaded)
if not uploaded_files:
    st.info("👈 Please upload hydrological data files from the left sidebar to start analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Supported Models", f"{len(RECOMMENDED_MODELS)}")
    with col2:
        st.metric("Supported Formats", "CSV / Excel")
    with col3:
        st.metric("Calibration Algorithms", "5 Options")
    with col4:
        st.metric("Flow Routing", "Muskingum")
    
    st.divider()
    
    st.subheader("📖 Usage Instructions")
    st.markdown("""
    1. **Upload Data**: Upload one or more hydrological data files
    2. **Flood Events**: System automatically identifies each flood event
    3. **Auto Calibration**: Multi-model calibration for each flood event
    4. **Result Comparison**: View performance metrics (NSE/KGE/RMSE/PBIAS), parameter tables, flow hydrographs
    5. **Intelligent Report**: Auto-generated analysis report with export support
    """)
    
    st.subheader("🔧 Calibration Algorithm Options")
    st.markdown("""
    - **Two-Stage (Recommended)**: dual_annealing + L-BFGS-B, fast global search + local fine optimization
    - **PSO**: Particle Swarm Optimization, suitable for large-scale problems
    - **SCE-UA**: Shuffled Complex Evolution, strong global optimization
    - **DE**: Differential Evolution, simple and efficient, suitable for continuous parameter optimization
    - **GA**: Genetic Algorithm, maintains diversity during evolution
    """)
    
    st.subheader("🌊 Upstream Outflow Routing")
    st.markdown("""
    Supports upstream outflow Muskingum channel routing:
    - Enable "Upstream Outflow Routing" in the sidebar
    - Specify upstream outflow column name (data must be in the same file)
    - Muskingum parameters (k, x) can be calibrated jointly or set to fixed values
    """)
    
    st.subheader("📊 Data Format Requirements")
    st.markdown("""
    Data should include the following columns (column names support both Chinese and English):
    - **Date Column**: date / time / 日期 / 时间
    - **Precipitation Column**: precip / rainfall / p / 降水 / 降雨
    - **Evaporation Column**: evap / et / 蒸发 (optional)
    - **Flow Column**: flow / discharge / q / 流量 / 径流
    - **Upstream Outflow Column**: upstream (optional, required when routing is enabled)
    """)

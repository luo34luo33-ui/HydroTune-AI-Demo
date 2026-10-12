"""Shared routing settings and validation for the Chinese and English apps."""
import numpy as np
import pandas as pd
import streamlit as st


def _remember_manual_values():
    st.session_state['_routing_manual_values'] = {
        'k_routing': st.session_state['routing_k_input'],
        'x_routing': st.session_state['routing_x_input'],
    }


def render_routing_settings(import_mode=False, english=False):
    """Return None for joint calibration or a fixed k/x pair for simulation."""
    if not import_mode:
        choice = st.radio(
            'Include Muskingum parameters in calibration?' if english else '马斯京根参数是否纳入率定',
            ['Yes', 'No'] if english else ['是', '否'],
            index=0, horizontal=True, key='routing_calibration_choice',
        )
        if choice in ('是', 'Yes'):
            st.caption('Calibration bounds: k = 0.5–5.0 time steps, x = 0–0.5.' if english
                       else '率定范围：k = 0.5–5.0 个计算时间步，x = 0–0.5。')
            return None

    values = st.session_state.get('_routing_manual_values', {'k_routing': 2.5, 'x_routing': 0.25}).copy()
    imported = (st.session_state.get('imported_k_routing'), st.session_state.get('imported_x_routing'))
    if import_mode and None not in imported and imported != st.session_state.get('_routing_imported_pair'):
        values = dict(zip(('k_routing', 'x_routing'), imported))
        if not (np.isfinite(imported).all() and 0.5 <= imported[0] <= 10 and 0 <= imported[1] <= 0.5):
            st.error('Imported Muskingum parameters are outside the allowed range.' if english
                     else '导入的马斯京根参数超出允许范围，请修改参数文件。')
            st.stop()
        st.session_state['_routing_imported_pair'] = imported
        st.session_state['routing_k_input'] = values['k_routing']
        st.session_state['routing_x_input'] = values['x_routing']

    st.session_state['_routing_manual_values'] = values
    # Separate saved values survive Streamlit deleting hidden widget state.
    for key, name in (('routing_k_input', 'k_routing'), ('routing_x_input', 'x_routing')):
        if key not in st.session_state:
            st.session_state[key] = values[name]
    st.markdown('**Muskingum Parameters (Fixed):**' if english else '**马斯京根参数（固定值）：**')
    col_k, col_x = st.columns(2)
    with col_k:
        k = st.number_input(
            'k (travel time in time steps)' if english else 'k（河道传播时间，计算时间步数）',
            min_value=0.5, max_value=10.0, step=0.1,
            key='routing_k_input', on_change=_remember_manual_values,
            help='Number of computational time steps; one step is one hour or one day.' if english
                 else '以计算时间步为单位；小时尺度每步为 1 小时，日尺度每步为 1 天。',
        )
    with col_x:
        x = st.number_input(
            'x (weighting factor)' if english else 'x（洪水坦化系数）',
            min_value=0.0, max_value=0.5, step=0.01,
            key='routing_x_input', on_change=_remember_manual_values,
        )
    values = {'k_routing': k, 'x_routing': x}
    st.session_state['_routing_manual_values'] = values
    return values


def validate_upstream_columns(file_dfs, column, english=False):
    """Reject invalid upstream mappings before any analysis or API calls."""
    if not column:
        raise ValueError('Enter the upstream discharge column name.' if english else '请填写上游出库列名。')
    for name, df in file_dfs:
        if column not in df.columns:
            raise ValueError(f'{name}: upstream column "{column}" not found.' if english
                             else f'{name}：找不到上游出库列“{column}”。')
        values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f'{name}: upstream discharge must contain finite numeric values.' if english
                             else f'{name}：上游出库列含有空值、非数字或无穷值，请修正数据。')


def simulate_imported_model(model_name, precip, evap, observed_flow, *, params,
                            spatial_data=None, temperature=None, timestep='daily',
                            upstream_flow=None, enable_routing=False, calib_events=None,
                            warmup_steps=0, manual_routing_params=None, **_optimizer_options):
    """Run fixed imported model/routing parameters without invoking an optimizer."""
    from src.hydro_calc import apply_configured_routing, calc_nse
    from src.models.registry import ModelRegistry

    model = ModelRegistry.get_model(model_name)
    model_params = {k: v for k, v in params.items() if k not in ('k_routing', 'x_routing')}
    result_params = model_params.copy()
    if enable_routing:
        if manual_routing_params is None:
            raise ValueError('Imported model simulation requires fixed Muskingum parameters')
        result_params.update(manual_routing_params)
    spatial = dict(spatial_data or {})
    spatial['timestep'] = timestep
    events = calib_events if calib_events else [
        dict(precip=precip, evap=evap, flow=observed_flow, upstream=upstream_flow)
    ]
    first_simulated = None
    scores = []
    for event in events:
        simulated = model.run(event['precip'], event['evap'], model_params, spatial,
                              temperature, warmup_steps)
        if enable_routing:
            simulated = apply_configured_routing(simulated, event.get('upstream'), result_params)
        if first_simulated is None:
            first_simulated = simulated
        observed = event['flow']
        obs = observed[warmup_steps:] if 0 < warmup_steps < len(observed) else observed
        sim = simulated[warmup_steps:] if 0 < warmup_steps < len(simulated) else simulated
        score = calc_nse(obs, sim)
        if np.isfinite(score):
            scores.append(score)
    return result_params, float(np.mean(scores)) if scores else -1e10, first_simulated

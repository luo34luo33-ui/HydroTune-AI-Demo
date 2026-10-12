"""Collapsed, session-local API settings shared by both deployment entrypoints."""
import streamlit as st
from src.llm_api import call_minimax, get_llm_config, DEFAULT_MINIMAX_MODEL


def render_llm_settings(language="zh"):
    en = language == "en"
    def tr(zh, eng):
        return eng if en else zh

    config = get_llm_config()
    with st.expander(tr("🔑 MiniMax API 设置", "🔑 MiniMax API Settings"), expanded=False):
        st.caption(tr(
            "密钥仅保存在当前会话，不写入文件。留空则使用服务器 Secrets。平台须与申请密钥的平台一致。",
            "Keys stay in this session only. Leave blank to use server Secrets. Select the platform that issued your key.",
        ))
        st.caption(tr("检测到 MiniMax 密钥：", "MiniMax key detected: ") +
                   (tr("是", "Yes") if config["api_key"] else tr("否", "No")))
        with st.form("minimax_settings_form"):
            enabled = st.checkbox(tr("优先使用 MiniMax", "Prefer MiniMax"),
                                  value=config["provider"] == "minimax", key="minimax_enabled")
            key = st.text_input("MiniMax API Key", type="password", key="minimax_key_input")
            regions = ["cn", "intl"]
            region = st.selectbox(tr("密钥所属平台", "Key platform"), regions,
                                  index=regions.index(config["region"]) if config["region"] in regions else 0,
                                  format_func=lambda x: tr("中国平台", "China") if x == "cn" else tr("国际平台", "International"),
                                  key="minimax_region_input")
            model = st.text_input(tr("模型名称", "Model name"),
                                  value=config["model"] or DEFAULT_MINIMAX_MODEL, key="minimax_model_input")
            apply = st.form_submit_button(tr("应用配置", "Apply settings"))
            test = st.form_submit_button(tr("应用并测试连接", "Apply and test connection"))
        if apply or test:
            st.session_state["llm_settings"] = {
                "provider": "minimax" if enabled or key.strip() else "auto",
                "api_key": key.strip(), "region": region,
                "model": model.strip() or DEFAULT_MINIMAX_MODEL,
            }
            st.session_state.pop("llm_last_error", None)
            st.session_state.pop("llm_test_result", None)
            if test:
                if st.session_state["llm_settings"]["provider"] != "minimax":
                    st.warning(tr("测试 MiniMax 请勾选“优先使用 MiniMax”或填入密钥。",
                                  "Select Prefer MiniMax or enter a key to test MiniMax."))
                else:
                    with st.spinner(tr("正在测试 MiniMax…", "Testing MiniMax…")):
                        result = call_minimax("Reply with OK only.")
                    st.session_state["llm_test_result"] = not result.startswith("[ERROR]")
            else:
                st.success(tr("配置已应用。", "Settings applied."))
        active = get_llm_config()
        st.caption(tr("当前服务选择：", "Current selection: ") + active["provider"])
        if active["provider"] == "auto":
            st.caption(tr("自动模式：有 DeepSeek 密钥时优先使用 DeepSeek，否则使用 MiniMax。",
                          "Auto: uses DeepSeek when its key exists, otherwise MiniMax."))
        if st.session_state.get("llm_test_result") is True:
            st.success(tr("MiniMax 连接成功。", "MiniMax connection succeeded."))
        if st.session_state.get("llm_last_error"):
            st.error(st.session_state["llm_last_error"])
        if st.button(tr("清除会话配置", "Clear session settings"), key="clear_llm_settings"):
            for name in ("llm_settings", "llm_test_result", "llm_last_error", "minimax_enabled",
                         "minimax_key_input", "minimax_region_input", "minimax_model_input"):
                st.session_state.pop(name, None)
            st.rerun()

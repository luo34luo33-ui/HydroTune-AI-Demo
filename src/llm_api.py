"""LLM clients with server secrets and session-local MiniMax overrides."""
import os
import re
import requests
from contextlib import contextmanager
from contextvars import ContextVar

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value:
        return value.strip()
    try:
        import streamlit as st
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return default


def _session_state():
    # Never store a visitor's key in process-wide environment variables.
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is None:
        return None
    import streamlit as st
    return st.session_state


_LANGUAGE = 'zh'
_explicit_settings = ContextVar('hydrotune_llm_settings', default=None)


@contextmanager
def llm_settings(settings):
    """Use visitor-local LLM settings outside a Streamlit script context."""
    token = _explicit_settings.set(settings)
    try:
        yield
    finally:
        _explicit_settings.reset(token)


def get_language() -> str:
    return _LANGUAGE


def set_language(lang: str):
    global _LANGUAGE
    _LANGUAGE = lang


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MINIMAX_ENDPOINTS = {
    "cn": "https://api.minimaxi.com/v1/chat/completions",
    "intl": "https://api.minimax.io/v1/chat/completions",
}
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"


def get_llm_config():
    state = _session_state()
    settings = _explicit_settings.get()
    if settings is None:
        settings = state.get("llm_settings", {}) if state is not None else {}
    return {
        "provider": settings.get("provider", _get_secret("LLM_PROVIDER", "auto")).lower(),
        "api_key": settings.get("api_key") or _get_secret("MINIMAX_API_KEY"),
        "region": settings.get("region") or _get_secret("MINIMAX_REGION", "cn"),
        "model": settings.get("model") or _get_secret("MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL),
    }


def _error(provider, message):
    """Only safe diagnostics reach reports/session state."""
    result = f"[ERROR] {provider}: {message}"
    state = _session_state()
    if state is not None:
        state["llm_last_error"] = result
    return result


def _request_chat(provider, url, key, payload):
    en = get_language() == 'en'
    try:
        response = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        }, timeout=(10, 120))
        # Never echo response bodies/exceptions: they can contain credentials.
        status = response.status_code
        if status >= 400:
            hints = {
                401: ("Invalid key or platform mismatch", "密钥无效或与所选平台不匹配"),
                403: ("Access denied; check model permissions", "访问被拒绝，请检查模型权限"),
                402: ("Check account balance", "请检查账户余额"),
                429: ("Rate/quota limit; check balance and retry later", "请求限流或配额不足，请检查余额并稍后重试"),
                400: ("Check model name and request settings", "请检查模型名称和请求配置"),
            }
            hint = hints.get(status, ("Service request failed", "服务请求失败"))[0 if en else 1]
            return _error(provider, f"HTTP {status} — {hint}")
        data = response.json()
        base = data.get("base_resp") or {}
        code = base.get("status_code", 0)
        if code:
            safe_code = str(code) if str(code).isdigit() else "unknown"
            return _error(provider, f"API code {safe_code}; " + (
                "check key, platform, balance and model permissions" if en else
                "请检查密钥、平台、余额和模型权限"))
        if data.get("error"):
            return _error(provider, "API rejected the request" if en else "API 拒绝了请求，请检查密钥、余额和模型权限")
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if isinstance(content, str):
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content or not isinstance(content, str) or content.startswith("<think>"):
            length = bool(choices and choices[0].get("finish_reason") == "length")
            return _error(provider, (
                "Token limit reached before answer" if en else "生成达到长度上限，尚未输出答案"
            ) if length else ("Empty response" if en else "服务返回空内容"))
        content = content.replace(key, "[REDACTED]") if key else content
        state = _session_state()
        if state is not None:
            state.pop("llm_last_error", None)
        return content
    except requests.exceptions.Timeout:
        return _error(provider, "Connection/read timeout; retry later" if en else "连接或响应超时，请稍后重试")
    except requests.exceptions.RequestException:
        return _error(provider, "Network connection failed" if en else "网络连接失败，请检查平台选择与网络")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError):
        return _error(provider, "Invalid API response" if en else "API 返回格式异常")


def _call_deepseek(prompt: str, system_prompt: str, api_key: str) -> str:
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.append({"role": "user", "content": prompt})
    return _request_chat("DeepSeek", DEEPSEEK_API_URL, api_key, {
        "model": "deepseek-chat", "messages": messages,
        "temperature": 0.7, "max_tokens": 2048, "stream": False,
    })


def call_minimax(prompt: str, system_prompt: str = "") -> str:
    """Shared dispatcher; explicit MiniMax selection overrides automatic DeepSeek."""
    config = get_llm_config()
    if config["provider"] not in ("auto", "minimax", "deepseek"):
        return _error("LLM", "LLM_PROVIDER 必须是 auto、minimax 或 deepseek / invalid provider")
    explicit = _explicit_settings.get() or {}
    ds_key = (explicit.get("api_key") if config["provider"] == "deepseek" else None) or _get_secret("DEEPSEEK_API_KEY")
    if config["provider"] == "deepseek" or (config["provider"] == "auto" and ds_key):
        if not ds_key:
            return _error("DeepSeek", "DEEPSEEK_API_KEY 未配置 / not configured")
        return _call_deepseek(prompt, system_prompt, ds_key)
    if not config["api_key"]:
        return _error("MiniMax", "MINIMAX_API_KEY 未配置，请展开侧栏设置 / configure in sidebar")
    if config["region"] not in MINIMAX_ENDPOINTS:
        return _error("MiniMax", "MINIMAX_REGION 必须是 cn 或 intl / must be cn or intl")
    messages = [{"role": "system", "content": system_prompt or (
        "You are HydroTune-AI, a professional hydrological analysis assistant."
        if get_language() == 'en' else "你是 HydroTune-AI，专业的水文模型分析助手。"
    )}, {"role": "user", "content": prompt}]
    return _request_chat("MiniMax", MINIMAX_ENDPOINTS[config["region"]], config["api_key"], {
        "model": config["model"], "messages": messages,
        "max_completion_tokens": 8192, "reasoning_split": True,
        "temperature": 0.7, "stream": False,
    })

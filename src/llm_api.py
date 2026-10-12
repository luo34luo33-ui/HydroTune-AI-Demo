"""
Minimax API 调用封装

密钥读取同时支持两种来源：环境变量（本地 .env）与 Streamlit Cloud Secrets（云端部署）。
本地开发由 .env 提供，云端由平台 Secrets 提供，共用同一套读取逻辑。
"""
import os
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 缺失时静默降级，云端无 .env 亦可运行
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """读取配置项：环境变量优先，Streamlit Cloud Secrets 回退。

    本地开发由 .env 经 load_dotenv() 注入环境变量；
    云端部署由平台 Secrets 提供，需通过 st.secrets 读取。
    非 Streamlit 进程访问 st.secrets 会抛异常，因此必须静默降级。
    """
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            try:
                candidate = secrets.get(key) if hasattr(secrets, "get") else secrets[key]
            except Exception:
                candidate = None
            if candidate:
                return str(candidate)
    except Exception:
        pass

    return default

# ============================================================
# 全局语言配置（zh / en）
# ============================================================
_LANGUAGE = 'zh'


def get_language() -> str:
    """获取当前语言"""
    return _LANGUAGE


def set_language(lang: str):
    """设置语言 (zh / en)"""
    global _LANGUAGE
    _LANGUAGE = lang


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def _call_deepseek(prompt: str, system_prompt: str, api_key: str) -> str:
    """调用 DeepSeek（OpenAI 兼容接口）"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": False,
    }

    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "") or ""
    if get_language() == 'en':
        return "[ERROR] DeepSeek returned empty content"
    return "[ERROR] DeepSeek返回空内容"


def call_minimax(prompt: str, system_prompt: str = "") -> str:
    """
    调用大模型 API，返回生成的文本

    配置了 DEEPSEEK_API_KEY 时优先使用 DeepSeek，否则回退 Minimax。

    Args:
        prompt: 用户输入
        system_prompt: 系统提示词

    Returns:
        LLM 生成的响应文本
    """
    # 优先：DeepSeek（境外云端可达性更好）
    ds_key = _get_secret("DEEPSEEK_API_KEY")
    if ds_key:
        try:
            return _call_deepseek(prompt, system_prompt, ds_key)
        except requests.exceptions.Timeout:
            if get_language() == 'en':
                return "[ERROR] DeepSeek API call timed out"
            return "[ERROR] DeepSeek API 调用超时"
        except requests.exceptions.RequestException as e:
            if get_language() == 'en':
                return f"[ERROR] DeepSeek API call failed: {e}"
            return f"[ERROR] DeepSeek API 调用失败: {e}"
        except Exception as e:
            if get_language() == 'en':
                return f"[ERROR] Error processing DeepSeek response: {e}"
            return f"[ERROR] 处理DeepSeek响应时出错: {e}"

    # 回退：Minimax
    API_KEY = _get_secret("MINIMAX_API_KEY")
    GROUP_ID = _get_secret("MINIMAX_GROUP_ID", "")

    if not API_KEY:
        lang = get_language()
        if lang == 'en':
            return "[ERROR] MINIMAX_API_KEY not found, please configure it in .env (local) or Streamlit Secrets (cloud)"
        return "[ERROR] 未找到MINIMAX_API_KEY，请在.env文件（本地）或Streamlit Secrets（云端）中配置"

    # URL支持GroupID参数
    if GROUP_ID:
        url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={GROUP_ID}"
    else:
        url = "https://api.minimax.chat/v1/text/chatcompletion_pro"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 构建消息
    messages = []
    if system_prompt:
        messages.append({
            "sender_type": "USER",
            "sender_name": "User",
            "text": f"系统指令：{system_prompt}"
        })
    messages.append({
        "sender_type": "USER",
        "sender_name": "User",
        "text": prompt
    })

    # Minimax API 要求的完整格式
    payload = {
        "model": "abab6.5-chat",
        "tokens_to_generate": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
        "reply_constraints": {
            "sender_type": "BOT",
            "sender_name": "HydroTune-AI"
        },
        "sample_messages": [],
        "plugins": [],
        "messages": messages,
        "bot_setting": [
            {
                "bot_name": "HydroTune-AI",
                "content": "You are HydroTune-AI, a professional hydrological model intelligence assistant. You are skilled at analyzing hydrological data, explaining model parameters, and generating professional hydrological analysis reports." if get_language() == 'en' else "你是HydroTune-AI，一个专业的水文模型智能助手。你擅长分析水文数据、解释水文模型参数、生成专业的水文分析报告。"
            }
        ],
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 检查返回状态
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            msg = base_resp.get('status_msg', 'unknown error')
            if get_language() == 'en':
                return f"[ERROR] Minimax API error: {msg}"
            return f"[ERROR] Minimax API 错误: {msg}"

        # 解析回复
        reply = data.get("reply", "")
        if reply:
            return reply
        elif "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("message", {}).get("content", "")
        else:
            if get_language() == 'en':
                return "[ERROR] Minimax returned empty content"
            return "[ERROR] Minimax返回空内容"

    except requests.exceptions.Timeout:
        if get_language() == 'en':
            return "[ERROR] Minimax API call timed out"
        return "[ERROR] Minimax API 调用超时"
    except requests.exceptions.RequestException as e:
        if get_language() == 'en':
            return f"[ERROR] Minimax API call failed: {e}"
        return f"[ERROR] Minimax API 调用失败: {e}"
    except Exception as e:
        if get_language() == 'en':
            return f"[ERROR] Error processing response: {e}"
        return f"[ERROR] 处理响应时出错: {e}"

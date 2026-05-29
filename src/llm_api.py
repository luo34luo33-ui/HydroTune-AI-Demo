"""
Minimax API 调用封装
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

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


def call_minimax(prompt: str, system_prompt: str = "") -> str:
    """
    调用 Minimax API，返回生成的文本

    Args:
        prompt: 用户输入
        system_prompt: 系统提示词

    Returns:
        LLM 生成的响应文本
    """
    API_KEY = os.getenv("MINIMAX_API_KEY")
    GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")

    if not API_KEY:
        lang = get_language()
        if lang == 'en':
            return "[ERROR] MINIMAX_API_KEY not found, please configure it in .env"
        return "[ERROR] 未找到MINIMAX_API_KEY，请在.env文件中配置"

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

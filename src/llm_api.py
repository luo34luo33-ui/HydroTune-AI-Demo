"""
Minimax API 调用封装
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()


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
            "sender_name": "Hydromind"
        },
        "sample_messages": [],
        "plugins": [],
        "messages": messages,
        "bot_setting": [
            {
                "bot_name": "Hydromind",
                "content": "你是Hydromind，一个专业的水文模型智能助手。你擅长分析水文数据、解释水文模型参数、生成专业的水文分析报告。"
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
            return f"[ERROR] Minimax API 错误: {base_resp.get('status_msg', '未知错误')}"

        # 解析回复
        reply = data.get("reply", "")
        if reply:
            return reply
        elif "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("message", {}).get("content", "")
        else:
            return "[ERROR] Minimax返回空内容"

    except requests.exceptions.Timeout:
        return "[ERROR] Minimax API 调用超时"
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Minimax API 调用失败: {e}"
    except Exception as e:
        return f"[ERROR] 处理响应时出错: {e}"

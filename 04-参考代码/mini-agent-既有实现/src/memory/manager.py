"""记忆管理器"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import json


class MemoryManager:
    """记忆管理器 - 管理短期记忆和长期记忆"""

    def __init__(self, max_short_term: int = 20, summary_interval: int = 10):
        self.max_short_term = max_short_term
        self.summary_interval = summary_interval
        self.sessions: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "messages": [],
            "summary": "",
            "facts": [],
            "user_profile": {},
            "turn_count": 0
        })

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """获取会话数据"""
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到短期记忆"""
        session = self.sessions[session_id]
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        session["turn_count"] += 1

        # 超过最大轮数，移除最早的消息
        if len(session["messages"]) > self.max_short_term:
            session["messages"] = session["messages"][-self.max_short_term:]

    def get_context_messages(self, session_id: str) -> List[Dict[str, str]]:
        """获取用于 LLM 的上下文消息"""
        session = self.sessions[session_id]
        messages = []

        # 添加系统提示 (如果有摘要)
        if session["summary"]:
            messages.append({
                "role": "system",
                "content": f"之前的对话摘要:\n{session['summary']}"
            })

        # 添加长期记忆中的事实
        if session["facts"]:
            facts_text = "\n".join([f"- {fact}" for fact in session["facts"][-5:]])
            messages.append({
                "role": "system",
                "content": f"已知事实:\n{facts_text}"
            })

        # 添加短期记忆中的消息
        for msg in session["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages

    def should_summarize(self, session_id: str) -> bool:
        """判断是否需要压缩摘要"""
        session = self.sessions[session_id]
        return session["turn_count"] % self.summary_interval == 0

    def update_summary(self, session_id: str, summary: str):
        """更新会话摘要"""
        self.sessions[session_id]["summary"] = summary

    def add_fact(self, session_id: str, fact: str):
        """添加长期记忆事实"""
        session = self.sessions[session_id]
        if fact not in session["facts"]:
            session["facts"].append(fact)
            # 保留最近 20 条事实
            if len(session["facts"]) > 20:
                session["facts"] = session["facts"][-20:]

    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_stats(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计"""
        session = self.sessions[session_id]
        return {
            "message_count": len(session["messages"]),
            "fact_count": len(session["facts"]),
            "turn_count": session["turn_count"],
            "has_summary": bool(session["summary"])
        }

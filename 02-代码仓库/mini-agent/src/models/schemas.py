"""Mini-Agent 数据模型定义"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    knowledge_base: str = "default"
    use_memory: bool = True
    enable_tools: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    memory_updated: bool = False
    tokens_used: int = 0
    latency_ms: float = 0


class KnowledgeUploadRequest(BaseModel):
    filename: str
    content_type: str
    tags: List[str] = []


class DocumentChunk(BaseModel):
    doc_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = {}


class RetrievalResult(BaseModel):
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None


class AgentState(BaseModel):
    messages: List[Message] = []
    context: str = ""
    memory: Dict[str, Any] = {}
    tools_used: List[str] = []
    iteration: int = 0
    intent: str = ""
    tool_calls: List[ToolCall] = []

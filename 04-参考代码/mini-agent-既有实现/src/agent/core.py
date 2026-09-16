"""Agent 核心 — ReAct 循环 + 工具调度 + 流式支持"""
import json
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.llm.factory import LLMFactory, BaseLLMProvider, LLMResponse
from src.rag.pipeline import RAGPipeline
from src.memory.manager import MemoryManager
from src.tools.registry import tool_registry, ToolRegistry

logger = logging.getLogger(__name__)

# ReAct 最大迭代次数（防止死循环）
MAX_ITERATIONS = 5

# 系统提示词模板
SYSTEM_PROMPT_TEMPLATE = """你是 Mini-Agent，一个智能助手。你可以：
1. 回答用户问题（基于知识库和搜索结果）
2. 调用工具执行计算、搜索、代码执行等任务
3. 分析用户上传的文档

规则：
- 优先使用工具获取准确信息，不要编造
- 回答简洁、准确、有帮助
- 如果工具返回错误，如实告知用户
- 使用中文回复
{rag_context}"""


class MiniAgent:
    """Mini-Agent 核心 — 实现 ReAct 推理-行动循环"""

    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: str = "",
        base_url: str = None,
        model: str = "gpt-4o-mini",
        rag: RAGPipeline = None,
        memory: MemoryManager = None,
        tools: ToolRegistry = None,
    ):
        self.llm: BaseLLMProvider = LLMFactory.create(
            provider=llm_provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.rag: RAGPipeline = rag or RAGPipeline()
        self.memory: MemoryManager = memory or MemoryManager()
        self.tools: ToolRegistry = tools or tool_registry

        # 将 RAG pipeline 注入到需要它的工具中
        self._bind_rag_tools()

    # ================================================================== #
    #  工具绑定
    # ================================================================== #
    def _bind_rag_tools(self):
        """将 RAG pipeline 实例注入到知识库相关工具中"""
        rag = self.rag

        def _knowledge_search(query: str, top_k: int = 3) -> str:
            results = rag.retrieve(query, top_k=top_k)
            if not results:
                return "知识库中未找到相关内容"
            parts = []
            for i, r in enumerate(results, 1):
                source = r.document.metadata.get("filename", "未知来源") if r.document.metadata else "未知来源"
                parts.append(f"[来源: {source}, 相关度: {r.score:.2f}]\n{r.document.content}")
            return "\n\n---\n\n".join(parts)

        def _read_file_content(doc_id: str) -> str:
            results = rag.collection.get(where={"doc_id": doc_id})
            if results and results["documents"]:
                return "\n\n".join(results["documents"][:10])
            return f"未找到文档: {doc_id}"

        # 替换工具函数
        if "knowledge_search" in self.tools.tools:
            self.tools.tools["knowledge_search"].function = _knowledge_search
        if "read_file_content" in self.tools.tools:
            self.tools.tools["read_file_content"].function = _read_file_content

    # ================================================================== #
    #  核心对话方法（非流式）
    # ================================================================== #
    async def chat(
        self,
        session_id: str,
        message: str,
        use_memory: bool = True,
        enable_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        处理用户消息 — ReAct 循环

        1. 更新记忆
        2. 检索 RAG 上下文
        3. 构建消息列表
        4. ReAct 循环: LLM → 判断工具 → 执行 → 再次 LLM
        5. 更新记忆 + 返回结果
        """
        start_time = time.time()
        tools_used: List[str] = []

        # 1. 添加用户消息到记忆
        if use_memory:
            self.memory.add_message(session_id, "user", message)

        # 2. 检索 RAG 上下文
        rag_context = self.rag.get_context(message)

        # 3. 构建系统提示
        rag_block = f"\n\n参考资料:\n{rag_context}" if rag_context else ""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_block)

        # 4. 构建消息列表
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if use_memory:
            messages.extend(self.memory.get_context_messages(session_id))
        else:
            messages.append({"role": "user", "content": message})

        # 5. 准备工具定义
        tool_defs = self.tools.get_all_tools() if enable_tools else None

        # 6. ReAct 循环
        reply = ""
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"[ReAct] 迭代 {iteration + 1}/{MAX_ITERATIONS}")

            response: LLMResponse = await self.llm.chat(
                messages=messages,
                tools=tool_defs,
            )

            # —— 无工具调用：LLM 直接回复 ——
            if not response.tool_calls:
                reply = response.content
                break

            # —— 有工具调用：执行工具 ——
            # 将 assistant 消息（含 tool_calls）加入上下文
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                }
                for tc in response.tool_calls
            ]
            messages.append(assistant_msg)

            # 逐个执行工具
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tools_used.append(tool_name)
                logger.info(f"[ReAct] 调用工具: {tool_name}")

                # 解析参数
                try:
                    tool_args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                except json.JSONDecodeError:
                    tool_args = {}

                # 执行
                tool_result = self.tools.execute(tool_name, **tool_args)
                logger.info(f"[ReAct] 工具结果: {tool_result[:200]}...")

                # 将工具结果加入上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })

            # 继续循环，让 LLM 基于工具结果生成回复
        else:
            # 达到最大迭代次数
            reply = f"⚠️ 达到最大推理轮次（{MAX_ITERATIONS}），请简化问题后重试。"

        # 7. 更新记忆
        if use_memory:
            self.memory.add_message(session_id, "assistant", reply)
            # 检查是否需要压缩摘要
            if self.memory.should_summarize(session_id):
                await self._summarize_and_update(session_id)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "session_id": session_id,
            "reply": reply,
            "sources": self._extract_sources(rag_context),
            "tools_used": tools_used,
            "memory_updated": use_memory,
            "tokens_used": response.tokens_used if 'response' in dir() else 0,
            "latency_ms": latency_ms,
        }

    # ================================================================== #
    #  流式对话方法
    # ================================================================== #
    async def chat_stream(
        self,
        session_id: str,
        message: str,
        use_memory: bool = True,
        enable_tools: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        流式对话 — 先处理工具调用（非流式），最后流式输出最终回复

        Yields SSE 格式 JSON 字符串:
          {"type": "thinking", "content": "..."}       — 思考过程
          {"type": "tool_call", "tool": "...", "args": {...}}  — 工具调用
          {"type": "tool_result", "tool": "...", "result": "..."} — 工具结果
          {"type": "content", "content": "..."}         — 流式文本片段
          {"type": "done", "reply": "...", "tools_used": [...]} — 完成
          {"type": "error", "message": "..."}           — 错误
        """
        start_time = time.time()
        tools_used: List[str] = []

        try:
            # 1. 前置处理
            if use_memory:
                self.memory.add_message(session_id, "user", message)

            rag_context = self.rag.get_context(message)
            rag_block = f"\n\n参考资料:\n{rag_context}" if rag_context else ""
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_block)

            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            if use_memory:
                messages.extend(self.memory.get_context_messages(session_id))
            else:
                messages.append({"role": "user", "content": message})

            tool_defs = self.tools.get_all_tools() if enable_tools else None

            # 2. ReAct 循环（工具调用阶段 — 非流式）
            final_messages = messages
            for iteration in range(MAX_ITERATIONS):
                response = await self.llm.chat(messages=final_messages, tools=tool_defs)

                if not response.tool_calls:
                    # 无工具调用，进入流式输出阶段
                    break

                # 广播工具调用信息
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tools_used.append(tool_name)
                    try:
                        tool_args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                    except json.JSONDecodeError:
                        tool_args = {}

                    yield json.dumps({"type": "tool_call", "tool": tool_name, "args": tool_args}, ensure_ascii=False)

                    # 执行工具
                    tool_result = self.tools.execute(tool_name, **tool_args)
                    yield json.dumps({"type": "tool_result", "tool": tool_name, "result": tool_result[:500]}, ensure_ascii=False)

                    # 加入上下文
                    final_messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}
                        }]
                    })
                    final_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result
                    })
            else:
                yield json.dumps({"type": "error", "message": f"达到最大推理轮次 {MAX_ITERATIONS}"}, ensure_ascii=False)
                return

            # 3. 流式输出最终回复
            full_reply = ""
            async for chunk in self.llm.chat_stream(messages=final_messages):
                full_reply += chunk
                yield json.dumps({"type": "content", "content": chunk}, ensure_ascii=False)

            # 4. 完成
            latency_ms = (time.time() - start_time) * 1000
            yield json.dumps({
                "type": "done",
                "reply": full_reply,
                "tools_used": tools_used,
                "latency_ms": round(latency_ms, 1)
            }, ensure_ascii=False)

            # 5. 更新记忆
            if use_memory:
                self.memory.add_message(session_id, "assistant", full_reply)
                if self.memory.should_summarize(session_id):
                    await self._summarize_and_update(session_id)

        except Exception as e:
            logger.exception("chat_stream error")
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)

    # ================================================================== #
    #  辅助方法
    # ================================================================== #
    def _build_system_prompt(self, rag_context: str) -> str:
        rag_block = f"\n\n参考资料:\n{rag_context}" if rag_context else ""
        return SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_block)

    def _extract_sources(self, rag_context: str) -> List[Dict[str, Any]]:
        """从 RAG 上下文中提取来源信息"""
        if not rag_context:
            return []
        sources = []
        results = self.rag.retrieve("", top_k=3)  # 简化：取最近检索结果
        for r in results:
            meta = r.document.metadata or {}
            sources.append({
                "content": r.document.content[:200],
                "score": r.score,
                "filename": meta.get("filename", ""),
                "doc_id": meta.get("doc_id", ""),
            })
        return sources

    async def _summarize_and_update(self, session_id: str):
        """压缩对话摘要"""
        session = self.memory.get_session(session_id)
        messages = session.get("messages", [])
        if len(messages) < 5:
            return

        summary_prompt = (
            "请将以下对话总结为简洁的摘要，保留关键信息和用户偏好：\n\n"
            + "\n".join(f"{m['role']}: {m['content'][:200]}" for m in messages[-10:])
            + "\n\n摘要："
        )
        try:
            response = await self.llm.chat([
                {"role": "user", "content": summary_prompt}
            ])
            self.memory.update_summary(session_id, response.content)
        except Exception as e:
            logger.warning(f"摘要生成失败: {e}")

    def get_memory_stats(self, session_id: str) -> Dict[str, Any]:
        return self.memory.get_stats(session_id)

    async def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """添加文档到知识库"""
        self.rag.add_document(doc_id, content, metadata)

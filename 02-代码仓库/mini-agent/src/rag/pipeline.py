"""RAG 检索管道 — ChromaDB + OpenAI Embedding + 文档分块"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    document: Document
    score: float


class OpenAIEmbeddingFunction:
    """OpenAI Embedding 适配器 — 供 ChromaDB 使用"""

    def __init__(self, api_key: str, base_url: str = None,
                 model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB 调用接口：文本列表 → 向量列表"""
        response = self.client.embeddings.create(
            model=self.model,
            input=input
        )
        return [item.embedding for item in response.data]


class RAGPipeline:
    """RAG 检索管道"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        embedding_model: str = "text-embedding-3-small",
        vector_db_path: str = "./data/vector_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._last_results: List[RetrievalResult] = []  # 缓存最近检索结果

        self._init_vector_store(api_key, base_url, embedding_model, vector_db_path)

    def _init_vector_store(self, api_key: str, base_url: str,
                           embedding_model: str, db_path: str):
        """初始化 ChromaDB + Embedding 函数"""
        import chromadb

        self.client = chromadb.PersistentClient(path=db_path)

        # 构建 Embedding 函数
        embedding_fn = None
        if api_key:
            try:
                embedding_fn = OpenAIEmbeddingFunction(
                    api_key=api_key,
                    base_url=base_url,
                    model=embedding_model
                )
                logger.info(f"使用 OpenAI Embedding: {embedding_model}")
            except Exception as e:
                logger.warning(f"OpenAI Embedding 初始化失败，使用默认: {e}")
        else:
            logger.info("未配置 API Key，使用 ChromaDB 默认 Embedding")

        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    # ================================================================== #
    #  文本分块
    # ================================================================== #
    def split_text(self, text: str) -> List[str]:
        """递归字符分块 — 按段落 > 句子 > 字符分割"""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        # 先按段落分
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # 段落太长，按句子分
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # 重叠：保留最后的部分
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text

                sentences = self._split_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                        current_chunk += (" " if current_chunk else "") + sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                            current_chunk = overlap_text
                        # 如果单个句子就超长，强制截断
                        if len(sent) > self.chunk_size:
                            for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                                chunks.append(sent[i:i + self.chunk_size].strip())
                        else:
                            current_chunk += (" " if current_chunk else "") + sent

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """按中英文句子分割"""
        import re
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    # ================================================================== #
    #  文档操作
    # ================================================================== #
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """添加文档到知识库（分块 + 向量化 + 存储）"""
        chunks = self.split_text(content)
        if not chunks:
            logger.warning(f"文档 {doc_id} 分块后为空")
            return

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "chunk_index": i, **(metadata or {})}
            for i in range(len(chunks))
        ]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"文档 {doc_id} 入库成功: {len(chunks)} 个分块")

    def delete_document(self, doc_id: str):
        """删除文档的所有分块"""
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"文档 {doc_id} 已删除: {len(results['ids'])} 个分块")

    # ================================================================== #
    #  检索
    # ================================================================== #
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """向量相似度检索"""
        k = top_k or self.top_k

        # 知识库为空时返回空
        if self.collection.count() == 0:
            self._last_results = []
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        retrieval_results = []
        if results and results["documents"]:
            for doc, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            ):
                retrieval_results.append(
                    RetrievalResult(
                        document=Document(content=doc, metadata=metadata),
                        score=1 - distance  # cosine distance → similarity
                    )
                )

        self._last_results = retrieval_results
        return retrieval_results

    def get_context(self, query: str) -> str:
        """获取增强上下文（用于注入 system prompt）"""
        results = self.retrieve(query)
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            source = r.document.metadata.get("filename", "") if r.document.metadata else ""
            prefix = f"[参考{i}]" + (f" (来源: {source})" if source else "")
            parts.append(f"{prefix} {r.document.content}")

        return "\n\n".join(parts)

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档（按 doc_id 去重）"""
        all_data = self.collection.get()
        doc_map: Dict[str, Dict] = {}
        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                doc_id = meta.get("doc_id", "")
                if doc_id and doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename", ""),
                        "chunk_count": 0,
                    }
                if doc_id in doc_map:
                    doc_map[doc_id]["chunk_count"] += 1
        return list(doc_map.values())

    def get_doc_content(self, doc_id: str) -> str:
        """获取文档的完整文本（拼接所有分块）"""
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results["documents"]:
            return "\n\n".join(results["documents"])
        return ""

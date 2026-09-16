"""知识库管理 API — 文档上传 / 列表 / 删除 / 搜索"""
import os
import logging
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from src.config import settings
from src.rag.parser import parse_file, get_file_type

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局 Agent 实例引用（延迟获取）
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from src.api.chat import get_agent
        _agent = get_agent()
    return _agent


# ====================================================================== #
#  文档上传
# ====================================================================== #

@router.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: str = ""
):
    """
    上传文档到知识库

    支持格式: PDF, DOCX, TXT, MD, 代码文件等
    流程: 上传 → 解析文本 → 分块 → Embedding → ChromaDB 存储
    """
    # 1. 验证文件大小
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB
        raise HTTPException(status_code=400, detail="文件大小不能超过 20MB")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    # 2. 解析文件内容
    try:
        text = await parse_file(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"文件解析失败: {file.filename}")
        raise HTTPException(status_code=500, detail=f"文件解析失败: {e}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析")

    # 3. 保存文件到磁盘
    os.makedirs(settings.upload_dir, exist_ok=True)
    safe_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(settings.upload_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # 4. 入库（分块 + 向量化 + 存储到 ChromaDB）
    doc_id = f"doc-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    metadata = {
        "filename": file.filename,
        "saved_filename": safe_filename,
        "file_type": get_file_type(file.filename),
        "file_size": len(content),
        "tags": tags,
        "uploaded_at": datetime.now().isoformat(),
    }

    try:
        agent = _get_agent()
        agent.rag.add_document(doc_id, text, metadata=metadata)
    except Exception as e:
        logger.exception(f"文档入库失败: {file.filename}")
        raise HTTPException(status_code=500, detail=f"文档入库失败: {e}")

    chunk_count = len(agent.rag.split_text(text))

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "file_type": get_file_type(file.filename),
        "file_size": len(content),
        "text_length": len(text),
        "chunk_count": chunk_count,
        "status": "uploaded",
        "message": f"文档上传成功，已分为 {chunk_count} 个分块入库"
    }


# ====================================================================== #
#  文档列表
# ====================================================================== #

@router.get("/knowledge/list")
async def list_documents():
    """列出所有知识库文档"""
    try:
        agent = _get_agent()
        docs = agent.rag.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.exception("获取文档列表失败")
        return {"documents": [], "total": 0, "error": str(e)}


# ====================================================================== #
#  文档搜索
# ====================================================================== #

@router.post("/knowledge/search")
async def search_documents(
    query: str = Query(..., description="搜索关键词"),
    top_k: int = Query(3, ge=1, le=10, description="返回结果数")
):
    """在知识库中搜索相关文档"""
    try:
        agent = _get_agent()
        results = agent.rag.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "content": r.document.content,
                    "score": round(r.score, 4),
                    "doc_id": r.document.metadata.get("doc_id", ""),
                    "filename": r.document.metadata.get("filename", ""),
                }
                for r in results
            ],
            "total": len(results)
        }
    except Exception as e:
        logger.exception("搜索失败")
        raise HTTPException(status_code=500, detail=f"搜索失败: {e}")


# ====================================================================== #
#  文档删除
# ====================================================================== #

@router.delete("/knowledge/{doc_id}")
async def delete_document(doc_id: str):
    """删除知识库中的文档"""
    try:
        agent = _get_agent()
        agent.rag.delete_document(doc_id)
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as e:
        logger.exception("删除文档失败")
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

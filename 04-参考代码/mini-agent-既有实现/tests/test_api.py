"""Mini-Agent 测试"""
import pytest
from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


def test_health_check():
    """测试健康检查接口"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_chat():
    """测试聊天接口"""
    response = client.post("/api/v1/chat", json={
        "session_id": "test-user",
        "message": "你好",
        "use_memory": True,
        "enable_tools": True
    })
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["session_id"] == "test-user"


def test_memory():
    """测试记忆接口"""
    # 获取记忆
    response = client.get("/api/v1/memory/test-user")
    assert response.status_code == 200

    # 清除记忆
    response = client.delete("/api/v1/memory/test-user")
    assert response.status_code == 200


def test_knowledge_list():
    """测试知识库列表接口"""
    response = client.get("/api/v1/knowledge/list")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data

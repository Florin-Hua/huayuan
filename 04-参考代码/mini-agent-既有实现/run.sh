#!/bin/bash
# Mini-Agent 快速启动脚本

echo "========================================="
echo "       Mini-Agent 快速启动"
echo "========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python 3.11+"
    exit 1
fi

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "📝 创建 .env 配置文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件，填入你的 OPENAI_API_KEY"
    echo "   然后重新运行此脚本"
    exit 0
fi

# 检查 API Key
if grep -q "OPENAI_API_KEY=sk-xxx" .env || grep -q "OPENAI_API_KEY=$" .env; then
    echo "⚠️  检测到 OPENAI_API_KEY 未配置"
    echo "   请编辑 .env 文件，填入你的 API Key"
    echo ""
    echo "   支持的 LLM 提供商："
    echo "   - OpenAI:    OPENAI_API_KEY=sk-xxx"
    echo "   - DeepSeek:  OPENAI_API_KEY=sk-xxx, OPENAI_BASE_URL=https://api.deepseek.com/v1"
    echo "   - Moonshot:  OPENAI_API_KEY=sk-xxx, OPENAI_BASE_URL=https://api.moonshot.cn/v1"
    echo ""
    exit 0
fi

# 安装依赖
echo "📦 检查并安装依赖..."
pip install -r requirements.txt -q

# 创建数据目录
mkdir -p data/{uploads,vector_db,sqlite}

# 启动服务
echo ""
echo "🚀 启动 Mini-Agent..."
echo "   访问地址: http://localhost:8000"
echo "   API 文档: http://localhost:8000/docs"
echo ""
echo "   按 Ctrl+C 停止服务"
echo "========================================="

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

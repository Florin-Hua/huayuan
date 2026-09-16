@echo off
REM Mini-Agent 快速启动脚本 (Windows)

echo =========================================
echo        Mini-Agent 快速启动
echo =========================================

REM 检查 .env 文件
if not exist .env (
    echo [1/3] 创建 .env 配置文件...
    copy .env.example .env
    echo.
    echo 请编辑 .env 文件，填入你的 OPENAI_API_KEY
    echo 然后重新运行此脚本
    pause
    exit /b
)

REM 检查 API Key
findstr /C:"OPENAI_API_KEY=sk-xxx" .env >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo 检测到 OPENAI_API_KEY 未配置
    echo 请编辑 .env 文件，填入你的 API Key
    echo.
    echo 支持的 LLM 提供商：
    echo - OpenAI:    OPENAI_API_KEY=sk-xxx
    echo - DeepSeek:  OPENAI_API_KEY=sk-xxx, OPENAI_BASE_URL=https://api.deepseek.com/v1
    echo - Moonshot:  OPENAI_API_KEY=sk-xxx, OPENAI_BASE_URL=https://api.moonshot.cn/v1
    echo.
    pause
    exit /b
)

REM 安装依赖
echo [2/3] 检查依赖...
pip install -r requirements.txt -q

REM 创建数据目录
if not exist data\uploads mkdir data\uploads
if not exist data\vector_db mkdir data\vector_db
if not exist data\sqlite mkdir data\sqlite

REM 启动服务
echo [3/3] 启动服务...
echo.
echo =========================================
echo    访问地址: http://localhost:8000
echo    API 文档: http://localhost:8000/docs
echo =========================================
echo.

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

pause

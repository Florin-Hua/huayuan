FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/uploads /app/data/vector_db /app/data/sqlite

# 暴露端口
EXPOSE 8000

# 启动
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

.PHONY: help install dev test docker-up docker-down clean

# 默认目标
help: ## 显示帮助信息
	@echo "Mini-Agent 常用命令:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 安装
install: ## 安装依赖
	pip install -r requirements.txt

dev: ## 开发模式启动
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test: ## 运行测试
	pytest tests/ -v

# Docker 命令
docker-up: ## Docker 启动
	docker-compose up -d --build

docker-down: ## Docker 停止
	docker-compose down

docker-logs: ## 查看日志
	docker-compose logs -f

# 清理
clean: ## 清理缓存和临时文件
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

# 环境初始化
setup: ## 初始化项目环境
	cp .env.example .env
	@echo "请编辑 .env 文件填入你的配置"

# 部署指南

本文档详细介绍如何部署 Andy-LLM 模型服务。

## 1. 环境准备

### 1.1 系统要求
- 操作系统：Linux/macOS/Windows
- CPU: 4核或以上
- 内存：至少 8GB（推荐 16GB）
- 存储：至少 10GB 可用空间
- Python 3.8+

### 1.2 硬件支持
- CPU 部署：所有平台
- NVIDIA GPU：CUDA 11.7+，显存 ≥ 8GB
- Apple Silicon：macOS 12.3+，支持 MPS 后端

### 1.3 依赖安装

1. 创建并激活虚拟环境：
```bash
python -m venv andyllm
source andyllm/bin/activate  # Linux/Mac
# 或
.\andyllm\Scripts\activate  # Windows
```

2. 安装部署依赖：
```bash
# 方式1：直接安装部署依赖
pip install -r requirements/deploy.txt

# 方式2：通过 setup.py 安装
pip install -e .

# 可选：如果需要完整功能（包括训练）
pip install -e ".[all]"
```

部署环境依赖说明：
- torch: PyTorch 推理支持
- transformers: 模型加载和推理
- fastapi & uvicorn: Web 服务
- gunicorn: WSGI 服务器
- prometheus-client: 监控指标
- python-json-logger: 结构化日志
- ninja: 性能优化
- python-jose & passlib: 安全认证
...

3. 根据硬件安装特定依赖：

对于 NVIDIA GPU：
```bash
# 安装 CUDA 版本的 PyTorch
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
```

对于 Apple Silicon：
```bash
# 安装支持 MPS 的 PyTorch
pip install torch torchvision torchaudio
```

4. 安装项目依赖：
```bash
# 克隆项目
git clone https://github.com/DWG-ShowMaker/andy-llm.git
cd andy-llm

# 安装项目依赖
pip install -e .
```

## 2. 模型准备

### 2.1 转换模型格式
将训练好的模型转换为部署格式：

```bash
python scripts/convert_to_vllm.py \
    --model_path checkpoints/best_model.pt \
    --output_dir vllm_model \
    --tokenizer_path data/processed/tokenizer.model
```

### 2.2 验证模型文件
确保 `vllm_model` 目录包含以下文件：
```
vllm_model/
├── config.json           # 模型配置
├── pytorch_model.bin     # 模型权重
├── tokenizer.model       # 分词器模型
└── tokenizer_config.json # 分词器配置
```

## 3. 服务部署

### 3.1 基本部署
最简单的部署方式，适合测试和开发：

```bash
python scripts/serve_model.py \
    --model vllm_model \
    --tokenizer_path data/processed/tokenizer.model \
    --device cpu \
    --port 8000
```

### 3.2 生产部署

1. 使用 Gunicorn 部署（Linux/macOS）：
```bash
pip install gunicorn
gunicorn scripts.serve_model:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

2. 使用 Supervisor 管理进程：
```ini
[program:andyllm]
command=/path/to/andyllm/bin/gunicorn scripts.serve_model:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
directory=/path/to/andy-llm
user=username
autostart=true
autorestart=true
stderr_logfile=/var/log/andyllm/err.log
stdout_logfile=/var/log/andyllm/out.log
```

3. 使用 Nginx 反向代理：
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3.3 Docker 部署

1. 构建镜像：
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8000

CMD ["python", "scripts/serve_model.py", "--model", "vllm_model", "--device", "cpu", "--port", "8000"]
```

2. 运行容器：
```bash
docker build -t andyllm .
docker run -p 8000:8000 andyllm
```

## 4. 性能优化

### 4.1 模型优化
```bash
# 半精度推理
python scripts/serve_model.py \
    --model vllm_model \
    --device cuda \
    --fp16 \
    --port 8000

# 量化模型
python scripts/serve_model.py \
    --model vllm_model \
    --device cuda \
    --quantize int8 \
    --port 8000
```

### 4.2 服务优化
- 启用请求批处理
- 使用响应流式传输
- 启用缓存
- 设置合适的超时时间

## 5. 监控和日志

### 5.1 Prometheus 监控
```python
from prometheus_client import Counter, Histogram

# 添加指标
request_counter = Counter('model_requests_total', 'Total model requests')
latency_histogram = Histogram('model_latency_seconds', 'Request latency')
```

### 5.2 日志配置
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('andyllm.log'),
        logging.StreamHandler()
    ]
)
```

## 6. 故障排除

### 6.1 常见问题

1. 内存不足：
- 减小批处理大小
- 启用梯度检查点
- 使用模型量化

2. GPU 问题：
- 检查 CUDA 版本
- 监控 GPU 使用率
- 清理 GPU 缓存

3. 性能问题：
- 使用性能分析工具
- 优化批处理大小
- 调整工作进程数

### 6.2 性能调优

1. 系统级优化：
```bash
# 增加文件描述符限制
ulimit -n 65535

# 优化内核参数
sysctl -w net.core.somaxconn=65535
```

2. 应用级优化：
```python
# 启用批处理
app.batch_size = 32
app.max_batch_wait_time = 0.1
```

## 7. 安全配置

1. 启用 HTTPS：
```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem"
)
```

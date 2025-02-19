# Andy-LLM: 轻量级对话语言模型

Andy-LLM 是一个基于 Transformer 架构的轻量级对话语言模型，专注于中文对话生成。本项目包含模型训练、数据处理和推理部署的完整流程。

## 特性

- 轻量级 Transformer 架构
- 支持上下文对话
- 混合精度训练
- 量化推理
- 支持 CPU/GPU/MPS 设备
- 对话数据格式处理
- 增量式推理

## 优化策略

### 1. 数据质量优化
- 过滤过短文本（少于50字符）
- 过滤重复内容（少于2个句号的文本）
- 过滤广告等垃圾文本
- 支持多数据源混合（新闻、维基百科、问答）

### 2. 训练优化
- 混合精度训练 (FP16)
- 梯度累积
- 多种学习率调度策略
- 早停机制
- 详细的评估指标

### 3. 推理优化
- KV缓存
- 重复惩罚
- 长度惩罚
- 支持多种采样策略

## 项目结构

```
andy-llm/
├── src/
│   ├── data/           # 数据处理
│   ├── model/          # 模型定义
│   ├── training/       # 训练相关
│   └── utils/          # 工具函数
├── scripts/            # 脚本文件
├── tests/              # 测试文件
├── configs/            # 配置文件
├── data/               # 数据目录
│   ├── raw/           # 原始数据
│   └── processed/     # 处理后数据
└── outputs/            # 输出目录
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv andyllm
source andyllm/bin/activate  # Linux/Mac
# 或
.\andyllm\Scripts\activate  # Windows

# 安装依赖
pip install -e ".[train]"
```

### 2. 数据准备

```bash
# 下载数据集
python scripts/download_data.py --datasets muice --output_dir data/raw

# 训练分词器
python scripts/train_tokenizer.py \
    --input_file data/raw/muice_train.jsonl \
    --model_prefix data/processed/tokenizer \
    --vocab_size 32000
```

### 3. 模型训练

```bash
# 使用 GPU 训练
python scripts/train.py \
    --device cuda \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir outputs/my_model

# 使用 CPU 训练
python scripts/train.py --device cpu
```

### 4. 模型推理

```bash
# 交互式对话
python scripts/chat.py \
    --model_path outputs/my_model \
    --device cuda

# API 服务
python scripts/serve.py --port 8000
```

## 训练配置

主要的训练参数：

```python
# 模型配置
vocab_size: int = 32000    # 词表大小
d_model: int = 512        # 隐藏层维度
nhead: int = 8           # 注意力头数
num_layers: int = 6      # Transformer层数
dim_feedforward: int = 2048  # 前馈网络维度
max_seq_length: int = 512    # 最大序列长度
dropout: float = 0.1     # Dropout比率

# 训练配置
batch_size: int = 8      # 批次大小
learning_rate: float = 1e-4  # 学习率
num_epochs: int = 3      # 训练轮数
warmup_steps: int = 100  # 预热步数
weight_decay: float = 0.01  # 权重衰减
max_grad_norm: float = 1.0  # 梯度裁剪
fp16: bool = True        # 混合精度训练
```

## API 使用

```python
from src.model import MiniLLM
from src.utils import Tokenizer

# 加载模型
model = MiniLLM.from_pretrained('outputs/my_model')
tokenizer = Tokenizer.from_pretrained('data/processed/tokenizer.model')

# 对话
response = model.chat(
    tokenizer=tokenizer,
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ],
    max_length=512,
    temperature=0.7
)
print(response)
```

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t andy-llm .

# 运行容器
docker run -d \
    -p 8000:8000 \
    -v /path/to/model:/app/model \
    andy-llm
```

### 本地部署

```bash
# 安装生产环境依赖
pip install -e ".[serve]"

# 启动服务
python scripts/serve.py \
    --model_path outputs/my_model \
    --port 8000
```

## 性能优化

1. 量化
```python
# INT8 量化
model.quantize(quantization_type='dynamic')
```

2. KV 缓存
```python
# 启用 KV 缓存
model.enable_kv_cache()
```

3. 批处理推理
```python
responses = model.batch_generate(
    prompts,
    batch_size=4
)
```

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 引用

如果您使用了本项目，请引用：

```bibtex
@software{andy_llm,
    title = {Andy-LLM: Lightweight Dialogue Language Model},
    author = {Your Name},
    year = {2024},
    url = {https://github.com/yourusername/andy-llm}
}
```
# Andy-LLM

一个基于 Transformer 架构的轻量级中文语言模型项目。

## 特性

- 基于 Transformer 架构的自回归语言模型
- 支持中文分词和生成
- 轻量级设计，易于训练和部署
- 支持多种设备（CPU/GPU/MPS）
- 提供 REST API 服务
- 完整的训练和推理流程
- 内置实验追踪
- 支持多种优化策略

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
├── src/                    # 源代码
│   ├── model/             # 模型定义
│   │   ├── config.py      # 模型配置
│   │   └── transformer.py # Transformer 模型实现
│   ├── data/              # 数据处理
│   │   └── dataset.py     # 数据集实现
│   ├── training/          # 训练相关
│   │   └── trainer.py     # 训练器
│   └── tokenizer.py       # 分词器实现
├── scripts/               # 工具脚本
│   ├── download_data.py   # 数据下载
│   ├── preprocess_data.py # 数据预处理
│   ├── train.py          # 训练脚本
│   ├── convert_to_vllm.py # 模型转换工具
│   ├── serve_model.py     # 模型服务器
│   └── test_vllm.py       # 测试脚本
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── checkpoints/          # 模型检查点
└── docs/                 # 文档
    ├── training.md       # 训练指南
    └── deployment.md     # 部署文档
```

## 快速开始

### 1. 环境配置
```bash
# 克隆项目
git clone https://github.com/DWG-ShowMaker/andy-llm.git
cd andy-llm

# 创建虚拟环境
python -m venv andyllm
source andyllm/bin/activate  # Linux/Mac
# 或
.\andyllm\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 2. 数据准备
```bash
# 下载和处理数据
python scripts/download_data.py
```

### 3. 模型训练
```bash
# 基础训练
python scripts/train.py \
    --model_size small \
    --batch_size 32 \
    --learning_rate 2e-4

# 优化训练
python scripts/train.py \
    --model_size small \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --warmup_steps 2000 \
    --num_epochs 20 \
    --dropout 0.1 \
    --gradient_accumulation_steps 4 \
    --scheduler cosine \
    --patience 3 \
    --min_delta 1e-4
```

### 4. 部署服务
```bash
# 转换模型
python scripts/convert_to_vllm.py \
    --model_path checkpoints/best_model.pt \
    --output_dir vllm_model \
    --tokenizer_path data/processed/tokenizer.model

# 启动服务
python scripts/serve_model.py \
    --model vllm_model \
    --tokenizer_path data/processed/tokenizer.model \
    --device mps \
    --port 8000
```

### 5. 测试服务
```bash
python scripts/test_vllm.py \
    --prompt "请介绍一下你自己" \
    --url "http://localhost:8000"
```

## 训练参数说明

- `dropout`: 随机丢弃率，防止过拟合
- `attention_dropout`: 注意力层的丢弃率
- `hidden_dropout`: 隐藏层的丢弃率
- `gradient_accumulation_steps`: 梯度累积步数
- `scheduler`: 学习率调度策略 (cosine/linear)
- `patience`: 早停耐心值
- `min_delta`: 早停最小改善阈值

## 推理参数说明

- `temperature`: 采样温度，控制生成的随机性
- `top_k`: Top-K采样的K值
- `top_p`: 核采样的概率阈值
- `repetition_penalty`: 重复惩罚系数
- `length_penalty`: 长度惩罚系数
- `num_beams`: 束搜索的束宽

## 文档

- [训练指南](docs/training.md)
- [部署指南](docs/deployment.md)
- [API 文档](docs/api.md)

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- Transformers
- FastAPI
- SentencePiece
- Datasets
- Wandb

## 许可证

MIT License
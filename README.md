# Andy-LLM: 轻量级中文对话语言模型

Andy-LLM 是一个基于 Transformer 架构的轻量级中文对话语言模型，专注于提供简单易用的训练和部署流程。

## 特性

- 轻量级 Transformer 架构
- 支持三种规模配置：tiny(5M)、small(20M)、medium(100M)
- 混合精度训练和量化推理
- 增量式对话生成
- 支持 CPU/GPU/MPS 设备
- 完整的训练部署流程

## 项目结构

```
andy-llm/
├── src/                      # 核心源代码
│   ├── data/                # 数据处理相关
│   │   ├── dataset.py       # 数据集实现
│   │   ├── format.py        # 数据格式化
│   │   └── dataset_config.py # 数据集配置
│   ├── model/               # 模型相关
│   │   ├── transformer.py   # Transformer实现
│   │   └── config.py        # 模型配置
│   ├── training/            # 训练相关
│   │   ├── trainer.py       # 训练器
│   │   ├── config.py        # 训练配置
│   │   └── early_stopping.py # 早停机制
│   └── tokenizer.py         # 分词器实现
│
├── scripts/                  # 命令行脚本
│   ├── data/                # 数据处理脚本
│   │   ├── download_data.py  # 下载数据
│   │   └── preprocess_data.py # 数据预处理
│   ├── train/               # 训练相关脚本
│   │   ├── train.py         # 训练入口
│   │   └── quantize.py      # 模型量化
│   └── serve/               # 服务部署脚本
│       ├── serve_model.py    # 原生服务
│       ├── serve_vllm.py     # vLLM服务
│       └── test_vllm.py      # 服务测试
│
├── configs/                  # 配置文件
│   └── model/               # 模型配置
│       ├── tiny.json
│       ├── small.json
│       └── medium.json
```

## 快速开始

### 1. 安装

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
pip install -e ".[train]"  # 安装训练依赖
# 或
pip install -e "."         # 仅安装推理依赖
```

### 2. 数据准备

```bash
# 下载示例数据集
python scripts/data/download_data.py \
    --datasets muice \
    --output_dir data/raw

# 数据预处理
python scripts/data/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --vocab_size 32000
```

### 3. 模型训练

```bash
# 使用默认配置训练
python scripts/train/train.py \
    --model_size small \
    --train_file data/processed/train.jsonl \
    --val_file data/processed/validation.jsonl \
    --tokenizer_path data/processed/tokenizer.model \
    --output_dir outputs/my_model

# 自定义训练参数
python scripts/train/train.py \
    --model_size medium \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --num_epochs 5 \
    --fp16 \
    --device cuda
```

### 4. 模型量化

```bash
# 动态量化
python scripts/train/quantize.py \
    --model_path outputs/best_model.pt \
    --output_path outputs/quantized.pt \
    --quantization_type dynamic \
    --tokenizer_path data/processed/tokenizer.model

# 静态量化
python scripts/train/quantize.py \
    --model_path outputs/best_model.pt \
    --output_path outputs/quantized_static.pt \
    --quantization_type static \
    --tokenizer_path data/processed/tokenizer.model
```

### 5. 推理服务

```bash
# 启动服务
python scripts/serve/serve_model.py \
    --model outputs/quantized.pt \
    --tokenizer_path data/processed/tokenizer.model \
    --device cuda \
    --port 8000

# 测试服务
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "你好，请介绍一下自己。", "temperature": 0.7}'
```

## 模型规格

| 参数 | Tiny | Small | Medium |
|------|------|--------|---------|
| 词表大小 | 10,000 | 30,000 | 50,000 |
| 隐藏维度 | 128 | 256 | 512 |
| 注意力头数 | 2 | 4 | 8 |
| 层数 | 2 | 4 | 6 |
| 参数量 | ~5M | ~20M | ~100M |
| 最小显存 | 2GB | 4GB | 8GB |

## 性能优化

1. 训练优化
```bash
# 混合精度训练
python scripts/train/train.py --fp16

# 梯度累积
python scripts/train/train.py --gradient_accumulation_steps 4

# 使用 Cosine 学习率调度
python scripts/train/train.py --scheduler cosine
```

2. 推理优化
```bash
# 量化推理
python scripts/serve/serve_model.py --quantize int8

# 批处理推理
python scripts/serve/serve_model.py --batch_size 32

# 启用 KV 缓存
python scripts/serve/serve_model.py --enable_cache
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 引用

如果您使用了本项目，请引用：

```bibtex
@software{andy_llm,
    title = {Andy-LLM: Lightweight Chinese Dialogue Language Model},
    author = {Boss Andy},
    year = {2024},
    url = {https://github.com/DWG-ShowMaker/andy-llm}
}
```

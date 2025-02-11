# Andy-LLM: 轻量级中文语言模型

## 项目简介
Andy-LLM 是一个从零开始实现的轻量级中文语言模型项目。该项目实现了一个基于 Transformer 架构的小型语言模型，支持中文文本的生成和续写。项目特点：

- 完整的训练和推理流程
- 模块化的代码结构
- 详细的注释和文档
- 支持多种规模的模型配置
- 内置实验追踪和可视化

## 项目目标
1. 实现一个基础的transformer模型架构
2. 准备和预处理训练数据
3. 训练模型并评估效果
4. 提供简单的推理接口

## 技术栈
- Python 3.8+
- PyTorch 2.0+
- SentencePiece (分词器)
- Weights & Biases (实验追踪)
- tqdm (进度显示)

## 项目结构
```text
andy-llm/
├── data/                    # 训练数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 预处理后的数据
├── src/                    
│   ├── model/              # 模型相关代码
│   │   ├── transformer.py  # transformer模型实现
│   │   └── config.py      # 模型配置
│   ├── data/               # 数据处理相关代码
│   │   ├── dataset.py     # 数据集实现
│   │   └── tokenizer.py   # 分词器
│   └── training/          
│       ├── trainer.py     # 训练器
│       └── utils.py       # 工具函数
├── scripts/
│   ├── train.py           # 训练脚本
│   └── inference.py       # 推理脚本
├── checkpoints/           # 模型检查点保存目录
└── requirements.txt       # 项目依赖
```

## 快速开始

### 1. 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 2. 数据准备
```bash
# 下载并预处理数据
python scripts/download_data.py
python scripts/preprocess_data.py
```

### 3. 模型训练
```bash
# 使用默认配置训练
python scripts/train.py

# 使用自定义配置训练
python scripts/train.py \
    --model_size small \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --num_epochs 20 \
    --warmup_steps 2000
```

### 4. 文本生成
```bash
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --tokenizer_path data/processed/tokenizer.model \
    --prompt "今天天气真不错" \
    --max_length 100
```

## 模型配置

### 可选的模型大小
1. tiny (适合测试)
   - vocab_size: 10000
   - d_model: 128
   - nhead: 2
   - num_layers: 2
   - 参数量：约5M

2. small (推荐配置)
   - vocab_size: 30000
   - d_model: 256
   - nhead: 4
   - num_layers: 4
   - 参数量：约20M

3. medium (需要更多算力)
   - vocab_size: 50000
   - d_model: 512
   - nhead: 8
   - num_layers: 6
   - 参数量：约100M

## 训练参数说明

### 必要参数
- `--model_size`: 模型大小 [tiny/small/medium]
- `--batch_size`: 训练批次大小
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数

### 可选参数
- `--max_length`: 最大序列长度，默认512
- `--warmup_steps`: 预热步数，默认1000
- `--weight_decay`: 权重衰减，默认0.01
- `--max_grad_norm`: 梯度裁剪阈值，默认1.0
- `--save_every`: 保存模型的轮数间隔，默认1

## 训练技巧

### 1. 显存优化
如果显存不足，可以：
- 减小batch_size
- 减小max_length
- 选择更小的模型配置（tiny）
- 使用梯度累积（gradient accumulation）

### 2. 训练稳定性
如果训练不稳定，可以：
- 降低学习率（如 1e-4）
- 增加warmup_steps（如 2000）
- 调整weight_decay（如 0.1）
- 使用更小的max_grad_norm（如 0.5）

### 3. 性能监控
- 使用wandb监控训练过程
- 关注验证集损失变化
- 观察学习率调度曲线
- 监控梯度范数

## 硬件要求
- 最小配置：
  - GPU: 8GB显存
  - RAM: 16GB内存
  - 存储: 20GB空间

- 推荐配置：
  - GPU: 16GB显存
  - RAM: 32GB内存
  - 存储: 50GB空间

## 常见问题

1. OOM（显存不足）
   - 首先尝试减小batch_size
   - 如果仍然不够，可以减小模型大小或序列长度

2. 训练损失不收敛
   - 检查学习率是否合适
   - 增加warmup_steps
   - 确保数据预处理正确

3. 生成效果不理想
   - 增加训练轮数
   - 使用更大的模型配置
   - 调整采样参数（temperature/top_p/top_k）

## 后续开发计划
1. 添加分布式训练支持
2. 实现模型量化功能
3. 添加更多中文预训练数据集
4. 支持模型压缩和蒸馏

## 贡献指南
欢迎提交Issue和Pull Request！

## 许可证
MIT License

## 联系方式
如有问题，请提交Issue或发送邮件至：746144374@qq.com
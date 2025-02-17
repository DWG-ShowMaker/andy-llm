# Andy-LLM(原名：miniLLM): 轻量级中文语言模型

## 项目简介
Andy-LLM 是一个基于 PyTorch 实现的轻量级中文语言模型训练框架。该项目旨在帮助开发者理解和实践 Transformer 架构的语言模型训练过程。

### 特点
- 完整的训练和推理流程
- 清晰的代码结构和注释
- 灵活的模型配置
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

## 开发环境要求

### 基础环境
- Python 3.8+
- CUDA 11.7+ (如果使用GPU)
- 16GB+ 内存
- 8GB+ GPU显存 (推荐)

### 依赖包
```bash
torch>=2.0.0
numpy>=1.21.0
transformers>=4.30.0
datasets>=2.12.0
tqdm>=4.65.0
sentencepiece>=0.1.99
wandb>=0.15.0
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

数据来源包括：
- CLUE新闻数据集
- 中文维基百科摘要 (2022年3月版)
- 社区问答数据
- CLUEWSC2020数据集 (备用)

数据处理包括：
- 文本长度过滤 (>50字符)
- 句子完整性过滤 (>2个句号)
- 垃圾文本过滤 (广告、重复内容等)
- 数据集划分 (训练集80%、验证集10%、测试集10%)

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

### 4. 模型推理
```bash
# 基础推理
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --tokenizer_path data/processed/tokenizer.model \
    --prompt "范闲苦着脸说啥？"

# 优化推理
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --tokenizer_path data/processed/tokenizer.model \
    --prompt "范闲苦着脸说啥？" \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 0.9 \
    --repetition_penalty 1.2 \
    --length_penalty 1.0 \
    --num_beams 4
```

## 优化参数说明

### 训练参数
- `dropout`: 随机丢弃率，防止过拟合
- `attention_dropout`: 注意力层的丢弃率
- `hidden_dropout`: 隐藏层的丢弃率
- `gradient_accumulation_steps`: 梯度累积步数
- `scheduler`: 学习率调度策略 (cosine/linear)
- `patience`: 早停耐心值
- `min_delta`: 早停最小改善阈值

### 推理参数
- `temperature`: 采样温度，控制生成的随机性
- `top_k`: Top-K采样的K值
- `top_p`: 核采样的概率阈值
- `repetition_penalty`: 重复惩罚系数
- `length_penalty`: 长度惩罚系数
- `num_beams`: 束搜索的束宽

## 性能优化建议

1. **显存优化**:
   - 使用混合精度训练
   - 启用梯度累积
   - 适当调整batch_size

2. **训练稳定性**:
   - 使用warmup预热
   - 启用梯度裁剪
   - 合理设置学习率

3. **生成质量**:
   - 调整采样参数
   - 使用重复惩罚
   - 启用束搜索

4. **推理速度**:
   - 启用KV缓存
   - 使用批量推理
   - 考虑模型量化

## 项目结构
```
andy-llm/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── src/                    # 源代码
│   ├── model/             # 模型定义
│   │   ├── config.py     # 模型配置
│   │   └── transformer.py # 模型实现
│   ├── data/              # 数据处理
│   │   └── dataset.py    # 数据集实现
│   └── training/          # 训练相关
│       └── trainer.py     # 训练器
├── scripts/               # 脚本文件
│   ├── download_data.py  # 数据下载
│   ├── preprocess_data.py # 数据预处理
│   ├── train.py          # 训练脚本
│   └── inference.py      # 推理脚本
└── docs/                 # 文档
    ├── training.md       # 训练指南
    └── deployment.md     # 部署指南
```

## 常见问题

### 1. 训练不收敛
检查以下配置：
```bash
python scripts/train.py \
    --learning_rate 1e-4 \
    --warmup_steps 2000 \
    --gradient_accumulation_steps 4
```

### 2. 生成质量差
调整以下参数：
```bash
python scripts/inference.py \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.2 \
    --num_beams 4
```

### 3. 训练速度慢
启用以下优化：
```bash
python scripts/train.py \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --scheduler cosine
```

## 待办事项
- [x] 添加混合精度训练
- [x] 添加梯度累积
- [x] 添加早停机制
- [ ] 添加模型量化
- [ ] 添加分布式训练
- [ ] 优化数据处理流程

## 贡献指南
欢迎提交 Issue 和 Pull Request！在提交 PR 之前，请确保：
1. 代码风格符合项目规范
2. 添加必要的注释和文档
3. 所有测试通过

## 许可证
MIT License

## 联系方式
- 邮箱：746144374@qq.com
- Issue：[GitHub Issues](https://github.com/DWG-ShowMaker/andy-llm/issues)
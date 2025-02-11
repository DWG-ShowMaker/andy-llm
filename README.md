# Andy-LLM: 轻量级中文语言模型

## 项目简介
Andy-LLM 是一个从零开始实现的轻量级中文语言模型项目。该项目实现了一个基于 Transformer 架构的小型语言模型，支持中文文本的生成和续写。

### 特点
- 完整的训练和推理流程
- 模块化的代码结构
- 详细的注释和文档
- 支持多种规模的模型配置
- 内置实验追踪和可视化
- 支持多种部署方式

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
```
andy-llm/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── src/                    # 源代码
│   ├── model/             # 模型定义
│   ├── data/              # 数据处理
│   └── training/          # 训练相关
├── scripts/               # 脚本文件
├── docs/                  # 文档
└── tests/                 # 测试文件
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
pip install -e .
```

### 2. 训练模型
```bash
# 准备数据
python scripts/prepare_data.py

# 开始训练
python scripts/train.py \
    --model_size tiny \
    --batch_size 32 \
    --num_epochs 10
```

### 3. 推理测试
```bash
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --tokenizer_path data/processed/tokenizer.model \
    --prompt "今天天气真不错" \
    --max_length 100
```

## 模型规格

### Tiny版本 (默认)
- 词表大小: 10,000
- 模型维度: 128
- 注意力头数: 2
- 层数: 2
- 参数量: ~5M

### Small版本
- 词表大小: 30,000
- 模型维度: 256
- 注意力头数: 4
- 层数: 4
- 参数量: ~20M

### Medium版本
- 词表大小: 50,000
- 模型维度: 512
- 注意力头数: 8
- 层数: 6
- 参数量: ~100M

## 部署方式

### 1. Ollama部署
```bash
# 构建模型
ollama create andy-llm -f Modelfile

# 运行模型
ollama run andy-llm
```

## 文档
- [训练指南](docs/training.md)
- [部署指南](docs/deployment.md)
- [LLM训练入门指南](docs/llm_guide.md)

## 硬件要求

### 最小配置
- GPU: 8GB显存
- RAM: 16GB内存
- 存储: 20GB空间

### 推荐配置
- GPU: 16GB显存
- RAM: 32GB内存
- 存储: 50GB空间

## 常见问题

### 1. 训练相关
- **显存不足**: 减小batch_size或使用梯度累积
- **训练不收敛**: 检查学习率和预热步数
- **生成质量差**: 增加训练轮数或调整采样参数

### 2. 部署相关
- **推理速度慢**: 启用KV缓存或使用TensorRT
- **服务不稳定**: 配置负载均衡和请求队列
- **内存泄漏**: 定期重启服务或使用监控

## 贡献指南
欢迎提交Issue和Pull Request！在提交PR之前，请确保：
1. 代码通过所有测试
2. 添加必要的文档和注释
3. 遵循项目的代码规范

## 许可证
MIT License

## 联系方式
- 邮箱：746144374@qq.com
- Issue：[GitHub Issues](https://github.com/DWG-ShowMaker/andy-llm/issues)
# CUDA 环境下的训练和部署指南

本文档将指导你如何在 CUDA 环境下训练和部署 AndyLLM 模型。

## 环境要求

- NVIDIA GPU 显卡 (建议至少 8GB 显存)
- CUDA Toolkit 11.7 或更高版本
- cuDNN 8.0 或更高版本
- PyTorch 2.0 或更高版本 (CUDA 版本)

## 安装步骤

1. 安装 CUDA Toolkit
```bash
# 下载并安装 CUDA Toolkit
# 访问 https://developer.nvidia.com/cuda-downloads 选择对应版本
```

2. 安装 cuDNN
```bash
# 下载并安装 cuDNN
# 访问 https://developer.nvidia.com/cudnn 下载对应版本
```

3. 安装 PyTorch (CUDA 版本)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. 验证 CUDA 是否可用
```python
import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前 CUDA 版本: {torch.version.cuda}")
print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
```

## 训练模型

1. 准备数据
```bash
python scripts/data/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --vocab_size 32000
```

2. 使用 CUDA 训练模型
```bash
python scripts/train/train.py \
    --train_file data/processed/train.jsonl \
    --val_file data/processed/validation.jsonl \
    --tokenizer_path data/processed/tokenizer.model \
    --output_dir outputs/my_model \
    --device cuda \  # 指定使用 CUDA
    --batch_size 32 \  # 根据显存大小调整
    --learning_rate 5e-5 \
    --num_epochs 10
```

### 训练参数优化

- **批次大小**: 根据显存大小调整，一般显存越大可以使用更大的批次
  - 8GB 显存: 16-32
  - 16GB 显存: 32-64
  - 24GB+ 显存: 64-128

- **梯度累积**: 如果显存不足，可以使用梯度累积
```bash
python scripts/train/train.py \
    --gradient_accumulation_steps 4 \  # 累积 4 个批次的梯度
    --batch_size 8  # 实际批次大小 = 8 * 4 = 32
```

- **混合精度训练**: 启用 FP16 训练以节省显存
```bash
python scripts/train/train.py \
    --fp16 \  # 启用混合精度训练
    --batch_size 64  # 可以使用更大的批次
```

## 模型部署

### 1. 标准部署

1. 量化模型 (可选，适用于 CPU 部署)
```bash
python scripts/train/quantize.py \
    --model_path outputs/my_model/best_model.pt \
    --output_path outputs/my_model/quantized_model.pt \
    --quantization_type dynamic
```

2. 启动标准服务
```bash
# 使用 CUDA
python scripts/serve/serve_model.py \
    --model outputs/my_model/best_model.pt \
    --tokenizer_path data/processed/tokenizer.model \
    --device cuda \
    --port 8000
```

### 2. vLLM 高性能部署

vLLM 是一个高性能的 LLM 推理引擎，可以显著提升模型的推理性能。

1. 安装 vLLM
```bash
pip install vllm
```

2. 转换模型为 vLLM 格式
```bash
python scripts/serve/convert_to_vllm.py \
    --model_path outputs/my_model/best_model.pt \
    --output_dir outputs/my_model_vllm \
    --tokenizer_path data/processed/tokenizer.model
```

3. 使用 vLLM 启动服务
```bash
python -m vllm.entrypoints.openai.api_server \
    --model outputs/my_model_vllm \
    --port 8000 \
    --tensor-parallel-size 1  # 根据 GPU 数量调整
```

### 性能对比

以下是不同部署方式的性能对比：

| 部署方式 | GPU | 批处理大小 | 吞吐量 (tokens/s) | 延迟 P90 (ms) |
|---------|-----|-----------|-----------------|--------------|
| 标准部署 | RTX 3080 | 1 | ~100 | ~50 |
| vLLM | RTX 3080 | 1 | ~300 | ~20 |
| vLLM (批处理) | RTX 3080 | 32 | ~1000 | ~30 |

### vLLM 优化建议

1. **Tensor 并行**
   - 多 GPU 时启用 Tensor 并行提升吞吐量
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model outputs/my_model_vllm \
       --tensor-parallel-size 4  # 使用 4 张 GPU
   ```

2. **批处理设置**
   - 调整批处理参数优化吞吐量
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model outputs/my_model_vllm \
       --max-batch-size 32 \
       --max-num-batched-tokens 8192
   ```

3. **显存优化**
   - 设置块大小和预分配显存
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model outputs/my_model_vllm \
       --block-size 16 \
       --swap-space 4  # 4GB 交换空间
   ```

4. **监控指标**
   - 使用 vLLM 的内置监控
   - 观察 GPU 利用率和显存使用
   - 监控请求队列长度

## 常见问题

1. **CUDA out of memory**:
   - 减小批次大小
   - 启用梯度累积
   - 使用混合精度训练
   - 清理显存缓存

2. **训练速度慢**:
   - 检查 GPU 利用率
   - 优化数据加载
   - 使用更大的批次大小
   - 考虑使用多 GPU 训练

3. **推理延迟高**:
   - 使用 TorchScript 或 ONNX
   - 启用 CUDA 图优化
   - 考虑模型量化
   - 优化批处理大小

## 性能基准

以下是在不同 GPU 上的性能基准测试结果：

| GPU 型号 | 显存 | 训练批次大小 | 训练速度 (样本/秒) | 推理延迟 (ms) |
|---------|------|------------|-----------------|-------------|
| RTX 3060 | 12GB | 32 | ~100 | ~50 |
| RTX 3080 | 16GB | 64 | ~200 | ~30 |
| A100 | 40GB | 128 | ~500 | ~10 |

注意：实际性能可能因具体配置和使用场景而异。 
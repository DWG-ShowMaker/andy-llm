import os
import torch
import argparse
from src.model.config import ModelConfig
from src.model.transformer import MiniLLM
import sentencepiece as spm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--output_path', type=str, required=True, help='量化后模型保存路径')
    parser.add_argument('--quantization_type', type=str, default='dynamic',
                      choices=['dynamic', 'static'], help='量化类型')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='分词器路径')
    return parser.parse_args()

def calibrate_model(model, tokenizer, num_samples=100):
    """使用一些样本数据校准模型"""
    print("正在校准模型...")
    
    # 生成一些随机输入进行校准
    for _ in range(num_samples):
        # 创建随机输入
        input_ids = torch.randint(
            0, tokenizer.vocab_size(),
            (1, 512),  # batch_size=1, seq_len=512
            device='cpu'
        )
        
        # 前向传播以收集统计信息
        with torch.no_grad():
            model(input_ids)

def main():
    args = parse_args()
    
    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 创建模型实例
    config = ModelConfig.from_dict(checkpoint['config'])
    model = MiniLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载分词器
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    
    # 量化模型
    print(f"正在进行{args.quantization_type}量化...")
    if args.quantization_type == 'static':
        # 对于静态量化，需要先进入训练模式
        model.train()
        # 准备量化
        quantized_model = model.quantize(args.quantization_type)
        # 校准
        calibrate_model(quantized_model, tokenizer)
        # 完成后切换回评估模式
        quantized_model.eval()
    else:
        # 动态量化直接在评估模式下进行
        model.eval()
        quantized_model = model.quantize(args.quantization_type)
    
    # 保存量化后的模型
    print(f"正在保存量化模型: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        'config': config.to_dict(),
        'model_state_dict': quantized_model.state_dict()
    }, args.output_path)
    
    # 比较模型大小
    original_size = os.path.getsize(args.model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(args.output_path) / (1024 * 1024)
    
    print(f"\n量化完成!")
    print(f"原始模型大小: {original_size:.2f}MB")
    print(f"量化后大小: {quantized_size:.2f}MB")
    print(f"压缩比: {original_size/quantized_size:.2f}x")

if __name__ == "__main__":
    main() 
import os
import torch
import argparse
import sentencepiece as spm
from src.model.config import ModelConfig
from src.model.transformer import MiniLLM
from torch.serialization import add_safe_globals

# 添加 ModelConfig 到安全全局变量列表
add_safe_globals([ModelConfig])

def get_device():
    """获取可用的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_model(model_path: str, device: torch.device):
    """加载模型和配置"""
    checkpoint = None
    try:
        # 加载检查点，使用 weights_only=False
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 确保配置是字典格式
        config_dict = checkpoint['config']
        if isinstance(config_dict, ModelConfig):
            config_dict = config_dict.to_dict()
        
        # 从字典创建配置，使用 from_dict 方法
        config = ModelConfig.from_dict(config_dict)
        
        # 创建模型并加载状态
        model = MiniLLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, config
    except Exception as e:
        if checkpoint is not None:
            print(f"加载失败的配置内容: {checkpoint.get('config', 'No config found')}")
        else:
            print(f"加载模型文件失败: {model_path}")
        raise e

def generate_text(
    model: MiniLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    **kwargs
):
    """生成文本"""
    # 启用KV缓存
    model.enable_kv_cache()
    
    # 添加重复惩罚
    repetition_penalty = kwargs.get('repetition_penalty', 1.2)
    
    # 添加长度惩罚
    length_penalty = kwargs.get('length_penalty', 1.0)
    
    # 添加采样策略
    do_sample = kwargs.get('do_sample', True)
    num_beams = kwargs.get('num_beams', 1)
    
    # 编码输入文本
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    
    # 生成文本
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=kwargs.get('max_length', 100),
        temperature=kwargs.get('temperature', 0.7),
        top_k=kwargs.get('top_k', 50),
        top_p=kwargs.get('top_p', 0.9),
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        do_sample=do_sample,
        num_beams=num_beams
    )
    
    # 解码输出
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    # 生成完成后禁用缓存
    model.disable_kv_cache()
    
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='分词器路径')
    parser.add_argument('--prompt', type=str, required=True, help='输入提示')
    parser.add_argument('--max_length', type=int, default=100, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--top_k', type=int, default=50, help='top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p采样')
    args = parser.parse_args()
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    
    # 加载模型
    try:
        model, config = load_model(args.model_path, device)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 生成文本
    try:
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print("\n生成的文本:")
        print(f"输入: {args.prompt}")
        print(f"输出: {generated_text}")
    except Exception as e:
        print(f"生成失败: {e}")

if __name__ == "__main__":
    main() 
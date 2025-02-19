import os
import json
import torch
import argparse
import shutil
from src.model.config import ModelConfig
from src.model.transformer import MiniLLM
from src.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='vLLM格式输出目录')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='分词器路径')
    return parser.parse_args()

def convert_to_vllm(model_path: str, output_dir: str, tokenizer_path: str):
    """将模型转换为vLLM格式"""
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # 加载分词器获取词表大小
    tokenizer = Tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"分词器词表大小: {vocab_size}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存模型配置
    original_config = ModelConfig.from_dict(checkpoint['config'])
    print(f"原始模型词表大小: {original_config.vocab_size}")
    
    # 创建新配置
    config = ModelConfig(
        vocab_size=vocab_size,  # 使用分词器的词表大小
        d_model=original_config.d_model,
        nhead=original_config.nhead,
        num_layers=original_config.num_layers,
        dim_feedforward=original_config.dim_feedforward,
        max_seq_length=original_config.max_seq_length,
        dropout=original_config.dropout,
        attention_dropout=original_config.attention_dropout,
        layer_norm_eps=original_config.layer_norm_eps,
        initializer_range=original_config.initializer_range,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    config_dict = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": vocab_size,
        "n_positions": config.max_seq_length,
        "n_ctx": config.max_seq_length,
        "n_embd": config.d_model,
        "n_layer": config.num_layers,
        "n_head": config.nhead,
        "n_inner": config.dim_feedforward,
        "activation_function": "gelu",
        "resid_pdrop": config.dropout,
        "embd_pdrop": config.dropout,
        "attn_pdrop": config.attention_dropout,
        "layer_norm_epsilon": config.layer_norm_eps,
        "initializer_range": config.initializer_range,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 2. 创建新模型并调整权重
    new_model = MiniLLM(config)
    old_model = MiniLLM(original_config)
    old_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 复制共享权重
    new_state_dict = {}
    with torch.no_grad():
        # 复制 token embedding
        old_embed = old_model.token_embedding.weight
        new_embed = new_model.token_embedding.weight
        min_vocab = min(old_embed.shape[0], new_embed.shape[0])
        new_embed[:min_vocab] = old_embed[:min_vocab]
        new_state_dict['token_embedding.weight'] = new_embed
        
        # 复制位置编码
        new_state_dict['pos_encoder.pe'] = old_model.pos_encoder.pe
        
        # 复制 transformer 层
        for name, param in old_model.state_dict().items():
            if 'transformer' in name:
                new_state_dict[name] = param
        
        # 复制输出层
        old_output = old_model.output_layer
        new_output = torch.nn.Linear(config.d_model, vocab_size)
        min_vocab = min(old_output.weight.shape[0], vocab_size)
        new_output.weight.data[:min_vocab] = old_output.weight.data[:min_vocab]
        new_output.bias.data[:min_vocab] = old_output.bias.data[:min_vocab]
        new_state_dict['output_layer.weight'] = new_output.weight
        new_state_dict['output_layer.bias'] = new_output.bias
    
    # 加载调整后的权重
    new_model.load_state_dict(new_state_dict)
    
    # 转换为GPT2格式的权重名称
    state_dict = {}
    for name, param in new_model.state_dict().items():
        print(f"处理权重: {name}")
        
        if 'token_embedding' in name:
            new_name = 'transformer.wte.' + name.split('.')[-1]
        elif 'pos_encoder.pe' in name:
            new_name = 'transformer.wpe.weight'
            param = param.squeeze(1)
        elif 'transformer' in name:
            parts = name.split('.')
            if 'layers' in parts:
                layer_idx = parts[parts.index('layers') + 1]
                if 'self_attn' in parts:
                    if 'in_proj_weight' in name:
                        qkv_size = param.size(0) // 3
                        q, k, v = param.split(qkv_size)
                        state_dict[f'transformer.h.{layer_idx}.attn.q_proj.weight'] = q
                        state_dict[f'transformer.h.{layer_idx}.attn.k_proj.weight'] = k
                        state_dict[f'transformer.h.{layer_idx}.attn.v_proj.weight'] = v
                        continue
                    elif 'in_proj_bias' in name:
                        qkv_size = param.size(0) // 3
                        q, k, v = param.split(qkv_size)
                        state_dict[f'transformer.h.{layer_idx}.attn.q_proj.bias'] = q
                        state_dict[f'transformer.h.{layer_idx}.attn.k_proj.bias'] = k
                        state_dict[f'transformer.h.{layer_idx}.attn.v_proj.bias'] = v
                        continue
                    elif 'out_proj' in name:
                        new_name = f'transformer.h.{layer_idx}.attn.o_proj.' + parts[-1]
                elif 'linear1' in name:
                    new_name = f'transformer.h.{layer_idx}.mlp.fc1.' + parts[-1]
                elif 'linear2' in name:
                    new_name = f'transformer.h.{layer_idx}.mlp.fc2.' + parts[-1]
                elif 'norm1' in name:
                    new_name = f'transformer.h.{layer_idx}.ln_1.' + parts[-1]
                elif 'norm2' in name:
                    new_name = f'transformer.h.{layer_idx}.ln_2.' + parts[-1]
                else:
                    print(f"警告: 未知的transformer层权重: {name}")
                    continue
            else:
                new_name = name
        elif 'output_layer' in name:
            new_name = 'lm_head.' + name.split('.')[-1]
        else:
            print(f"警告: 未知的权重: {name}")
            continue
        
        print(f"映射到: {new_name}")
        if new_name not in state_dict:
            state_dict[new_name] = param
    
    # 保存转换后的权重
    torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
    
    # 3. 复制分词器文件
    shutil.copy2(tokenizer_path, os.path.join(output_dir, 'tokenizer.model'))
    
    # 4. 创建tokenizer配置
    tokenizer_config = {
        "model_type": "gpt2",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "tokenizer_class": "PreTrainedTokenizer"
    }
    
    with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"\n转换完成! 模型文件已保存到: {output_dir}")
    print("目录结构:")
    print("  - config.json (模型配置)")
    print("  - pytorch_model.bin (模型权重)")
    print("  - tokenizer.model (分词器)")
    print("  - tokenizer_config.json (分词器配置)")

if __name__ == "__main__":
    args = parse_args()
    convert_to_vllm(args.model_path, args.output_dir, args.tokenizer_path) 
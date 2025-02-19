import argparse
import torch
from src.model.config import ModelConfig
from src.model.transformer import MiniLLM
from src.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--output_path', type=str, required=True,
                      help='输出路径')
    parser.add_argument('--quantization_type', type=str, choices=['dynamic', 'static'],
                      default='dynamic', help='量化类型')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='分词器路径')
    return parser.parse_args()

def get_model_size(state_dict):
    """计算模型大小
    
    Args:
        state_dict: 模型状态字典
        
    Returns:
        float: 模型大小（MB）
    """
    total_size = 0
    for param in state_dict.values():
        if isinstance(param, torch.Tensor):
            total_size += param.numel() * param.element_size()
        # 对于量化后的参数，可能是 _packed_params 类型
        elif hasattr(param, '_packed_params'):
            for p in param._packed_params:
                if isinstance(p, torch.Tensor):
                    total_size += p.numel() * p.element_size()
    return total_size / (1024 * 1024)  # 转换为 MB

def main():
    args = parse_args()
    
    # 设备检测和选择
    if torch.cuda.is_available():
        device = 'cuda'
        print("使用 CUDA 设备进行量化")
        quantization_backend = 'fbgemm'
    elif torch.backends.mps.is_available():
        device = 'cpu'  # MPS 不支持量化，使用 CPU
        print("检测到 MPS 设备，将使用 CPU (qnnpack) 进行量化")
        quantization_backend = 'qnnpack'
    else:
        device = 'cpu'
        print("使用 CPU 进行量化")
        # 检测是否为 ARM 架构
        import platform
        if platform.machine() in ['arm64', 'aarch64']:
            quantization_backend = 'qnnpack'
            print("检测到 ARM 架构，使用 qnnpack 后端")
        else:
            quantization_backend = 'fbgemm'
            print("检测到 x86 架构，使用 fbgemm 后端")
    
    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 加载配置
    config = ModelConfig.from_dict(checkpoint['config'])
    
    # 创建模型
    model = MiniLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cpu')  # 量化需要在 CPU 上进行
    model.eval()
    
    try:
        # 设置量化后端
        torch.backends.quantized.engine = quantization_backend
        
        # 进行量化
        print(f"正在进行{args.quantization_type}量化...")
        if args.quantization_type == 'dynamic':
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  # 只量化线性层
                dtype=torch.qint8
            )
        else:
            # 静态量化
            # 为 Embedding 层设置特殊的量化配置
            float_qparams_config = torch.quantization.float_qparams_weight_only_qconfig
            default_qconfig = torch.quantization.get_default_qconfig(quantization_backend)
            
            # 为不同的模块设置不同的量化配置
            qconfig_dict = {
                'token_embedding': float_qparams_config,
                'pos_encoder': None,  # 不量化位置编码
                'transformer': default_qconfig,
                'output_layer': default_qconfig
            }
            
            # 准备量化
            model.qconfig = default_qconfig
            
            # 为特定模块设置配置
            model.token_embedding.qconfig = float_qparams_config
            model.pos_encoder.qconfig = None
            model.transformer.qconfig = default_qconfig
            model.output_layer.qconfig = default_qconfig
            
            # 准备量化
            prepared_model = torch.quantization.prepare(model)
            
            # 进行校准
            print("正在进行校准...")
            with torch.no_grad():
                for i in range(10):
                    print(f"校准进度: {i+1}/10", end='\r')
                    # 生成随机输入
                    dummy_input = torch.randint(0, config.vocab_size, (1, 32))
                    # 创建简单的掩码
                    dummy_attention_mask = torch.ones(1, 32, dtype=torch.bool)
                    
                    # 前向传播
                    prepared_model(
                        input_ids=dummy_input,
                        attention_mask=dummy_attention_mask
                    )
            print("\n校准完成!")
            
            # 转换为量化模型
            quantized_model = torch.quantization.convert(prepared_model)
        
        # 保存量化后的模型
        print(f"正在保存量化后的模型到: {args.output_path}")
        torch.save({
            'config': config.to_dict(),
            'model_state_dict': quantized_model.state_dict(),
            'quantization_backend': quantization_backend
        }, args.output_path)
        
        print("量化完成!")
        
        # 打印模型大小比较
        print(f"\n模型大小比较:")
        print(f"原始模型: {get_model_size(model.state_dict()):.2f} MB")
        print(f"量化后: {get_model_size(quantized_model.state_dict()):.2f} MB")
        
        # 计算压缩率
        original_size = get_model_size(model.state_dict())
        quantized_size = get_model_size(quantized_model.state_dict())
        compression_ratio = (1 - quantized_size/original_size) * 100
        print(f"压缩率: {compression_ratio:.1f}%")
        
    except Exception as e:
        print(f"\n量化过程中出现错误: {str(e)}")
        print("\n建议:")
        print("1. 确保已安装最新版本的 PyTorch")
        print(f"2. 当前使用的量化后端: {quantization_backend}")
        print("3. 如果使用 ARM 设备 (如 M1/M2):")
        print("   - 动态量化通常更稳定")
        print("   - 确保使用 CPU 进行量化")
        print("4. 如果使用 CUDA:")
        print("   - 确保 CUDA 版本与 PyTorch 匹配")
        print("   - 尝试在 CPU 上进行量化后再移回 GPU")
        raise e

if __name__ == "__main__":
    main() 
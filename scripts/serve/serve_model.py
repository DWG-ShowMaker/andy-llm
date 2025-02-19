import argparse
import torch
import sys
import os
from torch.serialization import add_safe_globals

# 添加项目根目录到 Python 路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from src.model.config import ModelConfig
from src.model.transformer import MiniLLM
from src.tokenizer import Tokenizer

app = FastAPI()

# 添加 ModelConfig 到安全全局变量列表
add_safe_globals([ModelConfig])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='分词器路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='设备')
    parser.add_argument('--port', type=int, default=8000,
                      help='端口号')
    return parser.parse_args()

def load_quantized_model(model_path: str, device: str):
    """加载量化模型
    
    Args:
        model_path: 模型路径
        device: 设备类型
        
    Returns:
        model: 加载的模型
        config: 模型配置
        device: 实际使用的设备
    """
    print("正在加载量化模型...")
    
    # 加载检查点
    try:
        # 首先尝试使用 weights_only=True
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"使用 weights_only=True 加载失败，尝试完整加载...")
        # 如果失败，使用完整加载
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 获取配置
    config = ModelConfig.from_dict(checkpoint['config'])
    
    # 创建模型
    model = MiniLLM(config)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 确定设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，回退到 CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("警告: MPS 不可用，回退到 CPU")
        device = 'cpu'
    
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()
    
    print(f"模型将在 {device.upper()} 上运行")
    return model, config, device

# 加载模型和分词器
args = parse_args()

# 加载模型
model, config, device = load_quantized_model(args.model, args.device)
args.device = device  # 更新设备设置

# 加载分词器
tokenizer = Tokenizer.load(args.tokenizer_path)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data['prompt']
    
    # 编码输入
    inputs = tokenizer.encode(prompt)
    input_ids = torch.tensor([inputs], device=args.device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            repetition_penalty=data.get('repetition_penalty', 1.1)
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0].tolist())
    
    return JSONResponse({
        "text": generated_text
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"模型已加载到设备: {args.device}")
    print(f"服务运行在: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port) 
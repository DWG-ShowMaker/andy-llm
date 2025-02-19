import argparse
import torch
import sys
import os

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

def load_quantized_model(model_path, device='cpu'):
    """加载量化模型"""
    print("正在加载量化模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = ModelConfig.from_dict(checkpoint['config'])
    
    # 创建模型
    model = MiniLLM(config)
    
    if 'quantization_backend' in checkpoint:
        print(f"检测到量化模型 (backend: {checkpoint['quantization_backend']})")
        backend = checkpoint['quantization_backend']
        torch.backends.quantized.engine = backend
        
        # 获取原始状态字典
        state_dict = checkpoint['model_state_dict']
        
        # 过滤掉量化相关的参数
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not any(x in key for x in ['scale', 'zero_point', '_packed_params']):
                filtered_state_dict[key] = value
        
        # 加载非量化参数
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # 设置为 CPU 设备
        device = 'cpu'
    else:
        # 非量化模型直接加载
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
    model.eval()
    return model, config, device

# 加载模型和分词器
args = parse_args()

# 加载模型
model, config, device = load_quantized_model(args.model, args.device)
args.device = device  # 更新设备设置

if device == 'cpu':
    print("模型将在 CPU 上运行")
else:
    print(f"模型将在 {device} 上运行")

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
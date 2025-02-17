import argparse
from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import torch

app = FastAPI()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8000,
                      help='服务器端口')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='运行设备 (cuda/cpu)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                      help='模型并行大小')
    return parser.parse_args()

# 初始化模型
args = parse_args()
llm = LLM(
    model=args.model,
    device=args.device,
    tensor_parallel_size=args.tensor_parallel_size,
    trust_remote_code=True,  # 信任远程代码
    dtype='float16' if args.device == 'cuda' else 'float32'  # 根据设备选择精度
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    
    # 获取生成参数
    sampling_params = SamplingParams(
        temperature=data.get('temperature', 0.7),
        top_p=data.get('top_p', 0.9),
        max_tokens=data.get('max_tokens', 100),
        repetition_penalty=data.get('repetition_penalty', 1.2)
    )
    
    # 生成文本
    outputs = llm.generate(
        prompts=[data['prompt']],
        sampling_params=sampling_params
    )
    
    # 返回生成结果
    return JSONResponse({
        "text": outputs[0].outputs[0].text,
        "finish_reason": outputs[0].outputs[0].finish_reason
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"使用设备: {args.device}")
    print(f"模型路径: {args.model}")
    print(f"服务地址: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port) 
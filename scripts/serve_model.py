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
from transformers import AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor, TopPLogitsWarper, TemperatureLogitsWarper
from src.tokenizer import Tokenizer  # 现在应该可以正确导入了
import torch.nn.functional as F

app = FastAPI()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='分词器路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8000,
                      help='服务器端口')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                      help='运行设备')
    return parser.parse_args()

# 初始化模型和分词器
args = parse_args()
model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
tokenizer = Tokenizer.load(args.tokenizer_path)  # 加载我们自己的分词器
model = model.to(args.device)
model.eval()

class CustomMinLengthLogitsProcessor:
    """自定义最小长度处理器，避免 MPS 设备的兼容性问题"""
    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id
    
    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            # 创建一个与 scores 相同设备的 mask
            scores = scores.clone()
            scores[:, self.eos_token_id] = float('-inf')
        return scores

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    
    # 编码输入
    input_ids = tokenizer.encode(data['prompt'])
    input_ids = torch.tensor([input_ids]).to(args.device)
    
    # 设置生成参数
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    repetition_penalty = data.get('repetition_penalty', 1.2)
    
    # 创建 logits processors
    processors = LogitsProcessorList([
        CustomMinLengthLogitsProcessor(5, tokenizer.eos_token_id),
        TemperatureLogitsWarper(temperature),
        TopPLogitsWarper(top_p)
    ])
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=processors
        )
    
    # 解码输出
    generated_ids = outputs.sequences[0][len(input_ids[0]):]
    response_text = tokenizer.decode(generated_ids.tolist())
    
    # 返回生成结果
    return JSONResponse({
        "text": response_text,
        "finish_reason": "stop" if generated_ids[-1] == tokenizer.eos_token_id else "length"
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"使用设备: {args.device}")
    print(f"模型路径: {args.model}")
    print(f"分词器路径: {args.tokenizer_path}")
    print(f"服务地址: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port) 
import requests
import argparse
from typing import Dict, Any

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                      help='服务器URL')
    parser.add_argument('--prompt', type=str, default='请介绍一下你自己',
                      help='输入提示')
    return parser.parse_args()

def generate_text(url: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """调用API生成文本
    
    参数:
        url: API地址
        prompt: 输入提示
        **kwargs: 其他生成参数
    """
    # 设置默认参数
    params = {
        'prompt': prompt,
        'max_tokens': kwargs.get('max_tokens', 100),
        'temperature': kwargs.get('temperature', 0.7),
        'top_p': kwargs.get('top_p', 0.9),
        'repetition_penalty': kwargs.get('repetition_penalty', 1.2)
    }
    
    try:
        # 发送请求
        response = requests.post(f"{url}/generate", json=params)
        response.raise_for_status()  # 检查响应状态
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def test_health(url: str) -> bool:
    """测试服务健康状态"""
    try:
        response = requests.get(f"{url}/health")
        response.raise_for_status()
        return response.json()['status'] == 'healthy'
    except requests.exceptions.RequestException:
        return False

def main():
    args = parse_args()
    
    # 检查服务是否在线
    print("检查服务状态...")
    if not test_health(args.url):
        print("服务未启动或不可访问!")
        return
    
    print("服务正常运行!")
    
    # 测试文本生成
    print(f"\n输入提示: {args.prompt}")
    print("正在生成...")
    
    result = generate_text(
        args.url,
        args.prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    if result:
        print("\n生成结果:")
        print(f"文本: {result['text']}")
        print(f"结束原因: {result['finish_reason']}")
    else:
        print("生成失败!")

if __name__ == "__main__":
    main() 
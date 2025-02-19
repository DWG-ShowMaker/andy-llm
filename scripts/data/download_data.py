import os
import argparse
from datasets import load_dataset
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset
from src.data.dataset_config import DATASET_CONFIGS
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['wiki', 'dialogue'],
                      help='要下载的数据集列表')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                      help='数据保存目录')
    return parser.parse_args()

def download_from_modelscope(config, output_dir):
    """从 ModelScope 下载数据集"""
    print(f"正在下载数据集...")
    
    # 使用 MsDataset 下载数据集
    dataset = MsDataset.load(
        config.path,
        subset_name=config.subset,
        split='train',
        cache_dir=output_dir,
        use_streaming=False
    )
    
    # 保存为本地文件
    splits = ['train', 'test']
    for split in splits:
        try:
            split_data = MsDataset.load(
                config.path,
                subset_name=config.subset,
                split=split,
                cache_dir=output_dir,
                use_streaming=False
            )
            
            # 转换为列表并保存
            output_file = os.path.join(output_dir, f"{split}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    # 处理对话格式
                    dialogue = {
                        "system": item["system"],
                        "conversation": item["conversation"]  # 已经是 JSON 字符串
                    }
                    f.write(f"{json.dumps(dialogue, ensure_ascii=False)}\n")
            
            print(f"- 已保存 {split} 数据到: {output_file}")
        except Exception as e:
            print(f"警告: 无法加载 {split} 数据: {str(e)}")
    
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = DATASET_CONFIGS["muice"]  # 只使用 muice 数据集
    
    try:
        dataset = download_from_modelscope(config, args.output_dir)
        print(f"成功下载数据集")
    except Exception as e:
        print(f"下载数据集时出错: {str(e)}")

if __name__ == "__main__":
    main() 
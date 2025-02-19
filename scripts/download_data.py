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
    print(f"正在下载 {config.name} 数据集...")
    
    # 使用 MsDataset 下载数据集
    dataset = MsDataset.load(
        config.path,
        split='train',  # 默认使用训练集
        cache_dir=output_dir,
        use_streaming=False  # 完整下载到本地
    )
    
    # 保存为本地文件
    splits = ['train', 'validation', 'test']
    for split in splits:
        try:
            split_data = MsDataset.load(
                config.path,
                split=split,
                cache_dir=output_dir,
                use_streaming=False
            )
            
            # 转换为列表并保存
            output_file = os.path.join(output_dir, f"{config.name}_{split}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    if config.text_column in item:
                        f.write(f"{json.dumps({'text': item[config.text_column]}, ensure_ascii=False)}\n")
            
            print(f"- 已保存 {split} 数据到: {output_file}")
        except Exception as e:
            print(f"警告: 无法加载 {split} 数据: {str(e)}")
    
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"警告: 未知的数据集 {dataset_name}")
            continue
            
        config = DATASET_CONFIGS[dataset_name]
        
        try:
            if config.source == 'modelscope':
                dataset = download_from_modelscope(config, args.output_dir)
                print(f"成功下载 {dataset_name} 数据集")
            else:
                print(f"警告: 暂不支持的数据源 {config.source}")
                continue
                
        except Exception as e:
            print(f"下载 {dataset_name} 数据集时出错: {str(e)}")

if __name__ == "__main__":
    main() 
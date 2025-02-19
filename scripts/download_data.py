import os
import argparse
from datasets import load_dataset
from modelscope.hub.api import HubApi
from src.data.dataset_config import DATASET_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['wiki', 'belle'],
                      help='要下载的数据集列表')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                      help='数据保存目录')
    return parser.parse_args()

def download_from_huggingface(config, output_dir):
    """从 HuggingFace 下载数据集"""
    print(f"正在下载 {config.name} 数据集...")
    
    dataset = load_dataset(
        config.path,
        config.subset,
        cache_dir=output_dir
    )
    
    # 保存为本地文件
    for split in dataset.keys():
        output_file = os.path.join(output_dir, f"{config.name}_{split}.jsonl")
        dataset[split].to_json(output_file)
    
    return dataset

def download_from_modelscope(config, output_dir):
    """从 ModelScope 下载数据集"""
    print(f"正在下载 {config.name} 数据集...")
    
    hub_api = HubApi()
    dataset = hub_api.load_dataset(
        config.path,
        cache_dir=output_dir
    )
    
    # 保存为本地文件
    for split in dataset.keys():
        output_file = os.path.join(output_dir, f"{config.name}_{split}.jsonl")
        dataset[split].to_json(output_file)
    
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
            if config.source == 'huggingface':
                dataset = download_from_huggingface(config, args.output_dir)
            elif config.source == 'modelscope':
                dataset = download_from_modelscope(config, args.output_dir)
            else:
                print(f"警告: 未知的数据源 {config.source}")
                continue
                
            print(f"成功下载 {dataset_name} 数据集:")
            for split, ds in dataset.items():
                print(f"- {split}: {len(ds)} 条数据")
                
        except Exception as e:
            print(f"下载 {dataset_name} 数据集时出错: {str(e)}")

if __name__ == "__main__":
    main() 
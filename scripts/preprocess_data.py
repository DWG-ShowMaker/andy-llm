import os
import argparse
from datasets import load_dataset, concatenate_datasets
from src.data.dataset_config import DATASET_CONFIGS
from src.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['wiki', 'belle'],
                      help='要处理的数据集列表')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                      help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='处理后数据保存目录')
    parser.add_argument('--vocab_size', type=int, default=30000,
                      help='词表大小')
    return parser.parse_args()

def filter_and_clean(dataset, config):
    """数据过滤和清洗"""
    # 长度过滤
    dataset = dataset.filter(lambda x: len(x[config.text_column]) >= config.min_length)
    
    # 质量过滤
    if "质量过滤" in config.filter_rules:
        dataset = dataset.filter(lambda x: x[config.text_column].count('。') >= 2)
    
    # 格式化处理
    if "对话格式化" in config.filter_rules:
        dataset = dataset.map(format_dialogue)
    elif "学术格式化" in config.filter_rules:
        dataset = dataset.map(format_academic)
    elif "新闻格式化" in config.filter_rules:
        dataset = dataset.map(format_news)
    elif "章节合并" in config.filter_rules:
        dataset = dataset.map(merge_chapters)
    
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载并处理所有数据集
    processed_datasets = []
    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]
        print(f"\n处理 {dataset_name} 数据集...")
        
        # 加载数据
        dataset = load_dataset(
            'json',
            data_files={
                'train': os.path.join(args.input_dir, f"{dataset_name}_train.jsonl"),
                'validation': os.path.join(args.input_dir, f"{dataset_name}_validation.jsonl"),
                'test': os.path.join(args.input_dir, f"{dataset_name}_test.jsonl")
            }
        )
        
        # 过滤和清洗
        dataset = filter_and_clean(dataset, config)
        processed_datasets.append(dataset)
        
        print(f"处理后数据量:")
        for split, ds in dataset.items():
            print(f"- {split}: {len(ds)} 条")
    
    # 合并所有数据集
    final_dataset = concatenate_datasets(processed_datasets)
    
    # 训练分词器
    tokenizer = Tokenizer.train(
        texts=final_dataset['train'][config.text_column],
        vocab_size=args.vocab_size,
        model_path=os.path.join(args.output_dir, 'tokenizer.model')
    )
    
    # 保存处理后的数据
    for split in ['train', 'validation', 'test']:
        output_file = os.path.join(args.output_dir, f"{split}.jsonl")
        final_dataset[split].to_json(output_file)
    
    print("\n数据处理完成!")
    print(f"- 分词器保存至: {os.path.join(args.output_dir, 'tokenizer.model')}")
    print(f"- 处理后数据保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 
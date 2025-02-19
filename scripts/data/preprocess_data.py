import os
import argparse
from datasets import load_dataset
from src.data.dataset_config import DATASET_CONFIGS
from src.tokenizer import Tokenizer
from src.data.format import format_dialogue  # 导入格式化函数
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/raw',
                      help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='处理后数据保存目录')
    parser.add_argument('--vocab_size', type=int, default=32000,
                      help='词表大小')
    return parser.parse_args()

def filter_and_clean(dataset, config):
    """数据过滤和清洗"""
    # 格式化处理
    dataset = dataset.map(format_dialogue)
    
    # 长度过滤
    dataset = dataset.filter(lambda x: len(x['text']) >= config.min_length)
    
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n处理数据集...")
    
    # 加载数据
    dataset = load_dataset(
        'json',
        data_files={
            'train': os.path.join(args.input_dir, "train.jsonl"),
            'test': os.path.join(args.input_dir, "test.jsonl")
        }
    )
    
    # 过滤和清洗
    config = DATASET_CONFIGS["muice"]
    dataset = filter_and_clean(dataset, config)
    
    print(f"处理后数据量:")
    for split, ds in dataset.items():
        print(f"- {split}: {len(ds)} 条")
    
    # 训练分词器
    tokenizer = Tokenizer.train(
        texts=[example['text'] for example in dataset['train']],  # 使用格式化后的文本
        vocab_size=args.vocab_size,
        model_path=os.path.join(args.output_dir, 'tokenizer.model')
    )
    
    # 保存处理后的数据
    for split in ['train', 'test']:
        output_file = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset[split]:
                # 确保 conversation 是字符串
                if isinstance(item['conversation'], list):
                    item['conversation'] = json.dumps(item['conversation'])
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n数据处理完成!")
    print(f"- 分词器保存至: {os.path.join(args.output_dir, 'tokenizer.model')}")
    print(f"- 处理后数据保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 
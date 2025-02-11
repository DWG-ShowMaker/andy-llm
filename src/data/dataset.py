import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from typing import List, Dict, Any
import random

class TextDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: spm.SentencePieceProcessor,
        max_length: int = 512,
        min_length: int = 10
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        # 读取文本数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = [line.strip() for line in f if len(line.strip()) >= min_length]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        
        # 编码文本
        input_ids = self.tokenizer.encode(text)
        
        # 截断或填充到指定长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            padding_length = 0
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length  # 0 是PAD token的ID
        
        # 创建attention mask (1表示真实token，0表示padding)
        attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        
        # 转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # 创建标签，移位一个位置（预测下一个token）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 将padding位置的标签设为-100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 
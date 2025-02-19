import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from typing import List, Dict, Any
import random
import json

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
        
        # 读取jsonl数据
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if len(item['text'].strip()) >= min_length:
                    self.examples.append(item)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        text = item['text']
        
        # 编码文本
        input_ids = self.tokenizer.encode(text)
        
        # 截断或填充到指定长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            padding_length = 0
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length
        
        # 创建attention mask
        attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        
        # 转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # 创建标签
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'metadata': item.get('metadata', {})  # 保留元信息
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

class ChatDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        
    def __getitem__(self, idx):
        dialogue = self.data[idx]
        turns = dialogue.split('\n')
        
        # 构建上下文
        context = []
        for i in range(0, len(turns)-1, 2):
            human = turns[i]
            assistant = turns[i+1] if i+1 < len(turns) else ""
            
            # 编码当前回合
            inputs = self.tokenizer.encode(
                "\n".join(context + [human]),
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True
            )
            
            # 编码目标回答
            labels = self.tokenizer.encode(
                assistant,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True
            )
            
            context.extend([human, assistant])
            
            if len(inputs) + len(labels) <= self.max_length:
                attention_mask = [1] * len(inputs)
                return {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                
    def __len__(self):
        return len(self.data) 
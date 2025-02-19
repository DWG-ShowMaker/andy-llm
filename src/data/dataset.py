import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from typing import List, Dict, Any
import random
import json

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Any, max_length: int = 512):
        """初始化数据集
        
        参数:
            file_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 处理对话格式
                system = item['system']
                conversation = json.loads(item['conversation'])
                
                # 构建对话文本
                text = f"<system>{system}</system>\n"
                for msg in conversation:
                    if 'human' in msg:
                        text += f"<human>{msg['human']}</human>\n"
                    if 'assistant' in msg:
                        text += f"<assistant>{msg['assistant']}</assistant>\n"
                
                # 编码文本
                encoded = self.tokenizer.encode_as_ids(text)
                if len(encoded) > self.max_length:
                    encoded = encoded[:self.max_length]
                
                self.data.append({
                    'input_ids': encoded,
                    'attention_mask': [1] * len(encoded)
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
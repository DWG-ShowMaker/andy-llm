import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from typing import List, Dict, Any, Optional
import random
import json

class TextDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        system_prompt: Optional[str] = None
    ):
        """初始化数据集
        
        Args:
            data: 数据列表，每项包含 system 和 conversation 字段
            tokenizer: 分词器实例
            max_length: 最大序列长度
            system_prompt: 系统提示语，如果为 None 则使用数据中的 system
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取系统提示语
        system = self.system_prompt or item['system']
        
        # 处理对话数据
        conversation = item['conversation']
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
            
        # 构建对话文本
        dialogue = f"<system>{system}</system>\n"
        for turn in conversation:
            if 'human' in turn:
                dialogue += f"<human>{turn['human']}</human>\n"
            if 'assistant' in turn:
                dialogue += f"<assistant>{turn['assistant']}</assistant>\n"
                
        # 编码文本
        input_ids = self.tokenizer.encode(dialogue)
        
        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 转换为张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids  # 用于自回归训练
        }

def collate_fn(batch):
    """处理批次数据
    
    Args:
        batch: 批次数据列表
        
    Returns:
        处理后的批次数据字典
    """
    # 找出最大长度
    max_len = max(len(x['input_ids']) for x in batch)
    
    # 初始化张量
    input_ids = []
    attention_mask = []
    labels = []
    
    # 填充每个样本
    for item in batch:
        # 获取当前长度
        curr_len = len(item['input_ids'])
        padding_len = max_len - curr_len
        
        # 填充 input_ids
        input_ids.append(
            torch.cat([
                item['input_ids'],
                torch.zeros(padding_len, dtype=torch.long)
            ])
        )
        
        # 填充 attention_mask
        attention_mask.append(
            torch.cat([
                item['attention_mask'],
                torch.zeros(padding_len, dtype=torch.long)
            ])
        )
        
        # 填充 labels
        labels.append(
            torch.cat([
                item['labels'],
                torch.ones(padding_len, dtype=torch.long) * -100  # 忽略填充位置的损失
            ])
        )
    
    # 堆叠为批次
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }

def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # 添加自定义的 collate 函数
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
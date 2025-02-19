import os
import argparse
import torch
import sentencepiece as spm
from src.model.config import ModelConfig, TrainingConfig
from src.model.transformer import MiniLLM
from src.data.dataset import TextDataset, create_dataloader, ChatDataset
from src.training.trainer import Trainer
import wandb
from torch.utils.data import DataLoader
import json
from src.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument('--train_file', type=str, default='data/processed/train.jsonl',
                      help='训练数据文件路径')
    parser.add_argument('--val_file', type=str, default='data/processed/test.jsonl',
                      help='验证数据文件路径')
    parser.add_argument('--tokenizer_path', type=str, default='data/processed/tokenizer.model',
                      help='分词器路径')
    
    # 模型相关参数
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--max_length', type=int, default=512)
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=8,
                      help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='训练轮数')
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='训练设备')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='模型保存目录')
    
    # 增加模型配置选项
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dropout', type=float, default=0.1)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    
    # 添加新的训练优化参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='梯度累积步数')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                      help='学习率调度策略')
    parser.add_argument('--patience', type=int, default=3,
                      help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                      help='早停最小改善阈值')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载分词器
    tokenizer = spm.SentencePieceProcessor()
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"找不到分词器文件: {args.tokenizer_path}")
    tokenizer.load(args.tokenizer_path)
    
    # 加载数据
    print("加载数据...")
    train_data = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
            
    val_data = []
    with open(args.val_file, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # 创建数据集
    train_dataset = TextDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = TextDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # 创建数据加载器
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 加载模型配置
    model_config = ModelConfig.from_pretrained(args.model_size)
    model_config.vocab_size = tokenizer.get_piece_size()
    
    # 创建模型
    model = MiniLLM(model_config)
    model = model.to(device)
    
    # 创建训练配置
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scheduler=args.scheduler,
        device=args.device,
        fp16=args.fp16,
        output_dir=args.output_dir,
        save_every=args.save_every,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config
    )
    
    # 初始化 wandb
    wandb.init(project="andy-llm", config=args)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 
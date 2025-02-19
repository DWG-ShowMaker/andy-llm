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

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument('--train_file', type=str, default='data/raw/muice_train.jsonl',
                      help='训练数据文件路径')
    parser.add_argument('--val_file', type=str, default='data/raw/muice_test.jsonl',
                      help='验证数据文件路径')
    parser.add_argument('--tokenizer_path', type=str, default='data/processed/tokenizer.model')
    
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
    # 1. 解析参数
    args = parse_args()
    
    # 2. 检查数据文件是否存在
    for file_path in [args.train_file, args.val_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据文件: {file_path}")
    
    # 3. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 加载分词器
    tokenizer = spm.SentencePieceProcessor()
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"找不到分词器文件: {args.tokenizer_path}")
    tokenizer.load(args.tokenizer_path)
    
    # 5. 创建数据集和数据加载器
    train_dataset = TextDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = TextDataset(
        file_path=args.val_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 6. 计算总训练步数
    args.num_training_steps = (
        len(train_dataloader) 
        * args.num_epochs 
        // args.gradient_accumulation_steps
    )
    
    # 7. 创建模型配置
    model_config = ModelConfig(
        vocab_size=tokenizer.get_piece_size(),  # 使用分词器的词表大小
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_seq_length=args.max_length,
        dropout=0.1
    )
    
    # 8. 创建训练配置
    training_config = TrainingConfig(
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        fp16=args.device == 'cuda',  # 在 GPU 上启用混合精度训练
        max_grad_norm=1.0,
        weight_decay=0.01,
        warmup_steps=100,
        scheduler='cosine',
        num_training_steps=len(train_dataloader) * args.num_epochs
    )
    
    # 9. 创建模型
    model = MiniLLM(model_config)
    
    # 10. 创建训练器并开始训练
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config
    )
    
    # 初始化wandb
    wandb.init(project="andy-llm", config=args)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 
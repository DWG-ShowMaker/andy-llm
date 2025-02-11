import os
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Any,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 初始化最佳损失
        self.best_loss = float('inf')
        
        # 初始化wandb
        wandb.init(project="miniLLM", config=vars(config))
    
    def _create_scheduler(self) -> LambdaLR:
        """创建带有预热的学习率调度器"""
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = self.config.warmup_steps
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """训练模型"""
        for epoch in range(self.config.num_epochs):
            # 训练一个epoch
            train_loss = self._train_epoch()
            
            # 验证
            val_loss = self._validate()
            
            # 记录指标
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {self.best_loss:.4f}")
    
    def _train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc='Training')
        for batch in progress_bar:
            # 将数据移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            _, loss = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # 优化器步进
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新进度条
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_dataloader)
    
    @torch.no_grad()
    def _validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_dataloader, desc='Validating'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, loss = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        # 确保配置是字典格式
        config_dict = self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else vars(self.model.config)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': config_dict,
        }
        
        save_path = os.path.join(
            self.config.output_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        # 保存到临时文件，然后重命名，避免保存中断导致文件损坏
        temp_path = save_path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, save_path)
        
        # 如果是最佳模型，创建一个副本
        if loss == self.best_loss:
            best_path = os.path.join(self.config.output_dir, 'best_model.pt')
            os.replace(save_path, best_path)
            print(f"保存最佳模型，loss: {loss:.4f}") 
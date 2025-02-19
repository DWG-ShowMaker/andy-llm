import os
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
from .early_stopping import EarlyStopping
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import logging
import torch.nn.functional as F

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
        
        # 将模型移动到指定设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 多GPU支持
        if config.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # 配置优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 配置混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 添加学习率调度器选项
        if config.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.num_training_steps
            )
        elif config.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.num_training_steps
            )
        else:
            self.scheduler = self._create_scheduler()
        
        # 添加梯度累积
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
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
        # 添加更多评估指标
        metrics = {
            'loss': [],
            'perplexity': [],
            'accuracy': [],
            'bleu': [],
            'rouge': []
        }
        
        # 添加早停
        early_stopper = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练和评估
            train_metrics = self._train_epoch()
            val_metrics = self._validate()
            
            # 打印详细的训练信息
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Perplexity: {train_metrics['perplexity']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Perplexity: {val_metrics['perplexity']:.4f}")
            
            # 记录到wandb
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/perplexity': train_metrics['perplexity'],
                'val/loss': val_metrics['loss'],
                'val/perplexity': val_metrics['perplexity'],
                'val/bleu': val_metrics['bleu'],
                'val/rouge': val_metrics['rouge'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
            
            # 检查是否需要保存模型
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self._save_checkpoint(epoch + 1, val_metrics['loss'])
            
            # 早停检查
            if early_stopper(val_metrics['loss']):
                print("\nEarly stopping triggered!")
                break
        
        print("\nTraining completed!")
        wandb.finish()
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        # 添加进度条
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training",
            leave=True,
            dynamic_ncols=True  # 自动调整宽度
        )
        
        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 使用自动混合精度
            if self.config.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)), 
                        labels.view(-1),
                        ignore_index=-100
                    )
                
                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1),
                    ignore_index=-100
                )
                
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # 更新损失和进度条
            total_loss += loss.item()
            current_loss = total_loss / (step + 1)
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        # 计算平均损失和其他指标
        avg_loss = total_loss / len(self.train_dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity.item(),
            'accuracy': 0.0,  # 暂时不计算
            'bleu': 0.0,     # 暂时不计算
            'rouge': 0.0     # 暂时不计算
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        # 添加验证进度条
        progress_bar = tqdm(
            self.val_dataloader,
            desc='Validating',
            leave=True,
            dynamic_ncols=True
        )
        
        for batch in progress_bar:
            # 同样处理验证集数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                labels.view(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            
            # 更新进度条
            current_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # 计算平均损失和其他指标
        avg_loss = total_loss / len(self.val_dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity.item(),
            'bleu': 0.0,     # 暂时不计算
            'rouge': 0.0     # 暂时不计算
        }
    
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
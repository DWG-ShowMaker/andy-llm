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
import math
from contextlib import nullcontext
import json

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
        
        # 更新总训练步数
        self.config.update_training_steps(train_dataloader)
        
        # 将模型移动到指定设备
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # 多GPU支持
        if config.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # 配置优化器
        self.optimizer = self._create_optimizer()
        
        # 配置学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 配置混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 添加梯度累积
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
        
        # 初始化最佳损失
        self.best_loss = float('inf')
        
        # 初始化 wandb
        if not wandb.run:
            wandb.init(project="andy-llm", config=vars(config))
    
    def _create_optimizer(self):
        """创建优化器"""
        # 设置优化器参数组
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # 创建 AdamW 优化器
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.num_training_steps
            )
        else:  # default to linear
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.num_training_steps
            )
    
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
    
    def _train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # 创建进度条
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移动到正确的设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            with torch.cuda.amp.autocast() if self.config.fp16 else nullcontext():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # 如果使用梯度累积，需要缩放损失
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.config.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # 更新参数
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.config.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
            
            # 计算指标
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'perplexity': math.exp(total_loss / total_samples)
            })
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        metrics = {
            'loss': avg_loss,
            'perplexity': math.exp(avg_loss)
        }
        
        return metrics
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        # 添加验证进度条
        progress_bar = tqdm(
            self.val_dataloader,
            desc='Validating',
            leave=True,
            dynamic_ncols=True
        )
        
        for batch in progress_bar:
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 获取损失
            loss = outputs['loss']
            
            # 计算指标
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'perplexity': math.exp(total_loss / total_samples)
            })
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss,
            'perplexity': math.exp(avg_loss),
            'bleu': 0.0,     # 暂时不计算
            'rouge': 0.0     # 暂时不计算
        }
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """保存检查点和最佳模型
        
        Args:
            epoch: 当前轮数
            loss: 当前损失
        """
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'checkpoints'), exist_ok=True)
        
        # 获取模型配置
        if hasattr(self.model, 'config'):
            model_config = self.model.config
        else:
            # 如果模型没有直接的配置属性，尝试从训练配置中提取模型相关参数
            model_config = {
                'model_size': self.config.model_size,
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'nhead': self.config.nhead,
                'num_layers': self.config.num_layers,
                'dim_feedforward': self.config.dim_feedforward,
                'dropout': self.config.dropout,
                'attention_dropout': self.config.attention_dropout,
                'hidden_dropout': self.config.hidden_dropout,
                'layer_norm_eps': self.config.layer_norm_eps,
                'max_seq_length': self.config.max_length,
                'pad_token_id': self.config.pad_token_id,
                'bos_token_id': self.config.bos_token_id,
                'eos_token_id': self.config.eos_token_id,
            }
        
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': model_config,  # 只保存模型相关的配置
            'best_loss': self.best_loss,
            'training_config': {  # 单独保存训练配置
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'warmup_steps': self.config.warmup_steps,
                'weight_decay': self.config.weight_decay,
                'max_grad_norm': self.config.max_grad_norm,
                'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
                'scheduler': self.config.scheduler,
                'device': self.config.device,
                'fp16': self.config.fp16
            }
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(
            self.config.output_dir,
            'checkpoints',
            f'checkpoint_epoch_{epoch}_loss_{loss:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved checkpoint: {checkpoint_path}")
        
        # 如果是最佳模型，保存到 best_model.pt
        if loss <= self.best_loss:
            best_model_path = os.path.join(self.config.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved: {best_model_path} (loss: {loss:.4f})")
            
            # 更新最佳损失
            self.best_loss = loss
            
            # 保存模型配置
            config_path = os.path.join(self.config.output_dir, 'model_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False) 
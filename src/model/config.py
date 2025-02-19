from dataclasses import dataclass
from typing import Optional
import torch
import os
import json

class ModelConfig:
    """模型配置类"""
    def __init__(
        self,
        model_size: str = "tiny",
        vocab_size: int = 10000,
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ):
        """初始化模型配置"""
        self.model_size = model_size
        
        # 根据模型大小设置参数
        if model_size == "small":
            self.vocab_size = 30000
            self.d_model = 256
            self.nhead = 4
            self.num_layers = 4
            self.dim_feedforward = 1024
        elif model_size == "medium":
            self.vocab_size = 50000
            self.d_model = 512
            self.nhead = 8
            self.num_layers = 6
            self.dim_feedforward = 2048
        else:  # tiny
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.dim_feedforward = dim_feedforward
        
        # 设置其他参数
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            ModelConfig: 配置对象
        """
        # 只保留 ModelConfig 需要的参数
        valid_params = {
            'model_size', 'vocab_size', 'd_model', 'nhead', 'num_layers',
            'dim_feedforward', 'dropout', 'attention_dropout', 'hidden_dropout',
            'layer_norm_eps', 'max_seq_length', 'pad_token_id', 'bos_token_id',
            'eos_token_id'
        }
        
        # 过滤出有效参数
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in valid_params
        }
        
        # 使用默认值补充缺失参数
        defaults = {
            'model_size': "tiny",
            'vocab_size': 10000,
            'd_model': 128,
            'nhead': 2,
            'num_layers': 2,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'hidden_dropout': 0.1,
            'layer_norm_eps': 1e-5,
            'max_seq_length': 512,
            'pad_token_id': 0,
            'bos_token_id': 2,
            'eos_token_id': 3,
        }
        
        # 更新默认值
        defaults.update(filtered_dict)
        
        # 创建实例
        return cls(**defaults)

    def to_dict(self):
        """将配置转换为字典"""
        return {
            'model_size': self.model_size,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'max_seq_length': self.max_seq_length,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
        }

    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  nhead={self.nhead},\n"
            f"  num_layers={self.num_layers},\n"
            f"  dim_feedforward={self.dim_feedforward},\n"
            f"  dropout={self.dropout},\n"
            f"  attention_dropout={self.attention_dropout},\n"
            f"  hidden_dropout={self.hidden_dropout},\n"
            f"  layer_norm_eps={self.layer_norm_eps},\n"
            f"  max_seq_length={self.max_seq_length},\n"
            f"  pad_token_id={self.pad_token_id},\n"
            f"  bos_token_id={self.bos_token_id},\n"
            f"  eos_token_id={self.eos_token_id}\n"
            f")"
        )
    
    @classmethod
    def from_pretrained(cls, model_size: str) -> 'ModelConfig':
        """从预定义配置加载
        
        Args:
            model_size: 模型大小，可选 'tiny', 'small', 'medium'
            
        Returns:
            ModelConfig 实例
        """
        # 预定义配置
        configs = {
            'tiny': {
                'd_model': 128,
                'nhead': 2,
                'num_layers': 2,
                'dim_feedforward': 512,
                'max_seq_length': 512,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'hidden_dropout': 0.1
            },
            'small': {
                'd_model': 256,
                'nhead': 4,
                'num_layers': 4,
                'dim_feedforward': 1024,
                'max_seq_length': 512,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'hidden_dropout': 0.1
            },
            'medium': {
                'd_model': 512,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 2048,
                'max_seq_length': 512,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'hidden_dropout': 0.1
            }
        }
        
        if model_size not in configs:
            raise ValueError(f"未知的模型大小: {model_size}，可选: {list(configs.keys())}")
            
        # 创建配置实例
        config = configs[model_size]
        # 临时设置一个词表大小，后续会被更新
        config['vocab_size'] = 32000
        
        return cls(**config)
    
    def save_pretrained(self, save_dir: str):
        """保存配置到文件
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        config_file = os.path.join(save_dir, 'config.json')
        
        # 将配置转换为字典
        config_dict = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'layer_norm_eps': self.layer_norm_eps,
        }
        
        # 保存到文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, config_file: str) -> 'ModelConfig':
        """从 JSON 文件加载配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            ModelConfig 实例
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 16
    max_length: int = 512
    
    # 优化器参数
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    
    # 学习率调度
    scheduler: str = 'linear'  # 'linear' 或 'cosine'
    num_training_steps: Optional[int] = None  # 总训练步数，在 __post_init__ 中计算
    
    # 优化器超参数
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu: int = torch.cuda.device_count()
    fp16: bool = True
    
    # 保存和输出
    output_dir: str = 'outputs'
    save_every: int = 1
    
    # 早停配置
    patience: int = 3
    min_delta: float = 1e-4
    
    # 对话生成参数
    max_history_turns: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __post_init__(self):
        """初始化后的处理"""
        # 确保设备配置正确
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"不支持的设备类型: {self.device}")
        
        # 在非 CUDA 设备上禁用 fp16
        if self.device != 'cuda':
            self.fp16 = False
            
        # 确保学习率调度器类型正确
        if self.scheduler not in ['linear', 'cosine']:
            raise ValueError(f"不支持的学习率调度器: {self.scheduler}")
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def update_training_steps(self, train_dataloader):
        """更新总训练步数
        
        Args:
            train_dataloader: 训练数据加载器
        """
        self.num_training_steps = (
            len(train_dataloader) 
            * self.num_epochs 
            // self.gradient_accumulation_steps
        ) 
from dataclasses import dataclass
from typing import Optional
import torch

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
        initializer_range: float = 0.02,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ):
        """初始化模型配置
        
        参数:
            model_size: 模型大小 (tiny/small/medium)
            vocab_size: 词表大小
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: transformer层数
            dim_feedforward: 前馈网络维度
            dropout: dropout比率
            attention_dropout: 注意力层的dropout比率
            hidden_dropout: 隐藏层的dropout比率
            layer_norm_eps: 层归一化的epsilon值
            initializer_range: 初始化范围
            max_seq_length: 最大序列长度
            pad_token_id: padding token的ID
            bos_token_id: 开始符的ID
            eos_token_id: 结束符的ID
        """
        # 先设置基本参数
        self.model_size = model_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # 根据模型大小更新参数
        if self.model_size == "small":
            self.vocab_size = 30000
            self.d_model = 256
            self.nhead = 4
            self.num_layers = 4
            self.dim_feedforward = 1024
        elif self.model_size == "medium":
            self.vocab_size = 50000
            self.d_model = 512
            self.nhead = 8
            self.num_layers = 6
            self.dim_feedforward = 2048
    
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
            f"  initializer_range={self.initializer_range},\n"
            f"  max_seq_length={self.max_seq_length},\n"
            f"  pad_token_id={self.pad_token_id},\n"
            f"  bos_token_id={self.bos_token_id},\n"
            f"  eos_token_id={self.eos_token_id}\n"
            f")"
        )
    
    def to_dict(self) -> dict:
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
            'initializer_range': self.initializer_range,
            'max_seq_length': self.max_seq_length,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """从字典创建配置"""
        # 设置默认值
        defaults = {
            'model_size': 'tiny',
            'vocab_size': 10000,
            'd_model': 128,
            'nhead': 2,
            'num_layers': 2,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'hidden_dropout': 0.1,
            'layer_norm_eps': 1e-5,
            'initializer_range': 0.02,
            'max_seq_length': 512,
            'pad_token_id': 0,
            'bos_token_id': 2,
            'eos_token_id': 3,
        }
        # 更新默认值
        defaults.update(config_dict)
        return cls(**defaults)

@dataclass
class TrainingConfig:
    # 基础配置
    model_type: str = "chat"  # "chat" 或 "generation"
    max_length: int = 512
    batch_size: int = 16
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu: int = torch.cuda.device_count()
    fp16: bool = True  # 是否使用混合精度训练
    
    # 对话相关
    max_history_turns: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 训练策略
    warmup_steps: int = 1000
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 优化器配置
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8 
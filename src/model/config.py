from dataclasses import dataclass
from typing import Optional

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
            'max_seq_length': 512,
            'pad_token_id': 0,
            'bos_token_id': 2,
            'eos_token_id': 3,
        }
        # 更新默认值
        defaults.update(config_dict)
        return cls(**defaults) 
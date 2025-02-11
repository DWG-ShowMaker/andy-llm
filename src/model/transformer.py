import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码模块
    
    将位置信息编码到输入的张量中，使模型能够感知序列中token的位置关系。
    使用正弦和余弦函数的组合来生成位置编码。
    """
    def __init__(self, d_model: int, max_seq_length: int):
        """
        参数:
            d_model: 模型的隐藏维度
            max_seq_length: 最大序列长度
        """
        super().__init__()
        # 生成位置矩阵 [max_seq_length, 1]
        position = torch.arange(max_seq_length).unsqueeze(1)
        # 生成分母项
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, 1, d_model)
        # 使用正弦函数编码偶数维度
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数编码奇数维度
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 将位置编码注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [seq_len, batch_size, embedding_dim]
        返回:
            添加了位置编码的张量
        """
        return x + self.pe[:x.size(0)]

class MiniLLM(nn.Module):
    """小型语言模型的主体架构
    
    实现了一个基于Transformer的语言模型，包含:
    1. Token嵌入层
    2. 位置编码
    3. Transformer编码器层
    4. 输出层
    """
    def __init__(self, config):
        """
        参数:
            config: 模型配置对象，包含所有超参数
        """
        super().__init__()
        self.config = config
        
        # Token嵌入层：将输入的token ID转换为向量表示
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # 位置编码层：添加位置信息
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # 构建Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.d_model,  # 隐藏维度
            nhead=config.nhead,      # 注意力头数
            dim_feedforward=config.dim_feedforward,  # 前馈网络维度
            dropout=config.dropout,   # Dropout比率
            batch_first=True,        # 输入张量的batch维度在前
            norm_first=True          # 使用Pre-LN结构，提高训练稳定性
        )
        
        # 堆叠多层Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.num_layers
        )
        
        # 输出层：将隐藏状态映射回词表大小
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # 初始化模型参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数
        
        使用Xavier正态分布初始化所有维度大于1的参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播
        
        参数:
            input_ids: 输入token的ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 目标token的ID [batch_size, seq_len]
        """
        # 1. 嵌入层
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.pos_encoder(x)
        
        # 2. 处理注意力掩码
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # 3. Transformer层
        hidden_states = self.transformer(x, src_key_padding_mask=key_padding_mask)
        logits = self.output_layer(hidden_states)
        
        # 4. 计算损失
        loss = None
        if labels is not None:
            # 重塑logits和labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算损失，忽略padding位置
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # PyTorch默认使用-100作为忽略索引
                reduction='mean'
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """生成文本
        
        参数:
            input_ids: 输入序列
            max_length: 最大生成长度
            temperature: 采样温度，控制生成的随机性
            top_k: Top-K采样的K值
            top_p: Top-P采样的P值
            
        返回:
            生成的token序列
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        for _ in range(max_length):
            # 1. 获取模型预测
            logits, _ = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # 2. Top-K采样：只保留概率最高的K个token
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 3. Top-P采样（核采样）：只保留累积概率达到P的token
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 4. 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 5. 将新token添加到序列中
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 6. 检查是否生成了结束符号
            if (next_token == self.config.eos_token_id).all():
                break
        
        return input_ids 
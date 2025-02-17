import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

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
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
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
        
        # KV缓存相关
        self.enable_cache = False
        self.cached_states = None
        self.cached_length = 0
    
    def _init_parameters(self):
        """初始化模型参数
        
        使用Xavier正态分布初始化所有维度大于1的参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def enable_kv_cache(self):
        """启用KV缓存"""
        self.enable_cache = True
        self.cached_states = None
        self.cached_length = 0
    
    def disable_kv_cache(self):
        """禁用KV缓存"""
        self.enable_cache = False
        self.cached_states = None
        self.cached_length = 0
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        参数:
            input_ids: 输入token的ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 目标token的ID [batch_size, seq_len]
            use_cache: 是否使用KV缓存
        """
        # 1. 嵌入层
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        
        # 2. 处理缓存和位置编码
        if self.enable_cache and use_cache:
            if self.cached_states is None:
                # 首次运行，添加位置编码
                x = self.pos_encoder(x)
                self.cached_length = x.size(1)
            else:
                # 使用缓存时，只对新token添加位置编码
                new_tokens = x[:, -1:, :]
                position_ids = torch.arange(
                    self.cached_length,
                    self.cached_length + 1,
                    dtype=torch.long,
                    device=x.device
                )
                new_tokens = new_tokens + self.pos_encoder.pe[position_ids]
                x = new_tokens
                self.cached_length += 1
        else:
            x = self.pos_encoder(x)
        
        # 3. 处理注意力掩码
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # 4. Transformer层
        if self.enable_cache and use_cache:
            if self.cached_states is None:
                # 首次运行，生成并存储状态
                hidden_states = self.transformer(x, src_key_padding_mask=key_padding_mask)
                self.cached_states = hidden_states
            else:
                # 使用缓存的状态
                new_hidden_states = self.transformer(x, src_key_padding_mask=key_padding_mask)
                self.cached_states = torch.cat([self.cached_states, new_hidden_states], dim=1)
                hidden_states = self.cached_states
        else:
            hidden_states = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # 5. 输出层
        logits = self.output_layer(hidden_states)
        
        # 6. 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> torch.Tensor:
        """生成文本"""
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # 存储生成的序列
        generated_ids = input_ids
        
        # 生成文本
        while cur_len < max_length:
            # 前向传播获取logits
            logits = self.forward(
                input_ids=generated_ids,
                use_cache=True
            )[0]  # forward返回(logits, loss)，我们只需要logits
            
            # 获取最后一个时间步的logits
            next_token_logits = logits[:, -1, :]
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # 应用top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 应用top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累积超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样或贪婪解码
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 拼接新token
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            cur_len += 1
            
            # 检查是否生成了结束符
            if (generated_ids == self.config.eos_token_id).any(dim=1).all():
                break
        
        return generated_ids 
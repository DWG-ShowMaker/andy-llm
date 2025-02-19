import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict

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
    
    def to(self, device):
        """将模型移动到指定设备"""
        super().to(device)
        return self
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        参数:
            input_ids: 输入token的ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 目标token的ID [batch_size, seq_len]
            use_cache: 是否使用KV缓存
            past_key_values: 上一轮对话的KV缓存
        """
        # 确保输入张量在正确的设备上
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # 1. 嵌入层
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        
        # 2. 位置编码
        if past_key_values is None:
            # 首次对话，使用完整的位置编码
            x = self.pos_encoder(x)
        else:
            # 延续对话，只对新输入添加位置编码
            position_ids = torch.arange(
                past_key_values[0][0].size(-2),
                past_key_values[0][0].size(-2) + x.size(1),
                dtype=torch.long,
                device=x.device
            )
            x = x + self.pos_encoder.pe[position_ids]
        
        # 3. Transformer层
        hidden_states = self.transformer(
            x,
            src_key_padding_mask=attention_mask == 0 if attention_mask is not None else None,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # 4. 输出层
        logits = self.output_layer(hidden_states)
        
        # 5. 计算损失
        loss = None
        if labels is not None:
            # 使用 shift_logits 和 shift_labels 来计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """生成文本
        
        参数:
            input_ids: 输入token的ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: 核采样的概率阈值
            top_k: top-k采样的k值
            repetition_penalty: 重复惩罚系数
            do_sample: 是否使用采样
            pad_token_id: 填充token的ID
            eos_token_id: 结束token的ID
            past_key_values: 上一轮对话的KV缓存
        """
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # 存储生成的序列
        generated_ids = input_ids
        
        # 生成文本
        while cur_len < max_length:
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values
            )
            logits = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
            
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
            if (generated_ids == eos_token_id).any(dim=1).all():
                break
        
        return generated_ids 

    def quantize(self, quantization_type='dynamic'):
        """量化模型
        
        参数:
            quantization_type: 量化类型，可选 'dynamic' 或 'static'
        """
        if quantization_type == 'dynamic':
            # 动态量化，将线性层和嵌入层量化为INT8
            self.token_embedding = torch.quantization.quantize_dynamic(
                self.token_embedding,
                {torch.nn.Embedding},
                dtype=torch.qint8
            )
            self.output_layer = torch.quantization.quantize_dynamic(
                self.output_layer,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            self.transformer = torch.quantization.quantize_dynamic(
                self.transformer,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # 为Embedding层设置特殊的量化配置
            float_qparams_config = torch.quantization.float_qparams_weight_only_qconfig
            # 为其他层设置默认配置
            default_qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # 设置不同层的量化配置
            self.qconfig_dict = {
                'token_embedding': float_qparams_config,
                '': default_qconfig  # 全局默认配置
            }
            
            # 准备量化感知训练
            if self.training:
                torch.quantization.prepare_qat(self, inplace=True)
            else:
                torch.quantization.prepare(self, inplace=True)
            
            # 转换为量化模型
            torch.quantization.convert(self, inplace=True)
        
        return self 

    def chat(
        self,
        tokenizer,
        messages: List[Dict[str, str]],
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        system_prompt: str = "你是一个有帮助的AI助手。"
    ) -> str:
        """对话生成
        
        参数:
            tokenizer: 分词器
            messages: 对话历史，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: 核采样的概率阈值
            top_k: top-k采样的k值
            repetition_penalty: 重复惩罚系数
            do_sample: 是否使用采样
            system_prompt: 系统提示词
        """
        # 构建输入文本
        input_text = f"<system>{system_prompt}</system>\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                input_text += f"<human>{content}</human>\n"
            elif role == "assistant":
                input_text += f"<assistant>{content}</assistant>\n"
        input_text += "<assistant>"  # 添加生成标记
        
        # 编码输入
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 启用 KV 缓存
        self.enable_kv_cache()
        
        # 生成回复
        with torch.no_grad():
            output_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("</assistant>", add_special_tokens=False)[0]
            )
        
        # 禁用 KV 缓存
        self.disable_kv_cache()
        
        # 解码输出
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 提取助手回复部分
        response = response.split("<assistant>")[-1].split("</assistant>")[0].strip()
        
        return response 
import os
import sentencepiece as spm
from typing import List, Optional

class Tokenizer:
    def __init__(self, model_path: str):
        """初始化分词器
        
        Args:
            model_path: SentencePiece 模型路径
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # 特殊 token IDs
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        
        # 词表大小
        self.vocab_size = self.sp.get_piece_size()
    
    @classmethod
    def train(cls, texts: List[str], vocab_size: int, model_path: str) -> 'Tokenizer':
        """训练分词器
        
        Args:
            texts: 训练文本列表
            vocab_size: 词表大小
            model_path: 模型保存路径
            
        Returns:
            训练好的分词器
        """
        # 准备训练数据
        data_path = f"{model_path}.txt"
        with open(data_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # 训练参数
        train_args = [
            f'--input={data_path}',
            f'--model_prefix={model_path}',
            f'--vocab_size={vocab_size}',
            '--character_coverage=0.9995',
            '--model_type=bpe',
            '--pad_id=0',
            '--unk_id=1',
            '--bos_id=2',
            '--eos_id=3',
            '--pad_piece=<pad>',
            '--unk_piece=<unk>',
            '--bos_piece=<s>',
            '--eos_piece=</s>',
            '--user_defined_symbols=<system>,</system>,<human>,</human>,<assistant>,</assistant>'
        ]
        
        # 训练分词器
        spm.SentencePieceTrainer.train(' '.join(train_args))
        
        # 清理临时文件
        if os.path.exists(data_path):
            os.remove(data_path)
        
        # 返回训练好的分词器
        return cls(f"{model_path}.model")
    
    @classmethod
    def load(cls, model_path: str) -> 'Tokenizer':
        """加载分词器
        
        Args:
            model_path: 模型路径
            
        Returns:
            Tokenizer 实例
        """
        return cls(model_path)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本编码为 token IDs
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token
            
        Returns:
            token IDs 列表
        """
        ids = self.sp.encode_as_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """将 token IDs 解码为文本
        
        Args:
            ids: token IDs 列表
            skip_special_tokens: 是否跳过特殊 token
            
        Returns:
            解码后的文本
        """
        if skip_special_tokens:
            # 过滤掉特殊 token
            special_ids = {self.pad_token_id, self.unk_token_id, 
                          self.bos_token_id, self.eos_token_id}
            ids = [id for id in ids if id not in special_ids]
        
        # 过滤掉超出范围的 token ID
        valid_ids = []
        for id in ids:
            if 0 <= id < self.vocab_size:
                valid_ids.append(id)
            else:
                print(f"警告: token ID {id} 超出词表范围 [0, {self.vocab_size})")
                valid_ids.append(self.unk_token_id)
        
        # 如果没有有效的 token ID，返回空字符串
        if not valid_ids:
            return ""
        
        return self.sp.decode_ids(valid_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分词为 token 列表
        
        Args:
            text: 输入文本
            
        Returns:
            token 列表
        """
        return self.sp.encode_as_pieces(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将 token 列表转换为 ID 列表
        
        Args:
            tokens: token 列表
            
        Returns:
            ID 列表
        """
        return self.sp.piece_to_id(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """将 ID 列表转换为 token 列表
        
        Args:
            ids: ID 列表
            
        Returns:
            token 列表
        """
        return self.sp.id_to_piece(ids) 
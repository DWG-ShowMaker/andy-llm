import os
import sentencepiece as spm
from typing import List
import json

def train_tokenizer(
    input_files: List[str],
    vocab_size: int,
    model_prefix: str,
    model_type: str = "bpe"
):
    """训练 SentencePiece 分词器
    
    参数:
        input_files: 训练文件路径列表
        vocab_size: 词表大小
        model_prefix: 模型保存路径前缀
        model_type: 分词器类型 (bpe/unigram/char/word)
    """
    # 合并所有输入文件到一个临时文件
    temp_file = "data/processed/temp_all.txt"
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    
    with open(temp_file, 'w', encoding='utf-8') as outf:
        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as inf:
                for line in inf:
                    # 解析jsonl并提取文本
                    item = json.loads(line)
                    text = item['text'].strip()
                    if text:
                        outf.write(text + '\n')
    
    # 训练参数
    train_args = {
        'input': temp_file,                # 输入文件
        'model_prefix': model_prefix,      # 模型保存路径前缀
        'vocab_size': vocab_size,          # 词表大小
        'character_coverage': 0.9995,      # 字符覆盖率
        'model_type': model_type,          # 模型类型
        'pad_id': 0,                       # padding的ID
        'unk_id': 1,                       # 未知词的ID
        'bos_id': 2,                       # 句子开始的ID
        'eos_id': 3,                       # 句子结束的ID
        'pad_piece': '[PAD]',             # padding的标记
        'unk_piece': '[UNK]',             # 未知词的标记
        'bos_piece': '[BOS]',             # 句子开始的标记
        'eos_piece': '[EOS]',             # 句子结束的标记
        'user_defined_symbols': ['[MASK]'],# 添加特殊token
        'split_by_whitespace': False,      # 是否按空格分割
        'byte_fallback': True,             # 处理未知字符
        'num_threads': 8,                  # 使用多线程加速
        'train_extremely_large_corpus': True, # 优化大规模语料训练
    }
    
    # 训练分词器
    spm.SentencePieceTrainer.train(**train_args)
    print(f"分词器训练完成！保存至 {model_prefix}.model 和 {model_prefix}.vocab")
    
    # 清理临时文件
    os.remove(temp_file)

def main():
    # 创建处理后的数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 准备训练文件
    input_files = ["data/raw/train.jsonl"]
    if os.path.exists("data/raw/validation.jsonl"):
        input_files.append("data/raw/validation.jsonl")
    
    # 检查文件是否存在且非空
    valid_files = []
    for file_path in input_files:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            valid_files.append(file_path)
    
    if not valid_files:
        raise ValueError("没有找到有效的训练文件！请先运行download_data.py下载数据。")
    
    # 训练分词器
    train_tokenizer(
        input_files=valid_files,
        vocab_size=10000,  # 使用tiny模型的词表大小
        model_prefix="data/processed/tokenizer"
    )
    
    # 测试分词器
    print("\n测试分词器:")
    sp = spm.SentencePieceProcessor()
    sp.load("data/processed/tokenizer.model")
    
    test_texts = [
        "人工智能正在快速发展。",
        "这是一个测试句子。",
    ]
    
    for text in test_texts:
        tokens = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"\n原文: {text}")
        print(f"分词: {tokens}")
        print(f"ID: {ids}")

if __name__ == "__main__":
    main() 
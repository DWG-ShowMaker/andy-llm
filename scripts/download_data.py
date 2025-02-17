import os
import json
from datasets import load_dataset
from tqdm import tqdm
import random

def save_jsonl(data, file_path):
    """保存数据为jsonl格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def filter_data(item):
    """过滤低质量数据"""
    text = item['text']
    # 过滤过短文本
    if len(text) < 50:
        return False
    # 过滤重复内容
    if text.count('。') < 2:
        return False
    # 过滤广告等垃圾文本
    spam_keywords = ['优惠', '促销', 'www', 'http']
    if any(keyword in text for keyword in spam_keywords):
        return False
    return True

def main():
    """下载并处理中文数据集
    
    我们使用以下数据集组合：
    1. CLUE的新闻数据集
    2. 中文维基百科摘要数据
    3. 社区问答数据
    """
    # 创建数据目录
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        # 下载CLUE的新闻数据集
        print("正在下载CLUE新闻数据集...")
        news_dataset = load_dataset("clue", "tnews")
        
        # 下载中文维基百科摘要
        print("正在下载中文维基百科摘要...")
        try:
            wiki_dataset = load_dataset("wikipedia", "20220301.zh", split="train[:10000]")
        except Exception as e:
            print(f"下载维基百科数据失败: {e}")
            wiki_dataset = []
        
        # 下载社区问答数据
        print("正在下载社区问答数据...")
        qa_dataset = load_dataset("shibing624/medical", split="train")
        
        # 合并数据
        combined_data = []
        
        # 处理新闻数据
        print("处理新闻数据...")
        for item in tqdm(news_dataset['train']):
            if item['sentence'].strip():
                combined_data.append({
                    'text': item['sentence'],
                    'metadata': {
                        'source': 'clue_news',
                        'type': 'news',
                        'category': item.get('label_desc', '')
                    }
                })
        
        # 处理维基百科数据
        print("处理维基百科数据...")
        for item in tqdm(wiki_dataset):
            if item['text'].strip():
                paragraphs = [p.strip() for p in item['text'].split('\n') if len(p.strip()) > 50]
                if paragraphs:
                    combined_data.append({
                        'text': '\n\n'.join(paragraphs),
                        'metadata': {
                            'source': 'wikipedia',
                            'type': 'article',
                            'title': item.get('title', '')
                        }
                    })
        
        # 处理问答数据
        print("处理问答数据...")
        for item in tqdm(qa_dataset):
            if item['question'].strip() and item['answer'].strip():
                combined_data.append({
                    'text': f"问题：{item['question']}\n答案：{item['answer']}",
                    'metadata': {
                        'source': 'medical_qa',
                        'type': 'qa'
                    }
                })
        
        # 打乱数据
        print(f"总数据量: {len(combined_data)}")
        random.shuffle(combined_data)
        
        # 分割数据集
        total_size = len(combined_data)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        
        train_data = combined_data[:train_size]
        val_data = combined_data[train_size:train_size+val_size]
        test_data = combined_data[train_size+val_size:]
        
        # 过滤数据
        filtered_data = [item for item in combined_data if filter_data(item)]
        
        # 保存数据
        print("保存数据...")
        save_jsonl(filtered_data[:train_size], "data/raw/train.jsonl")
        save_jsonl(filtered_data[train_size:train_size+val_size], "data/raw/validation.jsonl")
        save_jsonl(filtered_data[train_size+val_size:], "data/raw/test.jsonl")
        
    except Exception as e:
        print(f"下载主要数据集失败: {e}")
        print("使用备用数据源...")
        
        try:
            # 使用CLUEWSC2020数据集作为备用
            print("正在下载CLUEWSC2020数据集...")
            backup_dataset = load_dataset("clue", "cluewsc2020")
            
            # 处理数据
            backup_data = []
            for split in ['train', 'validation', 'test']:
                for item in tqdm(backup_dataset[split], desc=f"处理{split}数据"):
                    if item['text'].strip():
                        backup_data.append({
                            'text': item['text'],
                            'metadata': {
                                'source': 'cluewsc',
                                'type': 'sentence',
                                'span1': item.get('span1_text', ''),
                                'span2': item.get('span2_text', '')
                            }
                        })
            
            # 打乱并分割数据
            random.shuffle(backup_data)
            total_size = len(backup_data)
            train_size = int(total_size * 0.8)
            val_size = int(total_size * 0.1)
            
            # 过滤数据
            filtered_data = [item for item in backup_data if filter_data(item)]
            
            # 保存数据
            save_jsonl(filtered_data[:train_size], "data/raw/train.jsonl")
            save_jsonl(filtered_data[train_size:train_size+val_size], "data/raw/validation.jsonl")
            save_jsonl(filtered_data[train_size+val_size:], "data/raw/test.jsonl")
            
        except Exception as e:
            print(f"下载备用数据集也失败了: {e}")
            print("创建最小示例数据...")
            
            # 创建示例数据
            example_data = [
                {
                    'text': "人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。",
                    'metadata': {'source': 'example', 'type': 'definition'}
                },
                {
                    'text': "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
                    'metadata': {'source': 'example', 'type': 'definition'}
                },
                {
                    'text': "大语言模型是一种基于深度学习的自然语言处理模型，它能够理解和生成人类语言。",
                    'metadata': {'source': 'example', 'type': 'definition'}
                }
            ]
            
            # 分割并保存示例数据
            train_size = int(len(example_data) * 0.8)
            val_size = int(len(example_data) * 0.1)
            
            # 过滤数据
            filtered_data = [item for item in example_data if filter_data(item)]
            
            save_jsonl(filtered_data[:train_size], "data/raw/train.jsonl")
            save_jsonl(filtered_data[train_size:train_size+val_size], "data/raw/validation.jsonl")
            save_jsonl(filtered_data[train_size+val_size:], "data/raw/test.jsonl")
    
    print("数据下载和处理完成！")
    # 打印数据集大小信息
    for split in ['train', 'validation', 'test']:
        path = f"data/raw/{split}.jsonl"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
                print(f"{split}集大小: {count}条数据")

if __name__ == "__main__":
    main() 
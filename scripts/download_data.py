import os
from datasets import load_dataset
import json

def main():
    """下载并处理中文数据集
    
    我们使用以下数据集：
    1. CLUE的新闻数据集
    2. 维基百科中文数据（最新版本）
    3. 书籍语料
    """
    # 创建数据目录
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        # 尝试下载CLUE新闻数据集
        print("正在下载CLUE新闻数据集...")
        dataset = load_dataset("clue", "tnews")
        
        # 保存训练集
        with open("data/raw/train.txt", 'w', encoding='utf-8') as f:
            for item in dataset['train']:
                if 'sentence' in item and item['sentence'].strip():
                    f.write(item['sentence'].strip() + '\n')
        
        # 保存验证集
        with open("data/raw/validation.txt", 'w', encoding='utf-8') as f:
            for item in dataset['validation']:
                if 'sentence' in item and item['sentence'].strip():
                    f.write(item['sentence'].strip() + '\n')
        
        # 保存测试集
        with open("data/raw/test.txt", 'w', encoding='utf-8') as f:
            for item in dataset['test']:
                if 'sentence' in item and item['sentence'].strip():
                    f.write(item['sentence'].strip() + '\n')
                    
    except Exception as e:
        print(f"下载CLUE数据集失败: {e}")
        # 如果CLUE数据集下载失败，尝试其他数据集
        try:
            print("正在下载中文维基百科数据集...")
            wiki_dataset = load_dataset("wikipedia", "20230901.zh")
            
            # 计算分割点
            total_size = len(wiki_dataset['train'])
            train_size = int(total_size * 0.8)
            val_size = int(total_size * 0.1)
            
            # 保存数据
            splits = {
                'train': wiki_dataset['train'].select(range(train_size)),
                'validation': wiki_dataset['train'].select(range(train_size, train_size + val_size)),
                'test': wiki_dataset['train'].select(range(train_size + val_size, total_size))
            }
            
            for split_name, split_data in splits.items():
                with open(f"data/raw/{split_name}.txt", 'w', encoding='utf-8') as f:
                    for item in split_data['text']:
                        if len(item.strip()) > 10:  # 过滤过短的文本
                            f.write(item.strip() + '\n\n')
                            
        except Exception as e:
            print(f"下载维基百科数据集也失败了: {e}")
            print("使用示例数据集...")
            
            # 创建一个小的示例数据集用于测试
            example_texts = [
                "人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。",
                "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
                "大语言模型是一种基于深度学习的自然语言处理模型，它能够理解和生成人类语言。",
                "强化学习是机器学习的重要方法之一，通过agent与环境的交互来学习最优策略。",
                # ... 可以添加更多示例文本
            ]
            
            # 保存示例数据
            splits = {
                'train': example_texts[:int(len(example_texts)*0.8)],
                'validation': example_texts[int(len(example_texts)*0.8):int(len(example_texts)*0.9)],
                'test': example_texts[int(len(example_texts)*0.9):]
            }
            
            for split_name, texts in splits.items():
                with open(f"data/raw/{split_name}.txt", 'w', encoding='utf-8') as f:
                    for text in texts:
                        f.write(text + '\n')
    
    print("数据下载和处理完成！")
    # 打印数据集大小信息
    for split in ['train', 'validation', 'test']:
        path = f"data/raw/{split}.txt"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"{split}集大小: {len(lines)}行")

if __name__ == "__main__":
    main() 
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DatasetConfig:
    # 必需参数（没有默认值）
    name: str
    source: str  # 'huggingface' 或 'modelscope'
    path: str
    text_column: str
    
    # 可选参数（有默认值）
    subset: Optional[str] = None
    min_length: int = 50
    max_length: int = 512
    filter_rules: List[str] = field(default_factory=list)

# 预定义数据集配置
DATASET_CONFIGS = {
    # 通用百科数据
    "wiki": DatasetConfig(
        name="wiki",
        source="modelscope",
        path="chinese-wikipedia-corpus",
        text_column="text",
        min_length=100,
        filter_rules=["长度过滤", "质量过滤"]
    ),
    
    # 科技文献
    "csl": DatasetConfig(
        name="csl",
        source="modelscope",
        path="chinese-scientific-literature",
        text_column="text",
        min_length=100,
        filter_rules=["学术格式化"]
    ),
    
    # 网络小说
    "novel": DatasetConfig(
        name="novel",
        source="modelscope",
        path="chinese-novel-corpus",
        text_column="content",
        min_length=200,
        filter_rules=["章节合并"]
    ),
    
    # 新闻语料
    "news": DatasetConfig(
        name="news",
        source="modelscope",
        path="chinese-news-corpus",
        text_column="content",
        min_length=100,
        filter_rules=["新闻格式化"]
    ),

    # 对话数据
    "dialogue": DatasetConfig(
        name="dialogue",
        source="modelscope",
        path="chinese-dialog-corpus",
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    )
} 
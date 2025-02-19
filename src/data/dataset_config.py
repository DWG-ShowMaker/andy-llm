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
    # 通用中文语料
    "wiki": DatasetConfig(
        name="wiki",
        source="huggingface",
        path="wikipedia",
        subset="20220301.zh",
        text_column="text",
        min_length=100,
        filter_rules=["长度过滤", "质量过滤"]
    ),
    
    # Belle对话数据
    "belle": DatasetConfig(
        name="belle",
        source="huggingface",
        path="BelleGroup/school_math_0.25M",  # 数学问答数据集
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    ),
    
    # Alpaca中文指令数据
    "alpaca": DatasetConfig(
        name="alpaca",
        source="huggingface",
        path="yahma/alpaca-cleaned",  # 清洗后的Alpaca数据集
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    ),
    
    # 通用知识问答
    "qa": DatasetConfig(
        name="qa",
        source="huggingface",
        path="Hello-SimpleAI/HC3-Chinese",  # 中文问答数据集
        text_column="human",
        min_length=50,
        filter_rules=["对话格式化"]
    ),

    # 中文新闻数据
    "news": DatasetConfig(
        name="news",
        source="huggingface",
        path="MLXD/chinese-news-2023",  # 中文新闻数据集
        text_column="content",
        min_length=100,
        filter_rules=["新闻格式化"]
    )
} 
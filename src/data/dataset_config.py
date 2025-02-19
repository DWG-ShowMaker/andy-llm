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
    # 指令微调数据
    "belle": DatasetConfig(
        name="belle",
        source="huggingface",
        path="BelleGroup/train_2M_CN",
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    ),
    
    # Alpaca 中文指令数据
    "alpaca": DatasetConfig(
        name="alpaca",
        source="huggingface",
        path="silk-road/alpaca-data-gpt4-chinese",
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    ),

    # ChatGPT 对话数据
    "chatgpt": DatasetConfig(
        name="chatgpt",
        source="huggingface",
        path="fnlp/moss-003-sft-data",
        text_column="conversation",
        min_length=50,
        filter_rules=["对话格式化"]
    ),

    # 医疗问答数据
    "medical": DatasetConfig(
        name="medical",
        source="huggingface",
        path="michael-wzhu/ChatMed_Consult_Dataset",
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    ),

    # 通用知识问答
    "qa": DatasetConfig(
        name="qa",
        source="huggingface",
        path="wangrui6/Zhihu-KOL",
        text_column="text",
        min_length=50,
        filter_rules=["对话格式化"]
    )
} 
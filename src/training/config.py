from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """训练配置"""
    device: str = 'cuda'  # 训练设备
    learning_rate: float = 1e-4  # 学习率
    num_epochs: int = 3  # 训练轮数
    output_dir: str = 'outputs'  # 输出目录
    fp16: bool = True  # 是否使用混合精度训练
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    weight_decay: float = 0.01  # 权重衰减
    warmup_steps: int = 100  # 预热步数
    scheduler: str = 'cosine'  # 学习率调度器类型
    num_training_steps: int = 1000  # 总训练步数
    n_gpu: int = 1  # GPU数量
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    patience: int = 3  # 早停耐心值
    min_delta: float = 0.001  # 早停最小改善值 
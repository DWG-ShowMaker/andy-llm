class EarlyStopping:
    """早停机制实现
    
    参数:
        patience (int): 容忍多少个epoch验证集指标没有改善
        min_delta (float): 最小改善阈值，小于这个值视为没有改善
        mode (str): 'min' 表示指标越小越好，'max' 表示指标越大越好
    """
    def __init__(self, patience=3, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        检查是否应该早停
        
        参数:
            val_loss (float): 当前epoch的验证集损失值
            
        返回:
            bool: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if self.mode == 'min':
            delta = val_loss - self.best_loss
            improvement = delta < -self.min_delta
        else:
            delta = self.best_loss - val_loss
            improvement = delta < -self.min_delta
            
        if improvement:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False 
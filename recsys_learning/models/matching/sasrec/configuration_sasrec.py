import torch

class SASRecConfig:
    def __init__(self):
        # 数据集
        self.num_users = 1000           # 用户数量
        self.num_items = 5000           # 物品数量
        self.train_radio = 0.8          # 训练集比例
        
        # 模型
        self.max_len = 50               # 最大序列长度
        self.embedding_dim = 64         # 隐藏层维度
        self.num_blocks = 2             # Transformer 块数量
        self.num_heads = 4              # 注意力头数量
        
        # 训练
        self.dropout_rate = 0.2         # 丢弃概率, 用于防止过拟合
        self.batch_size = 32            # 批次大小
        self.learning_rate = 0.001      # 学习率
        self.epochs = 5                 # 训练轮数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neg_sample_count = 1       # 负样本数量
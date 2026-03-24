import torch
import numpy as np

def get_auto_embedding_dim(num_classes):
    """根据特征数量自动计算嵌入维度

    公式
    emb_dim = floor(6 * num_classes**0.25)

    理论来源
    Deep & Cross Network (DCN) 论文(ADKDD'17)

    Args:
        num_classes (int): 特征数量
    
    Returns:
        int: 推荐的嵌入维度
    """
    return int(np.floor(6 * np.pow(num_classes, 0.25)))

class NormalEmbeddingInit(object):
    """
    一个正态分布Embedding初始化器

    Args:
        mean (float): 正态分布均值
        std (float): 正态分布标准差
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim, padding_idx=None):
        # 1. 创建一个Embedding (思考: 此处的padding_idx有什么作用?)
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # 2. 正态分布初始化
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        # 3. 如果有padding_idx，则将padding_idx的权重设置为0 (思考: 第1步已经设置了 padding_idx， 此处为什么需要再次设置0?)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        # 4. 返回
        return embed
"""
在推荐系统中, 特征通常分为两类
1. Dense Feature (稠密特征): 没有固定的“字典”，直接就是一个具体的数字，如年龄、账户余额、价格
2. Sparse Feature (稠密/稀疏特征): 从一个有限集合中选出一个值，如性别、职业、学历
3. Sequence Feature (序列特征): 如用户行为序列、商品序列, 例如: 用户过去点击的 50 个商品 ID 序列 [101, 205, 302...]
4. Multi-hot/Tag Feature (多值/标签特征): 如商品标签、用户标签, 例如: 一部电影的标签 [喜剧, 动作]，对应 ID [1, 5]
"""

from recsys_learning.utils.embedding import get_auto_embedding_dim, NormalEmbeddingInit


class SequenceFeature(object):
    """
    用于处理序列特征或多热特征的类。
    在推荐系统中，我们有很多需要利用序列模型处理的用户行为特征，以及需要进行池化操作的标签特征（多热）。
    注意: 如果使用此特征, 必须在训练前对特征值进行补齐(padding)处理。

    Args:
        name (str): 特征名称
        vocab_size (int): 词表大小(特征数量)
        embed_dim (int): 嵌入矩阵维度
        pooling (str): 池化方式, 支持 `["mean", "sum", "concat"]` (default=`"mean"`)
        shared_with (str): 权重共享, 例如点击序列和购买序列可能共享同一个物品Embedding表
        padding_idx (int, optional): padding_idx 对应的条目认为是 PADDING, 将会被Masked为0
        initializer(Initializer): Enbedding 权重初始化器
    """

    def __init__(self, name, vocab_size, embed_dim=None, pooling="mean", shared_with=None, padding_idx=None, initializer=NormalEmbeddingInit(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f'<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class SparseFeature(object):
    """The Feature Class for Sparse feature.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(self, name, vocab_size, embed_dim=None, shared_with=None, padding_idx=None, initializer=NormalEmbeddingInit(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f'<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class DenseFeature(object):
    """The Feature Class for Dense feature.

    Args:
        name (str): feature's name.
        embed_dim (int): embedding vector's length, the value fixed `1`. If you put a vector (torch.tensor) , replace the embed_dim with your vector dimension.
    """

    def __init__(self, name, embed_dim=1):
        self.name = name
        self.embed_dim = embed_dim

    def __repr__(self):
        return f'<DenseFeature {self.name}>'

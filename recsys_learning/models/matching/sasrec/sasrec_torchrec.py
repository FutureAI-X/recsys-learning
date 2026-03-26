"""
Date: create on 2022/5/8, update on 2022/5/8
References:
    paper: (ICDM'2018) Self-attentive sequential recommendation
    url: https://arxiv.org/pdf/1808.09781.pdf
    code: https://github.com/kang205/SASRec
Authors: Yuchen Wang, 615922749@qq.com
"""
import numpy as np
import torch
import torch.nn as nn

from recsys_learning.basic.features import DenseFeature, SequenceFeature, SparseFeature
from recsys_learning.basic.layers import MLP, EmbeddingLayer


class SASRec(torch.nn.Module):
    """SASRec: Self-Attentive Sequential Recommendation
    Args:
        features (list): 特征列表。在 SASRec 中，通常包含三个元素：[用户历史行为序列, 正样本序列, 负样本序列]
        max_len: 序列最大长度, 默认50
        dropout_rate: Dropout 率, 用于防止过拟合
        num_blocks: 堆叠的注意力层数
        num_heads: 多头注意力的头数
        item_feature: 可选的物品特征, 用于批内负采样(in-batch negative sampling)模式
    """

    def __init__(
        self,
        features,
        max_len=50,
        dropout_rate=0.5,
        num_blocks=2,
        num_heads=1,
        item_feature=None,
    ):
        super(SASRec, self).__init__()

        self.features = features            # 特征信息, 如[seq_feature, pos_feature, neg_feature]
        self.item_feature = item_feature    # 可选, 用于批内负采样(in-batch negative sampling)模式
        self.mode = None                    # 用于推理阶段, 指定计算用户表示还是物品表示: "user" or "item"
        self.max_len = max_len              # 最大序列长度

        self.item_num = self.features[0].vocab_size     # 物品总数
        self.embed_dim = self.features[0].embed_dim     # 嵌入向量的维度

        self.item_emb = EmbeddingLayer(self.features)   # 实例化嵌入层，用于将物品 ID 转换为向量。它接收 features 列表，可能处理共享嵌入（如正负样本与历史序列共享同一个物品 embedding 表）
        self.position_emb = torch.nn.Embedding(max_len, self.embed_dim)     # 位置嵌入层。SASRec 需要知道物品在序列中的顺序，因此为序列的每个位置（0 到 max_len-1）学习一个向量
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)                 # Dropout

        # 注意力块堆叠
        # 初始化 Transformer Encoder 的核心组件列表
        self.attention_layernorms = torch.nn.ModuleList()                   # Attention 前的层归一化 LayerNorm
        self.attention_layers = torch.nn.ModuleList()                       # 多头注意力
        self.forward_layernorms = torch.nn.ModuleList()                     # FFN 前层归一化 LayerNorm
        self.forward_layers = torch.nn.ModuleList()                         # FFN
        self.last_layernorm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)  # 最后的层归一化(块堆叠之后)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = torch.nn.MultiheadAttention(self.embed_dim, num_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.embed_dim, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def seq_forward(self, x, embed_x_feature):
        """
        x['seq'].shape: (batch_size, seq_len)
        embed_x_feature.shape: (batch_size, seq_len, embed_dim)
        """
        x = x['seq']                                                            # 获取历史行为序列, (batch_size, seq_len)

        embed_x_feature *= self.features[0].embed_dim**0.5                      # 乘以根号d, 这是 transformer 的标准操作, 有助于与稳定梯度
        embed_x_feature = embed_x_feature.squeeze()                             # 移除所有size为1的维度, 此处应无任何变化, (bacth_size, seq_len, embed_dim)

        positions = np.tile(np.array(range(x.shape[1])), [x.shape[0], 1])       # 位置编码数组, (batch_size, seq_len)
        embed_x_feature += self.position_emb(torch.LongTensor(positions))       # 查找位置嵌入，并将其添加到嵌入向量中

        embed_x_feature = self.emb_dropout(embed_x_feature)                     # 添加 Dropout

        timeline_mask = torch.BoolTensor(x == 0)                                # PADDING 掩码, (batch_size, seq_len)
        embed_x_feature *= ~timeline_mask.unsqueeze(-1)                         # PADDING 设置为0

        attention_mask = ~torch.tril(torch.ones((embed_x_feature.shape[1], embed_x_feature.shape[1]), dtype=torch.bool))    # 注意力掩码, (seq_len, seq_len)

        for i in range(len(self.attention_layers)):
            embed_x_feature = torch.transpose(embed_x_feature, 0, 1)
            Q = self.attention_layernorms[i](embed_x_feature)
            mha_outputs, _ = self.attention_layers[i](Q, embed_x_feature, embed_x_feature, attn_mask=attention_mask)

            embed_x_feature = Q + mha_outputs
            embed_x_feature = torch.transpose(embed_x_feature, 0, 1)

            embed_x_feature = self.forward_layernorms[i](embed_x_feature)
            embed_x_feature = self.forward_layers[i](embed_x_feature)
            embed_x_feature *= ~timeline_mask.unsqueeze(-1)

        seq_output = self.last_layernorm(embed_x_feature)

        return seq_output                                                       # (batch_size, seq_len, embed_dim)  

    def user_tower(self, x):
        """Compute user embedding for in-batch negative sampling.
        Takes the last valid position's output as user representation.
        """
        if self.mode == "item":
            return None
        # Get sequence embedding
        seq_embed = self.item_emb(x, self.features[:1])[:, 0]   # Only use seq feature   # (batch_size, seq_len, embed_dim)
        seq_output = self.seq_forward(x, seq_embed)             # (batch_size, max_len, embed_dim)

        # Get the last valid position for each sequence
        seq = x['seq']                                          # (batch_size, seq_len)
        seq_lens = (seq != 0).sum(dim=1) - 1                    # (batch_size,)
        seq_lens = seq_lens.clamp(min=0)                        # (batch_size,)
        batch_idx = torch.arange(seq_output.size(0), device=seq_output.device)
        user_emb = seq_output[batch_idx, seq_lens]  # [batch_size, embed_dim]

        if self.mode == "user":
            return user_emb
        return user_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]

    def item_tower(self, x):
        """Compute item embedding for in-batch negative sampling."""
        if self.mode == "user":
            return None
        if self.item_feature is not None:
            item_ids = x[self.item_feature.name]
            # Use the embedding layer to get item embeddings
            item_emb = self.item_emb.embedding[self.features[0].name](item_ids)
            if self.mode == "item":
                return item_emb
            return item_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]
        return None

    def forward(self, x):
        # Support inference mode
        if self.mode == "user":
            return self.user_tower(x)
        if self.mode == "item":
            return self.item_tower(x)

        # In-batch negative sampling mode
        if self.item_feature is not None:
            user_emb = self.user_tower(x)  # [batch_size, 1, embed_dim]
            item_emb = self.item_tower(x)  # [batch_size, 1, embed_dim]
            return torch.mul(user_emb, item_emb).sum(dim=-1).squeeze()

        # Original behavior: pairwise loss with pos/neg sequences
        embedding = self.item_emb(x, self.features)                                             # (batch_size, 3, seq_len, embed_dim)
        seq_embed, pos_embed, neg_embed = embedding[:, 0], embedding[:, 1], embedding[:, 2]     # (batch_size, seq_len, embed_dim)
        seq_output = self.seq_forward(x, seq_embed)                                             # (batch_size, seq_len, embed_dim)

        pos_logits = (seq_output * pos_embed).sum(dim=-1)
        neg_logits = (seq_output * neg_embed).sum(dim=-1)

        return pos_logits, neg_logits


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


if __name__ == '__main__':
    seq = SequenceFeature('seq', vocab_size=17, embed_dim=7, pooling='concat')
    pos = SequenceFeature('pos', vocab_size=17, embed_dim=7, pooling='concat', shared_with='seq')
    neg = SequenceFeature('neg', vocab_size=17, embed_dim=7, pooling='concat', shared_with='seq')

    seq = [seq, pos, neg]

    hist_seq = torch.tensor([[1, 2, 3, 4], [2, 3, 7, 8]])   # (batch_size, seq_len), (2, 4)
    pos_seq = hist_seq
    neg_seq = hist_seq

    x = {'seq': hist_seq, 'pos': pos_seq, 'neg': neg_seq}
    model = SASRec(features=seq)
    print('out', model(x))
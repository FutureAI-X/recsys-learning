import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, embedding_dim]
        # Conv1d expects [batch, channels, length], so we transpose
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs  # Residual connection
        return outputs


class SASRec(nn.Module):
    def __init__(self, num_items, max_len, embedding_dim, num_blocks, num_heads, dropout_rate, device):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.device = device
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # 1. 物品嵌入层 (包括一个额外的填充索引 0，通常索引 0 用于 padding)
        # 假设物品索引从 1 开始，0 是 padding
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # 2. 位置嵌入层
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # 3. 层归一化
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.ln_2 = nn.LayerNorm(embedding_dim)

        # 4. 自注意力层 (使用 PyTorch 内置的 MultiheadAttention)
        # 注意：PyTorch 的 MultiheadAttention 默认输入形状为 [seq_len, batch, hidden]
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
            for _ in range(num_blocks)
        ])

        # 5. 前馈神经网络层
        self.feed_forward_layers = nn.ModuleList([
            PointWiseFeedForward(embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.final_ln = nn.LayerNorm(embedding_dim)

    def generate_causal_mask(self, sequence_length):
        # 生成下三角矩阵作为掩码，防止关注未来信息
        # mask[i, j] = 0 if i >= j (可以看见), -inf if i < j (看不见)
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, log_seqs):
        """
        Args:
            log_seqs: (batch_size, seq_len)
        Returns:
            (batch_size, seq_len, num_items)
        """
        batch_size, seq_len = log_seqs.shape

        # 获取嵌入
        embeddings = self.item_emb(log_seqs)  # (batch_size, seq_len, embedding_dim)

        # 添加位置编码
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)   # (batch_size, seq_len)
        embeddings += self.pos_emb(positions)       # (batch_size, seq_len, embedding_dim)
        embeddings = self.emb_dropout(embeddings)

        # 生成因果掩码 (适用于整个批次，因为序列长度相同或通过 padding 对齐)
        # 如果序列长度动态变化，需要在每个样本上单独处理或动态生成
        causal_mask = self.generate_causal_mask(seq_len)

        # 通过多个注意力块和前馈网络
        outputs = embeddings
        for attention_layer, ff_layer in zip(self.attention_layers, self.feed_forward_layers):
            # 自注意力
            # key_padding_mask 用于处理 padding (假设 0 是 padding)
            key_padding_mask_bool = (log_seqs == 0)
            key_padding_mask = key_padding_mask_bool.to(causal_mask.dtype)
            key_padding_mask = key_padding_mask.masked_fill(key_padding_mask_bool, float('-inf'))

            attn_outputs, _ = attention_layer(
                query=outputs,
                key=outputs,
                value=outputs,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            outputs = self.ln_1(outputs + attn_outputs)

            # 前馈网络
            ff_outputs = ff_layer(outputs)
            outputs = self.ln_2(ff_outputs)

        outputs = self.final_ln(outputs)
        return outputs


# --- 使用示例 ---
if __name__ == "__main__":
    # 超参数配置
    num_items = 5000  # 物品总数
    max_len = 50  # 最大序列长度
    embedding_dim = 64  # 隐藏层维度
    num_blocks = 2  # Transformer 块数量
    num_heads = 4  # 注意力头数量
    dropout_rate = 0.2
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = SASRec(num_items, max_len, embedding_dim, num_blocks, num_heads, dropout_rate, device)
    model.to(device)
    model.train()

    # 模拟输入数据 (随机生成物品 ID，0 代表 padding), shape: (batch_size, max_len)
    # 实际应用中，这里应该是截断或填充后的用户历史序列
    input_seq = torch.randint(1, num_items + 1, (batch_size, max_len)).to(device)
    # 随机将一些位置设为 0 模拟 padding
    mask_rand = torch.rand_like(input_seq.float()) < 0.1
    input_seq[mask_rand] = 0

    # 前向传播, output: (batch_size, max_len, num_items)
    output_seq_emb = model(input_seq)

    print(f"输入形状: {input_seq.shape}")
    print(f"输出形状: {output_seq_emb.shape}")
    # 输出形状应为 [batch_size, seq_len, embedding_dim]
    # 在训练时，通常取输出序列的最后一个有效位置（或所有位置）与下一个真实物品计算损失
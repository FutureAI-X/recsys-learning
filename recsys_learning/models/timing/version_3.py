"""
diffusion_optimized.py
优化版 Diffusion - 提升模型效果
改进点：
1. 改进的扩散过程（吸收态+掩码）
2. 更强的去噪网络（Transformer + Attention）
3. 优化采样策略（DDIM加速 + 温度退火）
4. 更好的训练策略（梯度累积 + EMA）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import math
import random
from collections import deque
import copy


# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ========== 配置 ==========
SLOT_DIM = 288
SEQ_LEN = 10
HIDDEN_DIM = 128  # 增大隐藏维度
NUM_LAYERS = 2
NUM_HEADS = 4
DIFFUSION_STEPS = 50  # 增加扩散步数
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
LR_MIN = 1e-5
WARMUP_STEPS = 500
SEED = 42
USE_EMA = True
EMA_DECAY = 0.999


def slot_to_time(slot):
    hour = slot // 12
    minute = (slot % 12) * 5
    return f"{hour:02d}:{minute:02d}"


def generate_sample_data(num_users=500, min_clicks=30, max_clicks=60, seed=42):
    """生成更真实的用户数据，增加时序模式"""
    np.random.seed(seed)
    data = []
    for user_id in range(num_users):
        # 更丰富的用户类型
        user_type = np.random.choice(['morning', 'evening', 'bimodal', 'night', 'continuous'])
        num_clicks = np.random.randint(min_clicks, max_clicks)
        slots = []

        last_slot = None
        for _ in range(num_clicks):
            if user_type == 'morning':
                hour = int(np.clip(np.random.normal(8.5, 1.5), 6, 11))
                slot = hour * 12 + np.random.randint(0, 12)
            elif user_type == 'evening':
                hour = int(np.clip(np.random.normal(20, 1.5), 17, 23))
                slot = hour * 12 + np.random.randint(0, 12)
            elif user_type == 'night':
                hour = int(np.clip(np.random.normal(1, 1.5), 0, 5))
                slot = hour * 12 + np.random.randint(0, 12)
            elif user_type == 'continuous':
                # 连续浏览模式
                if last_slot is None:
                    hour = np.random.randint(0, 24)
                else:
                    hour = last_slot // 12 + np.random.randint(-2, 3)
                    hour = np.clip(hour, 0, 23)
                slot = hour * 12 + np.random.randint(0, 12)
            else:  # bimodal
                if np.random.random() < 0.6:
                    hour = int(np.clip(np.random.normal(8.5, 1.5), 6, 11))
                else:
                    hour = int(np.clip(np.random.normal(20, 1.5), 17, 23))
                slot = hour * 12 + np.random.randint(0, 12)

            slots.append(slot)
            last_slot = slot
        data.append(slots)
    return data


def prepare_train_data(user_sequences, seq_len=SEQ_LEN):
    X, y = [], []
    for seq in user_sequences:
        for i in range(len(seq) - seq_len):
            X.append(seq[i:i+seq_len])
            y.append(seq[i+seq_len])
    return X, y


# ========== 位置编码 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ========== 改进的编码器（Transformer） ==========
class ImprovedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(SLOT_DIM, HIDDEN_DIM)
        self.pos_encoding = PositionalEncoding(HIDDEN_DIM)

        # 使用Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embed(x)  # (batch, seq_len, hidden)
        emb = self.pos_encoding(emb)

        # Transformer需要 (seq_len, batch, hidden)
        emb = emb.transpose(0, 1)
        out = self.transformer(emb)
        out = out.transpose(0, 1)

        # 取最后一个位置的输出作为条件
        condition = out[:, -1, :]
        return self.norm(condition)


# ========== 改进的去噪网络（带Attention） ==========
class ImprovedDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embed = nn.Embedding(SLOT_DIM, HIDDEN_DIM)
        self.time_embed = nn.Sequential(
            nn.Embedding(DIFFUSION_STEPS, HIDDEN_DIM // 2),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )

        # Cross-attention机制
        self.cross_attn = nn.MultiheadAttention(HIDDEN_DIM, NUM_HEADS, dropout=0.1, batch_first=True)

        # 主网络
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM * 2),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(HIDDEN_DIM, SLOT_DIM)
        )

    def forward(self, x_class, condition, t):
        # x_class: (batch,)
        # condition: (batch, hidden)
        # t: (batch,)

        x_emb = self.class_embed(x_class).unsqueeze(1)  # (batch, 1, hidden)
        t_emb = self.time_embed(t).unsqueeze(1)  # (batch, 1, hidden)
        condition = condition.unsqueeze(1)  # (batch, 1, hidden)

        # Cross-attention
        attn_out, _ = self.cross_attn(x_emb, condition, condition)

        # 融合
        h = torch.cat([attn_out.squeeze(1), t_emb.squeeze(1)], dim=-1)
        return self.net(h)


# ========== EMA（指数移动平均） ==========
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ========== 改进的扩散过程 ==========
class ImprovedDiffusion:
    def __init__(self, seed=SEED):
        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = ImprovedEncoder().to(self.device)
        self.denoiser = ImprovedDenoiser().to(self.device)

        if USE_EMA:
            self.ema = EMA(self.denoiser, EMA_DECAY)

        # 改进的噪声调度（余弦调度）
        self._build_noise_schedule()

        print(f"ImprovedDiffusion 初始化，设备: {self.device}, 种子: {seed}")
        print(f"参数数量: Encoder: {sum(p.numel() for p in self.encoder.parameters()):,}, "
              f"Denoiser: {sum(p.numel() for p in self.denoiser.parameters()):,}")

    def _build_noise_schedule(self):
        """余弦噪声调度"""
        betas = []
        for i in range(DIFFUSION_STEPS):
            t1 = i / DIFFUSION_STEPS
            t2 = (i + 1) / DIFFUSION_STEPS
            beta = min(0.999, 1 - (math.cos(t2 * math.pi / 2) / math.cos(t1 * math.pi / 2)) ** 2)
            betas.append(beta)

        self.betas = torch.tensor(betas, device=self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _q_sample(self, x_start, t, noise=None):
        """前向加噪过程"""
        if noise is None:
            noise = torch.randint(0, SLOT_DIM, x_start.shape, device=self.device)

        alpha_bar = self.alpha_bars[t].unsqueeze(1)

        # 使用α_bar控制保留原始信息的比例
        retain_prob = alpha_bar
        mask = torch.rand(x_start.shape, device=self.device) < retain_prob
        x_noisy = torch.where(mask, x_start, noise)

        return x_noisy

    def fit(self, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
            grad_accum_steps=2):
        """训练循环"""
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.denoiser.parameters()),
            lr=LR, weight_decay=0.01
        )

        # 余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=LR_MIN
        )

        X_tensor = torch.LongTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        n_samples = len(X_train)

        print(f"\n开始训练，样本数: {n_samples:,}, 批次大小: {batch_size}")
        print(f"梯度累积步数: {grad_accum_steps}")

        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            idx = torch.randperm(n_samples)
            total_loss = 0
            total_acc = 0
            num_batches = 0

            optimizer.zero_grad()

            for i in range(0, n_samples, batch_size):
                batch_idx = idx[i:i+batch_size]
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]
                batch_size_actual = len(batch_idx)

                # 编码历史序列
                condition = self.encoder(batch_X)

                # 采样时间步
                t = torch.randint(1, DIFFUSION_STEPS, (batch_size_actual,), device=self.device)

                # 加噪
                noise = torch.randint(0, SLOT_DIM, (batch_size_actual,), device=self.device)
                x_noisy = self._q_sample(batch_y, t, noise)

                # 预测
                pred_logits = self.denoiser(x_noisy, condition, t)
                loss = F.cross_entropy(pred_logits, batch_y)

                # 梯度累积
                loss = loss / grad_accum_steps
                loss.backward()

                total_loss += loss.item() * grad_accum_steps
                total_acc += (pred_logits.argmax(1) == batch_y).float().mean().item()
                num_batches += 1

                if (i + batch_size) % (batch_size * grad_accum_steps) == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.denoiser.parameters()),
                        1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    if USE_EMA:
                        self.ema.update()

            scheduler.step()

            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_denoiser = copy.deepcopy(self.denoiser.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 epoch {epoch}")
                    break

            if epoch % 5 == 0 or epoch == epochs - 1:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | LR: {lr:.6f}")

        # 恢复最佳模型
        if hasattr(self, 'best_denoiser'):
            self.denoiser.load_state_dict(self.best_denoiser)

        print("训练完成")

    @torch.no_grad()
    def predict(self, history_slots, top_k=5, num_steps=None, temperature=0.8, use_ema=True):
        """
        改进的采样策略
        - DDIM加速采样
        - 温度退火
        """
        if use_ema and USE_EMA:
            self.ema.apply_shadow()

        self.encoder.eval()
        self.denoiser.eval()

        # 编码历史
        x = torch.LongTensor([history_slots]).to(self.device)
        condition = self.encoder(x)

        # DDIM采样（使用更少的步数）
        if num_steps is None:
            num_steps = min(20, DIFFUSION_STEPS // 2)

        step_indices = torch.linspace(0, DIFFUSION_STEPS - 1, num_steps, dtype=torch.long)

        # 从纯噪声开始
        x_t = torch.randint(0, SLOT_DIM, (1,), device=self.device)

        for step in reversed(range(len(step_indices))):
            t = step_indices[step]
            t_tensor = torch.tensor([t], device=self.device)

            # 预测
            logits = self.denoiser(x_t, condition, t_tensor)

            # 温度缩放
            logits = logits / temperature

            # 采样
            probs = F.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.squeeze(0), 1)

            # 温度退火
            temperature = max(0.5, temperature * 0.95)

        # 最终预测
        t0 = torch.tensor([0], device=self.device)
        final_logits = self.denoiser(x_t, condition, t0)
        final_probs = F.softmax(final_logits / 0.8, dim=-1).squeeze().cpu().numpy()

        if use_ema and USE_EMA:
            self.ema.restore()

        top_idx = np.argsort(final_probs)[-top_k:][::-1]
        return [(int(i), float(final_probs[i])) for i in top_idx]

    def predict_ensemble(self, history_slots, top_k=5, num_samples=5):
        """集成预测"""
        all_probs = []

        for _ in range(num_samples):
            preds = self.predict(history_slots, top_k=SLOT_DIM, temperature=0.8)
            probs = np.zeros(SLOT_DIM)
            for slot, prob in preds:
                probs[slot] = prob
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        top_idx = np.argsort(avg_probs)[-top_k:][::-1]
        return [(int(i), float(avg_probs[i])) for i in top_idx]


# ========== 评估函数 ==========
def evaluate_improved_diffusion(model, X_test, y_test):
    """评估模型"""
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0

    for i, (history, true) in enumerate(zip(X_test, y_test)):
        preds = model.predict(history, top_k=5, temperature=0.7)

        # Top-1准确率（宽松匹配，允许±2个slot）
        if abs(preds[0][0] - true) <= 2 or abs(preds[0][0] - true) >= 286:
            correct_top1 += 1

        # Top-3准确率
        for p, _ in preds[:3]:
            if abs(p - true) <= 2 or abs(p - true) >= 286:
                correct_top3 += 1
                break

        # Top-5准确率
        for p, _ in preds[:5]:
            if abs(p - true) <= 2 or abs(p - true) >= 286:
                correct_top5 += 1
                break

        if (i + 1) % 500 == 0:
            print(f"  评估进度: {i+1}/{len(X_test)}")

    n = len(y_test)
    print(f"  Improved Diffusion Top1: {correct_top1/n:.2%}")
    print(f"  Improved Diffusion Top3: {correct_top3/n:.2%}")
    print(f"  Improved Diffusion Top5: {correct_top5/n:.2%}")

    return correct_top1 / n


# ========== 主程序 ==========
def main():
    # 生成数据
    print("生成训练数据...")
    user_sequences = generate_sample_data(num_users=500, seed=SEED)
    X, y = prepare_train_data(user_sequences)

    # 划分数据集
    idx = np.random.permutation(len(X))
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:split+500], y[split:split+500]

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 训练优化版Diffusion
    print("\n" + "="*60)
    print("训练 Improved Diffusion")
    print("="*60)

    model = ImprovedDiffusion(seed=SEED)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 评估
    print("\n评估模型...")
    evaluate_improved_diffusion(model, X_test, y_test)

    # 展示预测示例
    print("\n" + "="*60)
    print("预测示例")
    print("="*60)
    for i in range(5):
        history = X_test[i][-5:]  # 显示最后5个历史
        true = y_test[i]
        preds = model.predict(history, top_k=3)

        hist_str = " -> ".join([slot_to_time(s) for s in history[-3:]])
        print(f"历史: [{hist_str}]")
        print(f"真实: {slot_to_time(true)}")
        print(f"预测: {', '.join([f'{slot_to_time(s)} ({p:.2%})' for s, p in preds])}")
        print()


if __name__ == "__main__":
    main()
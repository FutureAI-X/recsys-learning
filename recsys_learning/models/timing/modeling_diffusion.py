import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import set_seed

# ========== 配置 ==========
SLOT_DIM = 288
SEQ_LEN = 10
HIDDEN_DIM = 48  # 稍微增大
DIFFUSION_STEPS = 30  # 增加步数
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
SEED = 42

# ========== 稳定版编码器 ==========
class StableEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(SLOT_DIM, 24)
        self.gru = nn.GRU(24, HIDDEN_DIM, batch_first=True, num_layers=1)
        self.norm = nn.LayerNorm(HIDDEN_DIM)  # 添加归一化

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.gru(emb)
        return self.norm(h.squeeze(0))

# ========== 稳定版去噪网络 ==========
class StableDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embed = nn.Embedding(SLOT_DIM, HIDDEN_DIM)
        self.time_embed = nn.Embedding(DIFFUSION_STEPS, HIDDEN_DIM)

        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM * 2),
            nn.LayerNorm(HIDDEN_DIM * 2),  # 添加 LayerNorm
            nn.ReLU(),
            nn.Dropout(0.05),  # 轻微 Dropout
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, SLOT_DIM)
        )

    def forward(self, x_class, condition, t):
        x_emb = self.class_embed(x_class)
        t_emb = self.time_embed(t)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        return self.net(h)

# ========== 稳定版 Diffusion ==========
class StableDiffusion:
    def __init__(self, seed=42):
        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = StableEncoder().to(self.device)
        self.denoiser = StableDenoiser().to(self.device)

        x = torch.linspace(0, DIFFUSION_STEPS, DIFFUSION_STEPS)
        self.retain_probs = torch.cos((x / DIFFUSION_STEPS) * math.pi * 0.5) ** 2
        self.retain_probs = self.retain_probs.to(self.device)

        # 预计算转移矩阵（加速）
        self._precompute_transitions()

        print(f"StableDiffusion 初始化，设备: {self.device}, 种子: {seed}")

    def _precompute_transitions(self):
        """预计算每个时间步的转移概率分布"""
        self.transition_dist = []
        for t in range(DIFFUSION_STEPS):
            retain = self.retain_probs[t].item()
            uniform = (1 - retain) / SLOT_DIM
            dist = torch.full((SLOT_DIM,), uniform, device=self.device)
            self.transition_dist.append(dist)

    def _q_sample_fast(self, x_start, t):
        """快速加噪（向量化）"""
        batch_size = x_start.shape[0]

        # 生成随机数
        rand_vals = torch.rand(batch_size, device=self.device)

        # 初始化
        x_noisy = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            ti = t[i].item()
            if rand_vals[i] < self.retain_probs[ti]:
                x_noisy[i] = x_start[i]
            else:
                x_noisy[i] = torch.randint(0, SLOT_DIM, (1,), device=self.device).item()

        return x_noisy

    def fit(self, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.denoiser.parameters()),
            lr=LR, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        X_tensor = torch.LongTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        n_samples = len(X_train)

        print(f"开始训练，样本数: {n_samples}")

        for epoch in range(epochs):
            idx = torch.randperm(n_samples)
            total_loss = 0
            total_acc = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = idx[i:i+batch_size]
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]
                batch_size_actual = len(batch_idx)

                condition = self.encoder(batch_X)
                t = torch.randint(1, DIFFUSION_STEPS, (batch_size_actual,), device=self.device)
                x_noisy = self._q_sample_fast(batch_y, t)

                pred_logits = self.denoiser(x_noisy, condition, t)
                loss = F.cross_entropy(pred_logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.denoiser.parameters()), 1.0
                )
                optimizer.step()

                total_loss += loss.item()
                total_acc += (pred_logits.argmax(1) == batch_y).float().mean().item()

            scheduler.step()

            if epoch % 2 == 0 or epoch == epochs - 1:
                avg_loss = total_loss * batch_size / n_samples
                avg_acc = total_acc * batch_size / n_samples
                print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

        print("训练完成")

    @torch.no_grad()
    def predict(self, history_slots, top_k=3, num_samples=10, temperature=0.8):
        """
        稳定预测：
        - 增加采样次数
        - 使用温度缩放
        """
        self.encoder.eval()
        self.denoiser.eval()

        x = torch.LongTensor([history_slots]).to(self.device)
        condition = self.encoder(x)

        all_probs = []

        for _ in range(num_samples):
            x_class = torch.randint(0, SLOT_DIM, (1,), device=self.device)

            for step in reversed(range(1, DIFFUSION_STEPS)):
                t = torch.tensor([step], device=self.device)
                pred_logits = self.denoiser(x_class, condition, t)
                # 温度缩放
                pred_probs = F.softmax(pred_logits / temperature, dim=-1)
                x_class = torch.multinomial(pred_probs, 1).squeeze(-1)

            t0 = torch.tensor([0], device=self.device)
            final_logits = self.denoiser(x_class, condition, t0)
            probs = F.softmax(final_logits / temperature, dim=-1).squeeze().cpu().numpy()
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        top_idx = np.argsort(avg_probs)[-top_k:][::-1]
        return [(int(i), float(avg_probs[i])) for i in top_idx]

    def predict_ensemble(self, history_slots, top_k=3, num_runs=3):
        """集成预测：跑多次取平均"""
        all_preds = []
        for _ in range(num_runs):
            preds = self.predict(history_slots, top_k=top_k)
            all_preds.append(preds)

        # 合并结果
        slot_scores = {}
        for preds in all_preds:
            for slot, prob in preds:
                slot_scores[slot] = slot_scores.get(slot, 0) + prob

        sorted_slots = sorted(slot_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_slots[:top_k]

def evaluate_stable_diffusion(model, X_test, y_test):
    correct_top1 = 0
    correct_top3 = 0

    for i, (history, true) in enumerate(zip(X_test, y_test)):
        preds = model.predict(history, top_k=3, num_samples=15, temperature=0.7)

        if abs(preds[0][0] - true) <= 2 or abs(preds[0][0] - true) >= 286:
            correct_top1 += 1

        for p, _ in preds[:3]:
            if abs(p - true) <= 2 or abs(p - true) >= 286:
                correct_top3 += 1
                break

        if (i + 1) % 100 == 0:
            print(f"  评估进度: {i+1}/{len(X_test)}")

    n = len(y_test)
    print(f"  Stable Diffusion Top1: {correct_top1/n:.2%}, Top3: {correct_top3/n:.2%}")
    return correct_top1 / n

def train_diffusion(X_train, y_train, X_test, y_test, seed):
    model = StableDiffusion(seed=seed)
    model.fit(X_train, y_train, epochs=20)
    diff_acc = evaluate_stable_diffusion(model, X_test, y_test)
    return diff_acc
"""
diffusion_stable.py
稳定版 Diffusion - 通过固定种子和增加集成提高稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import math
import random


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
HIDDEN_DIM = 48  # 稍微增大
DIFFUSION_STEPS = 30  # 增加步数
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
SEED = 42


def slot_to_time(slot):
    hour = slot // 12
    minute = (slot % 12) * 5
    return f"{hour:02d}:{minute:02d}"


def generate_sample_data(num_users=300, min_clicks=30, max_clicks=60, seed=42):
    np.random.seed(seed)
    data = []
    for user_id in range(num_users):
        user_type = np.random.choice(['morning', 'evening', 'bimodal'])
        num_clicks = np.random.randint(min_clicks, max_clicks)
        slots = []

        for _ in range(num_clicks):
            if user_type == 'morning':
                hour = int(np.clip(np.random.normal(8.5, 1.5), 6, 11))
            elif user_type == 'evening':
                hour = int(np.clip(np.random.normal(20, 1.5), 17, 23))
            else:
                if np.random.random() < 0.6:
                    hour = int(np.clip(np.random.normal(8.5, 1.5), 6, 11))
                else:
                    hour = int(np.clip(np.random.normal(20, 1.5), 17, 23))

            slot = hour * 12 + np.random.randint(0, 12)
            slots.append(slot)
        data.append(slots)
    return data


def prepare_train_data(user_sequences, seq_len=SEQ_LEN):
    X, y = [], []
    for seq in user_sequences:
        for i in range(len(seq) - seq_len):
            X.append(seq[i:i+seq_len])
            y.append(seq[i+seq_len])
    return X, y


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
    def __init__(self, seed=SEED):
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


# ========== 稳定版 LSTM ==========
class StableLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(SLOT_DIM, 24)
        self.lstm = nn.LSTM(24, HIDDEN_DIM, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, SLOT_DIM)

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = self.norm(h.squeeze(0))
        return self.fc(h)


def train_stable_lstm(X_train, y_train, X_test, y_test, seed=SEED):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StableLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)

    X_tensor = torch.LongTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    n_samples = len(X_train)

    print("\n训练 Stable LSTM...")

    for epoch in range(20):
        idx = torch.randperm(n_samples)
        total_loss = 0
        for i in range(0, n_samples, 128):
            batch_idx = idx[i:i+128]
            batch_X = X_tensor[batch_idx]
            batch_y = y_tensor[batch_idx]

            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 3 == 0:
            print(f"  Epoch {epoch:2d} | Loss: {total_loss*128/n_samples:.4f}")

    model.eval()
    correct = 0
    correct_top3 = 0
    with torch.no_grad():
        for history, true in zip(X_test, y_test):
            x = torch.LongTensor([history]).to(device)
            logits = model(x)
            probs = F.softmax(logits, -1).squeeze().cpu().numpy()

            pred = np.argmax(probs)
            if abs(pred - true) <= 2 or abs(pred - true) >= 286:
                correct += 1

            top3 = np.argsort(probs)[-3:][::-1]
            for p in top3:
                if abs(p - true) <= 2 or abs(p - true) >= 286:
                    correct_top3 += 1
                    break

    n = len(y_test)
    print(f"  Stable LSTM Top1: {correct/n:.2%}, Top3: {correct_top3/n:.2%}")
    return correct / n


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


# ========== 多次运行取平均 ==========
def run_multiple_times(num_runs=3):
    results = {'lstm': [], 'diffusion': []}

    for run in range(num_runs):
        print(f"\n{'='*50}")
        print(f"第 {run+1}/{num_runs} 次运行")
        print('='*50)

        seed = 42 + run * 10
        set_seed(seed)

        user_sequences = generate_sample_data(num_users=200, seed=seed)
        X, y = prepare_train_data(user_sequences)

        idx = np.random.permutation(len(X))
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:split+300], y[split:split+300]

        # LSTM
        lstm_acc = train_stable_lstm(X_train, y_train, X_test, y_test, seed=seed)
        results['lstm'].append(lstm_acc)

        # Diffusion
        print("\n训练 Stable Diffusion...")
        model = StableDiffusion(seed=seed)
        model.fit(X_train, y_train, epochs=20)
        diff_acc = evaluate_stable_diffusion(model, X_test, y_test)
        results['diffusion'].append(diff_acc)

    # 统计结果
    print("\n" + "=" * 60)
    print("多次运行统计结果")
    print("=" * 60)
    print(f"LSTM:      {np.mean(results['lstm']):.2%} ± {np.std(results['lstm']):.2%}")
    print(f"Diffusion: {np.mean(results['diffusion']):.2%} ± {np.std(results['diffusion']):.2%}")
    print(f"各次结果 LSTM: {[f'{x:.2%}' for x in results['lstm']]}")
    print(f"各次结果 Diff: {[f'{x:.2%}' for x in results['diffusion']]}")
    print("=" * 60)


# ========== 主程序 ==========
def main():
    run_multiple_times(num_runs=1)


if __name__ == "__main__":
    main()
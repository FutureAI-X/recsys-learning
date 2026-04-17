"""
diffusion_fast.py
极致加速版 - 专为低配置设备优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import math


# ========== 极简配置 ==========
SLOT_DIM = 288
SEQ_LEN = 10           # 减少序列长度
HIDDEN_DIM = 32        # 大幅减少隐藏层
DIFFUSION_STEPS = 10   # 只用10步
BATCH_SIZE = 128       # 增大批次
EPOCHS = 20            # 减少轮数
LR = 0.002


def slot_to_time(slot):
    hour = slot // 12
    minute = (slot % 12) * 5
    return f"{hour:02d}:{minute:02d}"


def generate_sample_data(num_users=300, min_clicks=30, max_clicks=60):
    """减少数据量"""
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


# ========== 极简编码器 ==========
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(SLOT_DIM, 16)
        self.gru = nn.GRU(16, HIDDEN_DIM, batch_first=True, num_layers=1)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.gru(emb)
        return h.squeeze(0)


# ========== 极简去噪网络 ==========
class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embed = nn.Embedding(SLOT_DIM, HIDDEN_DIM)
        self.time_embed = nn.Embedding(DIFFUSION_STEPS, HIDDEN_DIM)

        # 极简网络：只有两层
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, SLOT_DIM)
        )

    def forward(self, x_class, condition, t):
        x_emb = self.class_embed(x_class)
        t_emb = self.time_embed(t)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        return self.net(h)


# ========== 极速 Diffusion ==========
class FastDiffusion:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = TinyEncoder().to(self.device)
        self.denoiser = TinyDenoiser().to(self.device)

        # 预计算保留概率
        x = torch.linspace(0, DIFFUSION_STEPS, DIFFUSION_STEPS)
        self.retain_probs = torch.cos((x / DIFFUSION_STEPS) * math.pi * 0.5) ** 2
        self.retain_probs = self.retain_probs.to(self.device)

        print(f"FastDiffusion 初始化，设备: {self.device}")
        print(f"  参数: SLOT={SLOT_DIM}, SEQ={SEQ_LEN}, HIDDEN={HIDDEN_DIM}, STEPS={DIFFUSION_STEPS}")

    def fit(self, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.denoiser.parameters()),
            lr=LR
        )

        # 转为 Tensor（一次性，避免重复转换）
        X_tensor = torch.LongTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

        n_samples = len(X_train)

        print(f"开始训练，样本数: {n_samples}")
        print(f"  每轮约 {n_samples//batch_size} 个 batch")

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

                # 快速加噪
                x_noisy = torch.zeros(batch_size_actual, dtype=torch.long, device=self.device)
                for j in range(batch_size_actual):
                    if torch.rand(1).item() < self.retain_probs[t[j]]:
                        x_noisy[j] = batch_y[j]
                    else:
                        x_noisy[j] = torch.randint(0, SLOT_DIM, (1,)).item()

                pred_logits = self.denoiser(x_noisy, condition, t)
                loss = F.cross_entropy(pred_logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += (pred_logits.argmax(1) == batch_y).float().mean().item()

            if epoch % 2 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:2d} | Loss: {total_loss*batch_size/n_samples:.4f} | Acc: {total_acc*batch_size/n_samples:.2%}")

        print("训练完成")

    @torch.no_grad()
    def predict(self, history_slots, top_k=3):
        self.encoder.eval()
        self.denoiser.eval()

        x = torch.LongTensor([history_slots]).to(self.device)
        condition = self.encoder(x)

        # 快速采样（只用3次）
        all_probs = []
        for _ in range(3):
            x_class = torch.randint(0, SLOT_DIM, (1,), device=self.device)

            for step in reversed(range(1, DIFFUSION_STEPS)):
                t = torch.tensor([step], device=self.device)
                pred_logits = self.denoiser(x_class, condition, t)
                x_class = torch.multinomial(F.softmax(pred_logits, -1), 1).squeeze(-1)

            t0 = torch.tensor([0], device=self.device)
            final_logits = self.denoiser(x_class, condition, t0)
            probs = F.softmax(final_logits, -1).squeeze().cpu().numpy()
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        top_idx = np.argsort(avg_probs)[-top_k:][::-1]
        return [(int(i), float(avg_probs[i])) for i in top_idx]


# ========== 极简 LSTM 对比 ==========
class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(SLOT_DIM, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, SLOT_DIM)

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(h.squeeze(0))


def train_tiny_lstm(X_train, y_train, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    X_tensor = torch.LongTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    n_samples = len(X_train)

    print("\n训练 Tiny LSTM...")

    for epoch in range(15):
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
    print(f"  Tiny LSTM Top1: {correct/n:.2%}, Top3: {correct_top3/n:.2%}")
    return correct / n


def evaluate_fast_diffusion(model, X_test, y_test):
    correct_top1 = 0
    correct_top3 = 0

    for i, (history, true) in enumerate(zip(X_test, y_test)):
        preds = model.predict(history, top_k=3)

        if abs(preds[0][0] - true) <= 2 or abs(preds[0][0] - true) >= 286:
            correct_top1 += 1

        for p, _ in preds[:3]:
            if abs(p - true) <= 2 or abs(p - true) >= 286:
                correct_top3 += 1
                break

        if (i + 1) % 100 == 0:
            print(f"  评估进度: {i+1}/{len(X_test)}")

    n = len(y_test)
    print(f"  Fast Diffusion Top1: {correct_top1/n:.2%}, Top3: {correct_top3/n:.2%}")
    return correct_top1 / n


# ========== 主程序 ==========
def main():
    print("=" * 50)
    print("极速 Diffusion vs Tiny LSTM")
    print("=" * 50)

    print("\n[1] 生成数据...")
    user_sequences = generate_sample_data(num_users=200)
    X, y = prepare_train_data(user_sequences)

    idx = np.random.permutation(len(X))
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:split+300], y[split:split+300]

    print(f"    训练: {len(X_train)}, 测试: {len(X_test)}")

    # Tiny LSTM
    lstm_acc = train_tiny_lstm(X_train, y_train, X_test, y_test)

    # Fast Diffusion
    print("\n[2] 训练 Fast Diffusion...")
    model = FastDiffusion()

    import time
    start = time.time()
    model.fit(X_train, y_train, epochs=15)
    print(f"  训练耗时: {time.time()-start:.1f} 秒")

    print("\n[3] 评估 Fast Diffusion...")
    diff_acc = evaluate_fast_diffusion(model, X_test, y_test)

    print("\n" + "=" * 50)
    print(f"Tiny LSTM:  {lstm_acc:.2%}")
    print(f"Fast Diff:  {diff_acc:.2%}")
    print("=" * 50)

    print("\n[4] 演示预测...")
    sample = X_test[0]
    true = y_test[0]
    preds = model.predict(sample)

    print(f"  历史: {[slot_to_time(s) for s in sample[-5:]]}")
    print(f"  真实: {slot_to_time(true)}")
    for i, (s, p) in enumerate(preds, 1):
        print(f"  {i}. {slot_to_time(s)} ({p:.2%})")


if __name__ == "__main__":
    main()
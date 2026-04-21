import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========== 稳定版 LSTM ==========
class StableLSTM(nn.Module):
    def __init__(self, hidden_dim, slot_dim):
        super().__init__()
        self.embed = nn.Embedding(slot_dim, 24)
        self.lstm = nn.LSTM(24, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, slot_dim)

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = self.norm(h.squeeze(0))
        return self.fc(h)

def train_stable_lstm(X_train, y_train, X_test, y_test, seed=42, hidden_dim=32, slot_dim=288):
    # set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StableLSTM(hidden_dim=hidden_dim, slot_dim=slot_dim).to(device)
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
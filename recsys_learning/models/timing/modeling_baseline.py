import numpy as np

# ========== 配置 ==========
SLOT_DIM = 288
SEQ_LEN = 10
HIDDEN_DIM = 48  # 稍微增大
DIFFUSION_STEPS = 30  # 增加步数
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
SEED = 42


# ========== 加权统计 Baseline（基于你的方案）==========
class WeightedSlotPredictor:
    """加权统计基线模型 - 基于最近历史加权 + 全局分布平滑"""

    def __init__(self, decay=0.7, global_weight=0.3):
        self.decay = decay  # 衰减因子，越近权重越高
        self.global_weight = global_weight  # 全局分布权重
        self.slot_dim = SLOT_DIM
        self.global_dist = None
        self.fitted = False

    def fit(self, X_train, y_train=None):
        """训练：统计全局分布"""
        # 收集所有槽位（包括X和y）
        all_slots = []
        for seq in X_train:
            all_slots.extend(seq)
        if y_train:
            all_slots.extend(y_train)

        self.global_dist = np.bincount(all_slots, minlength=self.slot_dim)
        self.global_dist = self.global_dist / self.global_dist.sum()
        self.fitted = True

        print(f"Baseline训练完成，总样本数: {len(all_slots)}")

    def predict(self, history_slots, top_k=3):
        """预测下一个槽位"""
        if not self.fitted:
            raise ValueError("模型未训练")

        if len(history_slots) == 0:
            # 返回全局最高频时段
            top_idx = np.argsort(self.global_dist)[-top_k:][::-1]
            return [(int(i), float(self.global_dist[i])) for i in top_idx]

        # 取最近10次
        recent = history_slots[-10:]

        # 计算权重（指数衰减）
        weights = [self.decay ** i for i in range(len(recent))][::-1]

        # 加权分布
        weighted_dist = np.zeros(self.slot_dim)
        for slot, weight in zip(recent, weights):
            weighted_dist[slot] += weight

        # 归一化
        if weighted_dist.sum() > 0:
            weighted_dist = weighted_dist / weighted_dist.sum()

        # 融合全局分布（平滑）
        final_dist = (1 - self.global_weight) * weighted_dist + self.global_weight * self.global_dist

        # Top-K
        top_indices = np.argsort(final_dist)[-top_k:][::-1]
        return [(int(i), float(final_dist[i])) for i in top_indices]


def evaluate_baseline(model, X_test, y_test):
    """评估 Baseline 模型"""
    correct_top1 = 0
    correct_top3 = 0

    for i, (history, true) in enumerate(zip(X_test, y_test)):
        preds = model.predict(history, top_k=3)

        # Top-1 准确率（考虑相邻时段）
        if abs(preds[0][0] - true) <= 2 or abs(preds[0][0] - true) >= 286:
            correct_top1 += 1

        # Top-3 准确率
        for p, _ in preds[:3]:
            if abs(p - true) <= 2 or abs(p - true) >= 286:
                correct_top3 += 1
                break

        if (i + 1) % 100 == 0:
            print(f"  评估进度: {i + 1}/{len(X_test)}")

    n = len(y_test)
    print(f"  Baseline Top1: {correct_top1 / n:.2%}, Top3: {correct_top3 / n:.2%}")
    return correct_top1 / n

def train_baseline(X_train, y_train, X_test, y_test):
    baseline = WeightedSlotPredictor(decay=0.7, global_weight=0.3)
    baseline.fit(X_train, y_train)
    return evaluate_baseline(baseline, X_test, y_test)
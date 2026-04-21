import random
import numpy as np
import torch


# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def slot_to_time(slot):
    hour = slot // 12
    minute = (slot % 12) * 5
    return f"{hour:02d}:{minute:02d}"
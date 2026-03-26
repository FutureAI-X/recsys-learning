import random
from torch.utils.data import Dataset, DataLoader
import torch

from recsys_learning.models.matching.sasrec.configuration_sasrec import SASRecConfig

def generate_simulated_data(num_users, num_items, max_len=200, min_len=5):
    """
    生成模拟的用户行为序列。
    返回: users (list), sequences (list of lists)
    注意：物品索引从 1 开始，0 保留给 padding
    """
    print("正在生成模拟数据...")
    users = []
    sequences = []
    
    for u in range(num_users):
        # 随机生成长度
        seq_len = random.randint(min_len, max_len)
        # 随机生成物品序列 (1 到 num_items)
        seq = [random.randint(1, num_items) for _ in range(seq_len)]
        users.append(u)
        sequences.append(seq)
        
    return users, sequences

class SASRecDataset(Dataset):
    def __init__(self, users, seqs, max_len):
        self.users = users
        self.seqs = seqs
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        input = seq[:-1]
        target = seq[1:]

        if len(input) > self.max_len:
            input = input[-self.max_len:]
            target = target[-self.max_len:]
        else:
            # 左侧填充 0 (Padding)
            padding_len = self.max_len - len(input)
            input = [0] * padding_len + input
            target = [0] * padding_len + target
        

        return torch.tensor(input, dtype=torch.long), \
               torch.tensor(target, dtype=torch.long), \
               torch.tensor([len(seq)-1], dtype=torch.long)
    

def get_dataloader_demo(config: SASRecConfig):
    users, seqs = generate_simulated_data(num_users=50, num_items=500, max_len=config.max_len)

    split_idx = int(len(users) * config.train_radio)

    train_users = users[:split_idx]
    test_users = users[split_idx:]

    train_seqs = seqs[:split_idx]
    test_seqs = seqs[split_idx:]
    train_dataset = SASRecDataset(users=train_users, seqs=train_seqs, max_len=config.max_len)
    test_dataset = SASRecDataset(test_users, test_seqs, config.max_len)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    return train_dataloader, test_dataloader
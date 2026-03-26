from recsys_learning.models.matching.sasrec.configuration_sasrec import SASRecConfig
from recsys_learning.models.matching.sasrec.modeling_sasrec import SASRecModel
from recsys_learning.datasets.dataset_sasrec_demo import get_dataloader_demo

import torch.optim as optim
import torch

import time
from tqdm import tqdm

def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking Loss
    Loss = -sum(log(sigmoid(pos_score - neg_score)))
    """
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-24))

def train_epoch(model: SASRecModel, dataloader, optimizer, scheduler, config: SASRecConfig):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, valid_lens in tqdm(dataloader):
        inputs = inputs.to(config.device)       # (B, L)
        targets = targets.to(config.device)     # (B, L)
        
        optimizer.zero_grad()

        seq_emb = model(inputs)                 # (B, L, H)

        B, L, H = seq_emb.shape
        seq_emb_flat = seq_emb.view(B * L, H)
        targets_flat = targets.view(B * L)

        mask = (targets != 0).unsqueeze(-1)     # (B, L, 1)
        mask_flat = mask.view(-1)

        valid_indices = torch.nonzero(mask_flat.squeeze(), as_tuple=True)[0]

        valid_seq_dim = seq_emb_flat[valid_indices]     # (N, H)
        valid_targets = targets_flat[valid_indices]     # (N)

        pos_item_embes = model.item_emb(valid_targets)  # (N, H)
        pos_scores = torch.sum(valid_seq_dim * pos_item_embes, dim=-1)   # (N)

        # 生成负样本
        neg_samples = torch.randint(1, config.num_items + 1, (valid_targets.shape[0], config.neg_sample_count), device=config.device)   # (N, K)
        neg_item_embes = model.item_emb(neg_samples)    # (N, K, H)
        neg_scores = torch.sum(valid_seq_dim.unsqueeze(1) * neg_item_embes, dim=-1) # (N, K)

        pos_scores = pos_scores.unsqueeze(1)    # (N, 1)
        loss = bpr_loss(pos_scores, neg_scores)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def evaluate(model, dataloader, config, num_items, k=10):
    model.eval()
    hits = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets, valid_lens in tqdm(dataloader):
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            # 取每个序列的最后一个有效目标作为测试项
            # 注意：在模拟数据中，targets 的最后一个非0元素就是我们要预测的
            # 为了简化，我们假设测试时预测序列的最后一个位置
            # 构造测试用的 target item
            batch_size = inputs.size(0)
            
            # 找到每个样本最后一个非零目标的索引
            # 这里简化：直接取序列最后一个位置对应的 target (如果是0则跳过)
            last_target = targets[:, -1] 
            valid_mask = (last_target != 0)
            
            if not valid_mask.any():
                continue
                
            test_inputs = inputs[valid_mask]
            test_targets = last_target[valid_mask]
            
            # 采样负物品用于评估 (1个正 + 99个负 = 100个候选)
            num_neg = 20
            negs = torch.randint(1, num_items + 1, (test_targets.shape[0], num_neg), device=config.device)
            
            # 构造候选集 [正, 负..., 负]
            candidates = torch.cat([test_targets.unsqueeze(1), negs], dim=1) # [B, 101]
            
            # 计算得分
            scores = model.predict(test_inputs, candidates) # [B, 101]
            
            # 获取排名
            # 降序排列，看正样本 (第0列) 是否在前 K 个
            _, ranks = torch.sort(scores, descending=True)
            
            # 更直观的方法：
            # 正样本的分数是 scores[:, 0]
            # 比正样本分数高的有多少个？
            pos_scores = scores[:, 0]
            greater_count = torch.sum(scores > pos_scores.unsqueeze(1), dim=1) # 比正样本分高的数量
            
            # 排名 = greater_count + 1
            ranks = greater_count + 1
            
            hits += torch.sum(ranks <= k).item()
            total += test_targets.shape[0]
            
    return hits / total if total > 0 else 0.0

def main():
    config = SASRecConfig()

    train_dataloader, test_dataloader = get_dataloader_demo(config=config)

    model = SASRecModel(
        num_items=config.num_items,
        max_len=config.max_len,
        embedding_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
        device=config.device
    )

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Start training...")

    for epoch in range(config.epochs):
        start_time = time.time()
        loss = train_epoch(model, train_dataloader, optimizer, scheduler, config)
        scheduler.step()
        end_time = time.time()

        hit_rate_10 = evaluate(model, test_dataloader, config, num_items=config.num_items, k=10)

        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {loss:.4f}, Hit@10: {hit_rate_10:.4f}, Time: {end_time - start_time:.2f}s")
        

if __name__=="__main__":
    main()
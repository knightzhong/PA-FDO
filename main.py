import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs import Config
from data_loader import build_paired_dataloader
from models import VectorFieldNet, RankProxy
from solver import ConditionalFlowMatching

def seed_everything(seed=42):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    cfg = Config()
    # print(f"Experiment: {cfg.TASK_NAME} | Device: {cfg.DEVICE}")
    
    # 1. 加载数据 (Triplets Dataloader)
    # unpack 的时候注意顺序
    train_loader, task, offline_x, offline_y, mean_x, std_x, mean_y, std_y = build_paired_dataloader(cfg)
    
    # 移动统计量到设备
    mean_x, std_x = mean_x.to(cfg.DEVICE), std_x.to(cfg.DEVICE)
    mean_y, std_y = mean_y.to(cfg.DEVICE), std_y.to(cfg.DEVICE)
    input_dim = offline_x.shape[1]
    
    # ==========================================
    # Part A: 训练 RankProxy (ListNet)
    # ==========================================
    # print("\nTraining RankProxy...")
    proxy = RankProxy(input_dim=input_dim).to(cfg.DEVICE)
    proxy_opt = torch.optim.AdamW(proxy.parameters(), lr=1e-4, weight_decay=1e-5)
    
    all_x = offline_x.to(cfg.DEVICE)
    all_y = offline_y.to(cfg.DEVICE).view(-1, 1)
    
    # ListNet 参数
    list_size = 512
    
    for epoch in range(2000): # 2000 epoch 很快
        proxy.train()
        proxy_opt.zero_grad()
        
        # 随机采样 List
        idx = torch.randperm(all_x.shape[0])[:list_size]
        x_batch = all_x[idx]
        y_batch = all_y[idx]
        
        y_pred = proxy(x_batch)
        
        # ListNet Loss
        # y_true 放大 10 倍增加区分度
        p_true = F.softmax(y_batch.t() * 10.0, dim=1)
        p_pred = F.log_softmax(y_pred.t(), dim=1)
        loss = -torch.sum(p_true * p_pred)
        
        loss.backward()
        proxy_opt.step()
        
        if (epoch+1) % 500 == 0:
            print(f"Proxy Epoch {epoch+1} | Loss: {loss.item():.4f}")

    # Output Adaptation
    proxy.eval()
    with torch.no_grad():
        all_preds = proxy(all_x)
        proxy_mu = all_preds.mean().item()
        proxy_std = all_preds.std().item()
    
    # Proxy Wrapper (归一化输出)
    class NormalizedProxy(nn.Module):
        def __init__(self, m, mu, std):
            super().__init__()
            self.model = m
            self.mu = mu
            self.std = std
        def forward(self, x):
            return (self.model(x) - self.mu) / (self.std + 1e-8)
            
    norm_proxy = NormalizedProxy(proxy, proxy_mu, proxy_std)
    
    # ==========================================
    # Part B: 训练 Flow Matching (PA-FDO)
    # ==========================================
    # print("\nTraining Flow Model...")
    net = VectorFieldNet(input_dim=input_dim, hidden_dim=cfg.LATENT_DIM)
    cfm = ConditionalFlowMatching(net, cfg.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    
    for epoch in range(cfg.EPOCHS):
        net.train()
        total_loss = 0
        
        # 注意：dataloader 返回 6 个 Tensor
        for x_anc, x_bet, x_wor, y_bet, y_wor, y_anc in train_loader:
            optimizer.zero_grad()
            # 传入 6 个参数计算 Loss
            loss = cfm.compute_loss(x_anc, x_bet, x_wor, y_bet, y_wor, y_anc)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # if (epoch + 1) % 20 == 0:
        #     print(f"Flow Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
            
    # ==========================================
    # Part C: 推理与评估 (分层采样)
    # ==========================================
    
    # 策略：从 50th - 90th 分位数区间采样
    # 目的：确保起始点本身就有一定质量，从而更容易优化到 100th
    y_flat = offline_y.view(-1)
    # y_sorted_indices = torch.argsort(y_flat)
    
    # 找到分位点
    q50 = torch.quantile(y_flat, 0.5)
    q90 = torch.quantile(y_flat, 0.9)
    
    # Mask: 选中间段的样本
    mask_start = (y_flat >= q50) & (y_flat <= q90)
    candidate_indices = torch.where(mask_start)[0]
    
    # 随机选 128 个
    perm = torch.randperm(len(candidate_indices))[:cfg.NUM_SAMPLES]
    selected_indices = candidate_indices[perm]
    
    x_starts = offline_x[selected_indices].to(cfg.DEVICE)
    y_starts = offline_y[selected_indices].view(-1, 1).to(cfg.DEVICE)
    
    # print(f"Sampling Starts Mean Score: {y_starts.mean().item():.4f} (Target Range: {q50:.2f}-{q90:.2f})")
    
    # 设定目标: 每个样本的目标是其当前分数的 1.2 倍，或者是全集最大值的 1.1 倍
    # 更加稳健的做法：统一设为 Max * 1.05
    y_max = offline_y.max().item()
    y_targets = torch.full_like(y_starts, y_max * 1.1)
    
    # 采样
    x_final = cfm.sample(
        x_starts, 
        y_target=y_targets, 
        y_start=y_starts,
        proxy=norm_proxy,
        steps=cfg.ODE_STEPS,
        cfg_scale=4.0,  # 强 CFG 引导
        grad_scale=4.0  # 强 Proxy 引导 (靠 Uncertainty 自动刹车)
    )
    
    # 反标准化
    x_denorm = x_final * std_x + mean_x
    
    # Oracle 评估
    if task.is_discrete:
        # 离散化处理
        vocab_size = 4
        seq_len = input_dim // vocab_size
        x_reshaped = x_denorm.view(x_denorm.shape[0], seq_len, vocab_size)
        x_indices = torch.argmax(x_reshaped, dim=2).cpu().numpy()
        scores = task.predict(x_indices)
    else:
        scores = task.predict(x_denorm.cpu().numpy())
        
    scores = scores.reshape(-1)
    
    # 归一化统计
    task_to_min = {'TFBind8-Exact-v0': 0.0}
    task_to_max = {'TFBind8-Exact-v0': 1.0}
    y_min_val = task_to_min.get(cfg.TASK_NAME, offline_y.min().item())
    y_max_val = task_to_max.get(cfg.TASK_NAME, offline_y.max().item())
    
    norm_scores = (scores - y_min_val) / (y_max_val - y_min_val)
    
    percentiles = np.percentile(norm_scores, [100, 80, 50])
    
    print("-" * 30)
    print(f"Result (Valid {len(scores)}): Mean {norm_scores.mean():.4f}")
    print(f"Percentiles (100/80/50): {percentiles[0]:.4f} | {percentiles[1]:.4f} | {percentiles[2]:.4f}")
    print("-" * 30)
    
    return percentiles

if __name__ == "__main__":
    all_res = []
    for seed in range(4): # 跑 4 个种子取平均
        print(f"=== Seed {seed} ===")
        seed_everything(seed)
        res = main()
        all_res.append(res)
    
    all_res = np.array(all_res)
    mu = all_res.mean(0)
    std = all_res.std(0)
    
    print("\n" + "="*40)
    print(f"FINAL RESULT (Avg over {len(all_res)} seeds)")
    print(f"100th: {mu[0]:.4f} ± {std[0]:.4f}")
    print(f" 80th: {mu[1]:.4f} ± {std[1]:.4f}")
    print(f" 50th: {mu[2]:.4f} ± {std[2]:.4f}")
    print("="*40)
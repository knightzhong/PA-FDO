import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs import Config
from data_loader import build_dynamic_loader # 注意这里函数名变了
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

def batch_optimal_transport_match(x_src, x_tgt):
    """
    Minibatch Optimal Transport (近似)
    在当前 batch 内重新排列 x_tgt，使得它与 x_src 的总距离最小。
    简单实现：贪心最近邻或者简单的排序匹配。
    这里使用余弦相似度最大化匹配 (如果是高维数据，余弦比欧氏更稳)。
    """
    # 归一化计算 Cosine Similarity
    x_src_norm = F.normalize(x_src, p=2, dim=1)
    x_tgt_norm = F.normalize(x_tgt, p=2, dim=1)
    
    # (B, B) 相似度矩阵
    sim_matrix = torch.mm(x_src_norm, x_tgt_norm.t())
    
    # 简单的贪心匹配: 每个 src 找最相似的 tgt
    # 注意：这可能导致多个 src 映射到同一个 tgt (多对一)
    # 为了保持多样性，最好是一对一。但为了训练效率，argmax 足矣，
    # 意味着我们只学习“最容易到达的那个高分点”
    best_indices = torch.argmax(sim_matrix, dim=1)
    
    return x_tgt[best_indices], best_indices

def main():
    cfg = Config()
    
    # 1. 加载数据 (动态池模式)
    task, ds_all, ds_gold, (mean_x, std_x, mean_y, std_y) = build_dynamic_loader(cfg)
    
    # 统计量上设备
    mean_x, std_x = mean_x.to(cfg.DEVICE), std_x.to(cfg.DEVICE)
    mean_y, std_y = mean_y.to(cfg.DEVICE), std_y.to(cfg.DEVICE)
    input_dim = ds_all.tensors[0].shape[1]
    
    # DataLoader (Random Samplers)
    loader_all = torch.utils.data.DataLoader(ds_all, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    loader_gold = torch.utils.data.DataLoader(ds_gold, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 用于无限循环的 iterator
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    iter_all = cycle(loader_all)
    iter_gold = cycle(loader_gold)
    
    # ==========================================
    # Part A: 训练 RankProxy (ListNet) - 保持不变
    # ==========================================
    print("\nTraining RankProxy...")
    # ... (此处省略 RankProxy 训练代码，与之前完全一致，请保留之前的 Proxy 训练逻辑) ...
    # 为了节省篇幅，我直接实例化并加载一个假设训练好的 Proxy
    # 在实际运行中，请保留你之前的 Proxy 训练代码块
    
    # --- 临时模拟 Proxy 训练代码块 (请替换回你的完整代码) ---
    proxy = RankProxy(input_dim=input_dim).to(cfg.DEVICE)
    proxy_opt = torch.optim.AdamW(proxy.parameters(), lr=1e-4)
    # 简单训练一下防止报错
    temp_x = ds_all.tensors[0][:100].to(cfg.DEVICE)
    for _ in range(10): 
        loss = proxy(temp_x).mean()
        loss.backward()
        proxy_opt.step()
    # ----------------------------------------------------

    # Proxy Wrapper
    proxy.eval()
    with torch.no_grad():
        # 用全量数据校准 mean/std
        all_x_gpu = ds_all.tensors[0].to(cfg.DEVICE)
        all_preds = proxy(all_x_gpu)
        proxy_mu = all_preds.mean().item()
        proxy_std = all_preds.std().item()

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
    # Part B: 训练 Flow Matching (PA-FDO 动态版)
    # ==========================================
    print("\nTraining Flow Model (PA-FDO Dynamic)...")
    net = VectorFieldNet(input_dim=input_dim, hidden_dim=cfg.LATENT_DIM)
    cfm = ConditionalFlowMatching(net, cfg.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    
    # 训练步数 (Iterations) 而非 Epochs
    total_steps = 5000 
    
    for step in range(total_steps):
        net.train()
        optimizer.zero_grad()
        
        # 1. 采样 Anchor (起点)
        x_anc, y_anc = next(iter_all)
        x_anc = x_anc.to(cfg.DEVICE)
        y_anc = y_anc.view(-1, 1).to(cfg.DEVICE)
        
        # 2. 采样 Candidates (潜在终点 - 真实高分数据)
        x_gold, y_gold = next(iter_gold)
        x_gold = x_gold.to(cfg.DEVICE)
        y_gold = y_gold.view(-1, 1).to(cfg.DEVICE)
        
        # 3. Minibatch OT 配对 (Manifold Matching)
        # 为每个 anchor 找到 batch 内最合适的 gold target
        # 这样构建的 (x_anc, x_better) 对是符合几何邻近性的
        with torch.no_grad():
            x_better, _ = batch_optimal_transport_match(x_anc, x_gold)
            y_better = torch.ones_like(y_anc) * y_gold.max() # 这里的 y_better 可以取实际值，也可以取 Target Max
            y_better = y_gold # 这里我们用实际匹配到的分值
        
        # 4. 生成自对抗负样本 (Self-Generated Worse)
        # 利用当前模型走一步，看看会去哪
        # 如果去的地方分低，它就是最好的 x_worse
        net.eval() # 采样时用 eval 模式 (关闭 Dropout)
        with torch.no_grad():
            # 试探步: t=0, 朝着 y_better 走
            # 使用 1-step Euler 预测
            v_initial = net(x_anc, torch.zeros(x_anc.shape[0], 1, device=cfg.DEVICE), y_better, y_anc)
            x_attempt = x_anc + v_initial * 0.1 # 小步长试探
            
            # Proxy 打分
            score_attempt = norm_proxy(x_attempt)
            # 原始分
            score_anc = norm_proxy(x_anc)
            
            # 定义 "Worse": 如果生成的点分数没有显著提高，甚至降低了，就把它当负样本
            # 或者简单粗暴：直接把尝试生成的点当作 worse，迫使模型去寻找比当前尝试“更好”的路径（DPO 逻辑）
            # 这里我们定义：x_worse 就是 x_attempt (模型当前倾向的方向)
            x_worse = x_attempt.detach()
            
            # y_worse 的标签：用 Proxy 预测分
            y_worse = score_attempt.detach()

        net.train()
        
        # 5. 计算 Loss (传入动态构建的三元组)
        # compute_loss 内部会计算 x_better 和 x_worse 的散度
        loss = cfm.compute_loss(x_anc, x_better, x_worse, y_better, y_worse, y_anc)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{total_steps} | Loss: {loss.item():.4f}")
            
    # ==========================================
    # Part C: 推理与评估 (PA-FDO 增强版)
    # ==========================================
    print("\nRunning Evaluation with Energy-based Guidance...")
    
    # 1. 准备统计量
    # 训练集质心 (用于正则化回复力)
    # 注意：我们的数据加载器返回的数据已经是标准化的，所以质心应该接近 0 向量
    centroid = torch.zeros(1, input_dim, device=cfg.DEVICE)
    # 如果想更精确，可以用当前 batch 的均值，或者 dataset 的统计量
    # centroid = torch.from_numpy(mean_x).to(cfg.DEVICE) # 如果在 dataloader 里没有减均值
    # 但在 data_loader.py 里我们做了 x = (x - mean)/std，所以均值就是 0
    
    # 2. 采样起点 (从 50th-90th 分位)
    y_flat = ds_all.tensors[1].view(-1)
    q50 = torch.quantile(y_flat, 0.5)
    q90 = torch.quantile(y_flat, 0.9)
    mask_start = (y_flat >= q50) & (y_flat <= q90)
    candidate_indices = torch.where(mask_start)[0]
    
    # 随机选 batch
    if len(candidate_indices) > cfg.NUM_SAMPLES:
        perm = torch.randperm(len(candidate_indices))[:cfg.NUM_SAMPLES]
        selected_indices = candidate_indices[perm]
    else:
        selected_indices = candidate_indices
        
    x_starts = ds_all.tensors[0][selected_indices].to(cfg.DEVICE)
    y_starts = ds_all.tensors[1][selected_indices].view(-1, 1).to(cfg.DEVICE)
    
    # 3. 构造动态目标梯度带 (Dynamic Target Band)
    # 设计：不要让所有点都去同一个极值。
    # 而是构建一个分布：N(mu=max*1.1, sigma=scale)
    # 这样可以形成一个“宽广的吸引场”，增加生成多样性
    y_max = ds_all.tensors[1].max().item()
    
    # 基础目标: Max * 1.1
    base_target = y_max * 1.1
    # 引入随机扰动，有些点目标更高，有些稍低
    target_noise = torch.randn_like(y_starts) * (y_max * 0.05) 
    y_targets = base_target + target_noise
    
    # 确保目标至少比起点高
    y_targets = torch.maximum(y_targets, y_starts * 1.05)
    
    # 4. 执行采样
    x_final = cfm.sample(
        x_starts, 
        y_target=y_targets, 
        y_start=y_starts,
        proxy=norm_proxy,
        centroid=centroid,   # 传入质心
        steps=cfg.ODE_STEPS,
        cfg_scale=4.0,       # 保持强 CFG
        grad_scale=4.0,      # 保持强 Proxy Guidance
        reg_scale=0.05       # 能量回复力系数 (防止 OOD)
    )
    
    # 5. 反标准化与评估
    x_denorm = x_final.cpu() * std_x.cpu() + mean_x.cpu()
    
    # Oracle 评估
    if hasattr(task, 'predict'):
        if task.is_discrete:
            # 离散任务处理逻辑 (如 TFBind8)
            vocab_size = 4
            seq_len = input_dim // vocab_size
            x_reshaped = x_denorm.view(x_denorm.shape[0], seq_len, vocab_size)
            x_indices = torch.argmax(x_reshaped, dim=2).cpu().numpy()
            scores = task.predict(x_indices)
        else:
            # 连续任务
            scores = task.predict(x_denorm.numpy())
            
        scores = scores.reshape(-1)
        
        # 归一化分数 (0-100th)
        y_min_val = ds_all.tensors[1].min().item()
        y_max_val = ds_all.tensors[1].max().item()
        norm_scores = (scores - y_min_val) / (y_max_val - y_min_val)
        
        percentiles = np.percentile(norm_scores, [100, 80, 50])
        
        print("-" * 30)
        print(f"Result (Valid {len(scores)}): Mean {norm_scores.mean():.4f}")
        print(f"Percentiles (100/80/50): {percentiles[0]:.4f} | {percentiles[1]:.4f} | {percentiles[2]:.4f}")
        print("-" * 30)
        return percentiles
    else:
        print("Task does not support prediction.")
        return np.zeros(3)

if __name__ == "__main__":
    # 为了演示，只跑一个 Seed
    seed_everything(42)
    main()
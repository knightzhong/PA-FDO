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
    # task, ds_all, ds_gold, (mean_x, std_x, mean_y, std_y) = build_dynamic_loader(cfg)
    # 1. 加载数据 (只拿 dataset_fixed)
    task, ds_fixed, _, (mean_x, std_x, mean_y, std_y) = build_dynamic_loader(cfg) # <--- 接口变了
    ds_all = ds_fixed
    # Loader (直接 shuffle 这个包含配对的 dataset)
    loader = torch.utils.data.DataLoader(ds_fixed, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # test
    # print(f"Norm Formula: y_norm = (y_raw - {mean_y.item():.4f}) / {std_y.item():.4f}")
    # target_norm_1 = (1.0 - mean_y) / std_y
    # print(f"To get Raw 1.0, we need Norm: {target_norm_1.item():.4f}")
    # import sys
    # sys.exit(0)
    # 统计量上设备
    mean_x, std_x = mean_x.to(cfg.DEVICE), std_x.to(cfg.DEVICE)
    mean_y, std_y = mean_y.to(cfg.DEVICE), std_y.to(cfg.DEVICE)
    input_dim = ds_all.tensors[0].shape[1]
    
    # DataLoader (Random Samplers)
    # loader_all = torch.utils.data.DataLoader(ds_all, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    # loader_gold = torch.utils.data.DataLoader(ds_gold, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 用于无限循环的 iterator
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    # iter_all = cycle(loader_all)
    # iter_gold = cycle(loader_gold)
    iter_loader = cycle(loader)
    
    # ==========================================
    # Part A: 训练 ListNet Proxy (ICLR 2025 Strategy)
    # ==========================================
    print("\nTraining RankProxy with ListNet Loss (RaM Strategy)...")
    
    # 1. 初始化
    proxy = RankProxy(input_dim=input_dim).to(cfg.DEVICE)
    proxy_opt = torch.optim.AdamW(proxy.parameters(), lr=1e-4, weight_decay=1e-5) # 论文参数
    
    def listnet_loss(y_pred, y_true, temp=1.0):
        # ---------------------------------------------------------
        # 关键修改：给 y_true 除以一个小的温度系数 (tau)
        # 这会拉大高分和低分的差距，让 Target 分布更尖锐
        # ---------------------------------------------------------
        tau = 0.1  # <--- 建议尝试 0.1 或 0.05
        
        # 预测值的温度可以保持 1.0，或者也设为 tau，通常只锐化 Target 效果就很好
        pred_temp = 1.0 
        
        # 计算 Log Softmax (预测)
        p_y_pred = F.log_softmax(y_pred.t() / pred_temp, dim=1)
        
        # 计算 Softmax (真实标签)，除以 tau 进行锐化
        # 比如 y_true=[2.0, 1.0], tau=0.1 -> [20, 10] -> Softmax 差距巨大
        p_y_true = F.softmax(y_true.t() / tau, dim=1)
        
        return -torch.sum(p_y_true * p_y_pred)

    # 准备全量数据
    all_x = ds_all.tensors[0].to(cfg.DEVICE)
    all_y = ds_all.tensors[1].to(cfg.DEVICE).view(-1, 1)
    num_samples = all_x.shape[0]
    
    # 3. Listwise 训练循环
    # 论文建议 List Length (Batch Size) m=100 或 1000
    list_size = 512 
    maxepo = 5000
    for epoch in range(maxepo):
        proxy.train()
        proxy_opt.zero_grad()
        
        # === Data Augmentation: 随机采样形成 List ===
        # 每次迭代都重新采样，相当于无限的数据增强
        idx = torch.randperm(num_samples)[:list_size]
        x_batch = all_x[idx]
        y_batch = all_y[idx]
        
        # === Forward ===
        y_pred = proxy(x_batch)
        
        # === ListNet Loss ===
        # 温度 temp=1.0 是标准设定，如果想要更sharp的分布可以调小
        loss = listnet_loss(y_pred, y_batch)
        
        loss.backward()
        proxy_opt.step()
        
        if (epoch + 1) % 20 == 0:
            pred_std = y_pred.std().item()
            print(f"RaM-ListNet Epoch {epoch+1}/{maxepo} | Loss: {loss.item():.4f} | Pred Std: {pred_std:.4f}")
            # 如果 Pred Std 一直很小 (< 0.01)，说明输出还没拉开差距


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
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)#torch.optim.Adam(net.parameters(), lr=cfg.LR)
    
    # 训练步数 (Iterations) 而非 Epochs
    total_steps = 20000 
    
    for step in range(total_steps):
        net.train()
        optimizer.zero_grad()
        
        # 1. 直接获取锁定的配对 (4项)
        # x_anc: 起点
        # y_anc: 起点分数
        # x_better: 锁定的终点 (OT配对好的)
        # y_better: 锁定的终点分数
        x_anc, y_anc, x_better, y_better = next(iter_loader)
        
        # 上设备
        x_anc = x_anc.to(cfg.DEVICE)
        y_anc = y_anc.view(-1, 1).to(cfg.DEVICE)
        x_better = x_better.to(cfg.DEVICE)
        y_better = y_better.view(-1, 1).to(cfg.DEVICE)
        
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
            # score_anc = norm_proxy(x_anc)
            
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
    
    # 2. 构造目标 (Target) - 【关键修改】
    # 我们不仅要超越 y_max，我们要去星辰大海！
    # 之前是 y_max (1.5)，现在我们直接设为 5.0 (对应 Raw ~ 0.7)
    # 如果 5.0 能稳住，下次就设 8.8 (Raw 1.0)
    base_target = 5.0 
    
    y_targets = torch.full_like(y_starts, base_target)
    
    print(f"[Info] Aggressive Targets: Norm={base_target} (Approx Raw 0.7)")
    # 4. 执行采样
    x_final = cfm.sample(
        x_starts, 
        y_target=y_targets, 
        y_start=y_starts,
        proxy=norm_proxy,
        centroid=centroid,   # 传入质心
        steps=cfg.ODE_STEPS,
        # === 🚨 严格执行这组参数 🚨 ===
        
        # 1. 关掉火箭助推 (CFG)
        # 既然模型能生成 0.95，不需要 CFG 放大，求稳！
        cfg_scale=1.0,   
        
        # 2. 开启导航 (Gradient)
        # 之前为了测试关了，现在必须开！有了导航，才能把 80% 的 0.4 变成 0.9
        # 放心，有 Clipping (5.0) 保护，开启梯度也不会炸
        grad_scale=1.0,  
        
        # 3. 保持安全绳
        reg_scale=0.1
    )
    
    # 5. 反标准化与评估
    x_denorm = x_final.cpu() * std_x.cpu() + mean_x.cpu()
    print(x_denorm)
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
        print(scores)
        
        # 归一化分数 (0-100th)
        task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268}
        task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067}
        oracle_y_min = task_to_min.get(cfg.TASK_NAME, ds_all.tensors[1].min().item())
        oracle_y_max = task_to_max.get(cfg.TASK_NAME, ds_all.tensors[1].max().item())
        # y_min_val = ds_all.tensors[1].min().item()
        # y_max_val = ds_all.tensors[1].max().item()
        norm_scores = (scores - oracle_y_min) / (oracle_y_max - oracle_y_min)
        
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
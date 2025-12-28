import torch
import numpy as np
import design_bench
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def get_design_bench_data(task_name):
    """
    加载并标准化 Design-Bench 数据
    """
    print(f"Loading task: {task_name}...")
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})
    
    # === 1. 处理 X (输入) ===
    if task.is_discrete:
        raw_x = task.x
        if raw_x.ndim == 3: raw_x = raw_x.squeeze(-1)
        x_indices = torch.tensor(raw_x, dtype=torch.long)
        
        # 强制转为 4 维 One-Hot (N, L, 4)
        vocab_size = 4
        x_onehot = F.one_hot(x_indices, num_classes=vocab_size).float()
        
        # Dequantization Noise
        noise = 0.05 * torch.randn_like(x_onehot)
        x_continuous = x_onehot + noise
        
        # Flatten
        offline_x = x_continuous.view(x_continuous.shape[0], -1).numpy()
    else:
        offline_x = task.x
    
    # === 2. 计算统计量 (Z-Score) ===
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x < 1e-6, 1.0, std_x)
    
    offline_x_norm = (offline_x - mean_x) / std_x
    
    # === 3. 处理 Y (分数) ===
    offline_y = task.y.reshape(-1)
    mean_y = np.mean(offline_y)
    std_y = np.std(offline_y)
    if std_y == 0: std_y = 1.0
    
    offline_y_norm = (offline_y - mean_y) / std_y
    
    print(f"Data Processed. X_dim={offline_x_norm.shape[1]}. Y_norm Mean={offline_y_norm.mean():.2f}, Std={offline_y_norm.std():.2f}")
    
    return task, offline_x_norm, offline_y_norm, mean_x, std_x, mean_y, std_y

def build_contrastive_triplets(x_all, y_all, k=20, dist_threshold=10.0):
    """
    构建三元组 (Anchor, Better, Worse)
    策略：
    1. 计算 KNN。
    2. Better: 在邻居中找分更高的，取 Gain 最大的。
    3. Worse: 在邻居中找分更低的，取 Loss 最大 (分数最低) 的。
    4. 如果找不到，则由自身 (Identity) 填充，并在 Loss 中 Mask 掉。
    """
    # 转 Tensor
    if isinstance(x_all, np.ndarray): x_all = torch.tensor(x_all, dtype=torch.float32)
    if isinstance(y_all, np.ndarray): y_all = torch.tensor(y_all, dtype=torch.float32)
    
    print(f"Constructing Contrastive Triplets (N={len(x_all)}, K={k})...")
    
    # 1. 计算距离矩阵 (注意显存，如果 N>20000 需分块，这里假设显存足够)
    # 对于 TFBind8 (N=32898), 32898^2 * 4 bytes ≈ 4GB 显存，可以接受。
    # 如果爆显存，请改用 KeOps 或 Faiss。
    dists = torch.cdist(x_all, x_all, p=2)
    
    # 2. 获取 Top-K 邻居
    topk_vals, topk_indices = torch.topk(dists, k=k, dim=1, largest=False)
    
    # 3. 准备数据容器
    anchors = x_all
    y_anchors = y_all
    
    betters = []
    y_betters = []
    worses = []
    y_worses = []
    
    # 统计计数
    cnt_has_better = 0
    cnt_has_worse = 0
    
    # === 循环处理每个样本 (虽然慢点，但逻辑最清晰，不易出错) ===
    # 也可以向量化，但为了"准确"，我们先写循环保证逻辑
    for i in range(len(x_all)):
        curr_y = y_all[i]
        neighbor_idx = topk_indices[i] # (K,)
        neighbor_y = y_all[neighbor_idx]
        
        # --- 找 Better ---
        # 条件：分数比我高，且距离在阈值内
        # 注意：排除自己 (距离 > 1e-6)
        mask_better = (neighbor_y > curr_y) & (topk_vals[i] < dist_threshold) & (topk_vals[i] > 1e-6)
        
        if mask_better.any():
            # 策略：在比我好的邻居里，选分数最高的 (Max Gain)
            # 也可以选最近的，这里我们选分数最高的以鼓励优化
            valid_indices = neighbor_idx[mask_better]
            valid_scores = y_all[valid_indices]
            best_idx = valid_indices[torch.argmax(valid_scores)]
            
            betters.append(x_all[best_idx])
            y_betters.append(y_all[best_idx])
            cnt_has_better += 1
        else:
            # 没找到更好的，就用自己 (Identity)
            betters.append(x_all[i])
            y_betters.append(y_all[i])
            
        # --- 找 Worse (负样本) ---
        # 条件：分数比我低
        mask_worse = (neighbor_y < curr_y) & (topk_vals[i] < dist_threshold) & (topk_vals[i] > 1e-6)
        
        if mask_worse.any():
            # 策略：在比我差的邻居里，选分数最低的 (Max Loss) -> 形成最强对比
            valid_indices = neighbor_idx[mask_worse]
            valid_scores = y_all[valid_indices]
            worst_idx = valid_indices[torch.argmin(valid_scores)] # 分数最小
            
            worses.append(x_all[worst_idx])
            y_worses.append(y_all[worst_idx])
            cnt_has_worse += 1
        else:
            worses.append(x_all[i])
            y_worses.append(y_all[i])
            
    print(f"Triplets Built. Has Better: {cnt_has_better/len(x_all):.1%}, Has Worse: {cnt_has_worse/len(x_all):.1%}")
    
    # Stack
    betters = torch.stack(betters)
    worses = torch.stack(worses)
    y_betters = torch.tensor(y_betters, dtype=torch.float32)
    y_worses = torch.tensor(y_worses, dtype=torch.float32)
    
    return anchors, betters, worses, y_anchors, y_betters, y_worses

def build_paired_dataloader(config):
    task, x_norm, y_norm, mean_x, std_x, mean_y, std_y = get_design_bench_data(config.TASK_NAME)
    
    # 构造对比三元组
    anchors, betters, worses, y_anc, y_bet, y_wor = build_contrastive_triplets(
        x_norm, y_norm, 
        k=config.TOP_K_NEIGHBORS, # 建议 50
        dist_threshold=5.0
    )
    
    # 封装
    # 注意顺序：Anchor, Better, Worse, Y_Better, Y_Worse, Y_Anchor
    dataset = TensorDataset(anchors, betters, worses, y_bet, y_wor, y_anc)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 辅助张量
    offline_x = torch.tensor(x_norm, dtype=torch.float32)
    offline_y = torch.tensor(y_norm, dtype=torch.float32)
    
    return (
        loader, task, offline_x, offline_y,
        torch.tensor(mean_x).float(), torch.tensor(std_x).float(),
        torch.tensor(mean_y).float(), torch.tensor(std_y).float(),
    )
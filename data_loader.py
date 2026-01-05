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

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def build_dynamic_loader(cfg):
    """
    构建动态数据池 (修复了离散数据 X 处理 + Y 标准化)
    """
    import design_bench
    import torch.nn.functional as F
    
    task = design_bench.make(cfg.TASK_NAME)
    
    # === 1. 处理 X (输入) - 区分离散和连续 ===
    if task.is_discrete:
        print(f"Detected Discrete Task: {cfg.TASK_NAME}. Applying One-Hot + Dequantization.")
        # 原始数据通常是 (N, L) 的整数索引
        raw_x = task.x
        if raw_x.ndim == 3: raw_x = raw_x.squeeze(-1)
        
        # 1. 转 Long Tensor
        x_indices = torch.tensor(raw_x, dtype=torch.long)
        
        # 2. One-Hot 编码 (N, L, 4) -> 扩展维度
        # 注意: 这里假设 Vocab=4 (DNA), 如果是其他任务可能不同，但 DesignBench TFBind 都是 4
        vocab_size = 4 
        x_onehot = F.one_hot(x_indices, num_classes=vocab_size).float()
        
        # 3. Dequantization (去量化): 核心步骤！
        # 给离散的 0/1 加上微小的噪声，使其变成连续分布，适合 Flow Matching 训练
        # 这样模型就能在连续空间里学习流场了
        noise = 0.05 * torch.randn_like(x_onehot)
        x_continuous = x_onehot + noise
        
        # 4. Flatten 平铺 (N, L*4)
        x = x_continuous.view(x_continuous.shape[0], -1).numpy()
        
    else:
        # 连续任务 (如 AntMorphology) 直接用
        x = task.x

    y = task.y
    
    # === 2. X 标准化 (Global Z-Score) ===
    # 即使是 One-Hot+Noise，也建议做一次标准化让均值为0，方差为1，配合神经网络初始化
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0) + 1e-8
    x = (x - mean_x) / std_x
    
    # === 3. Y 标准化 (Normalize Y) ===
    # 这一步是为了解决之前的 Flow 爆炸问题
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0) + 1e-8
    y = (y - mean_y) / std_y
    
    # 转 Tensor
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    
    # === 4. 构建高分池 ===
    y_flat = y_tensor.view(-1)
    k = int(len(y_flat) * 0.1) # Top 10%
    topk_vals, topk_indices = torch.topk(y_flat, k)
    
    x_gold = x_tensor[topk_indices]
    y_gold = y_tensor[topk_indices]
    
    print(f"Data Loaded: Input Dim={x.shape[1]} | Total {len(x_tensor)} | Gold {len(x_gold)}")
    
    # === 5. 全局固定配对 (Global Fixed OT) ===
    print("Computing Global Fixed Optimal Transport Pairs...")
    
    # 把数据放到 GPU 上算会快很多 (如果显存够)
    # TFBind8: 30000 x 3000 的矩阵，完全没问题
    device = cfg.DEVICE if torch.cuda.is_available() else 'cpu'
    x_all_gpu = x_tensor.to(device)
    x_gold_gpu = x_gold.to(device)
    
    # 计算余弦相似度矩阵 (Cosine Similarity)
    # Normalize
    x_all_norm = F.normalize(x_all_gpu, p=2, dim=1)
    x_gold_norm = F.normalize(x_gold_gpu, p=2, dim=1)
    
    # Matrix Multiplication: (N, D) @ (M, D).T -> (N, M)
    sim_matrix = torch.mm(x_all_norm, x_gold_norm.t())
    
    # 找到每个样本最近的 Gold 样本索引
    best_indices = torch.argmax(sim_matrix, dim=1).cpu() # 转回 CPU
    
    # 构建锁定的目标
    fixed_x_better = x_gold[best_indices]
    fixed_y_better = y_gold[best_indices]
    
    print(f"Fixed Pairs Computed. Mapping {len(x_tensor)} low samples -> {len(x_gold)} gold samples.")
    
    # === 6. 构建 Dataset ===
    # Dataset 包含 4 项：(起点x, 起点y, 终点x, 终点y)
    dataset_fixed = TensorDataset(x_tensor, y_tensor, fixed_x_better, fixed_y_better)
    
    stats = (
        torch.tensor(mean_x), torch.tensor(std_x),
        torch.tensor(mean_y), torch.tensor(std_y)
    )
    
    # 注意：不再返回 dataset_all/dataset_gold，直接返回配对好的 dataset_fixed
    return task, dataset_fixed, None, stats
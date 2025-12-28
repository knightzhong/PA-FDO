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
    构建动态数据池，而非静态 DataLoader。
    返回:
        task: 任务对象
        dataset_all: 包含所有 (x, y) 的 TensorDataset
        dataset_gold: 仅包含 Top K% 高分 (x, y) 的 TensorDataset
        stats: (mean_x, std_x, mean_y, std_y)
    """
    import design_bench
    task = design_bench.make(cfg.TASK_NAME)
    
    # 1. 获取离线数据
    x = task.x
    y = task.y
    
    # 2. 标准化
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0) + 1e-8
    x = (x - mean_x) / std_x
    
    # y 不做标准化，保持真实物理意义，或者仅做简单归一化供 Proxy 使用
    # 这里我们记录统计量，但 dataset 里存原始值方便比较
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0) + 1e-8
    
    # 转 Tensor
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    
    # 3. 构建高分池 (Top 10%)
    # 动态配对的核心：我们希望流向这些高分区域
    y_flat = y_tensor.view(-1)
    k = int(len(y_flat) * 0.1) # Top 10%
    topk_vals, topk_indices = torch.topk(y_flat, k)
    
    x_gold = x_tensor[topk_indices]
    y_gold = y_tensor[topk_indices]
    
    print(f"Data Loaded: Total {len(x_tensor)} | Gold Pool {len(x_gold)} (Min Score: {topk_vals.min():.4f})")
    
    dataset_all = TensorDataset(x_tensor, y_tensor)
    dataset_gold = TensorDataset(x_gold, y_gold)
    
    stats = (
        torch.tensor(mean_x), torch.tensor(std_x),
        torch.tensor(mean_y), torch.tensor(std_y)
    )
    
    return task, dataset_all, dataset_gold, stats
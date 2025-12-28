import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import design_bench

# ==========================================
# 1. 配置与辅助类
# ==========================================
class Config:
    TASK_NAME = 'TFBind8-Exact-v0'
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Proxy 训练参数
    PROXY_LR = 1e-3
    PROXY_EPOCHS = 50
    
    # 优化参数 (纯梯度上升)
    NUM_SAMPLES = 128   # 优化多少个差样本
    OPT_STEPS = 100     # 优化步数
    OPT_LR = 0.1        # 梯度上升的学习率 (步长)

class ScoreProxy(nn.Module):
    """简单的打分网络 (MLP)"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def seed_everything(seed=42):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# ==========================================
# 2. 数据加载 (Robust One-Hot Version)
# ==========================================
def get_data(task_name, device):
    print(f"Loading task: {task_name}...")
    task = design_bench.make(task_name)
    
    # 原始数据
    x = task.x
    y = task.y
    
    # 手动处理离散数据 -> 32维连续向量
    if task.is_discrete:
        print("Processing discrete data manually (One-Hot)...")
        if x.ndim == 3: x = x.squeeze(-1)
        
        # 转 Long
        x_indices = torch.tensor(x, dtype=torch.long).to(device)
        
        # One-Hot (TFBind8 Vocab=4)
        vocab_size = 4
        x_onehot = F.one_hot(x_indices, num_classes=vocab_size).float()
        
        # Flatten: (N, 8, 4) -> (N, 32)
        x_flat = x_onehot.view(x_onehot.shape[0], -1)
    else:
        x_flat = torch.tensor(x, dtype=torch.float32).to(device)
        if x_flat.dim() > 2: x_flat = x_flat.view(x_flat.shape[0], -1)
        
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device).view(-1, 1)
    
    return task, x_flat, y_tensor

# ==========================================
# 3. 纯梯度优化器 (Gradient Ascent)
# ==========================================
def optimize_with_proxy(x_start, proxy, steps=100, lr=0.1):
    """
    不使用 Flow，直接对 x 计算 Proxy 的梯度并更新
    x_{t+1} = x_t + lr * grad(Score(x_t))
    """
    # 复制一份，设置为需要梯度
    x = x_start.clone().detach().requires_grad_(True)
    
    # 使用优化器来管理更新 (相当于梯度上升)
    # 我们要最大化 Score，所以 Loss = -Score
    optimizer = torch.optim.Adam([x], lr=lr) 
    
    print(f"Starting Pure Gradient Ascent (Steps={steps}, LR={lr})...")
    for i in range(steps):
        optimizer.zero_grad()
        
        # 预测分数
        pred_score = proxy(x)
        
        # 目标：最大化分数 => 最小化负分数
        loss = -pred_score.sum()
        
        loss.backward()
        optimizer.step()
        
        # (可选) 约束 x 保持在合理范围内?
        # 真实的 One-Hot 是 0/1，但在连续松弛空间里，我们通常不强制截断
        # 这样可以看出它是否会“飞”到奇怪的数值
        
    return x.detach()

# ==========================================
# 4. 主程序
# ==========================================
def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    print(f"=== Baseline Experiment: Pure Proxy Optimization ===")
    
    # 1. 准备数据
    task, all_x, all_y = get_data(cfg.TASK_NAME, cfg.DEVICE)
    input_dim = all_x.shape[1]
    
    # 2. 训练 Proxy
    print("\n[1/3] Training Proxy Model...")
    proxy = ScoreProxy(input_dim).to(cfg.DEVICE)
    opt = torch.optim.Adam(proxy.parameters(), lr=cfg.PROXY_LR)
    loss_fn = nn.MSELoss()
    
    for i in range(cfg.PROXY_EPOCHS):
        opt.zero_grad()
        pred = proxy(all_x)
        loss = loss_fn(pred, all_y)
        loss.backward()
        opt.step()
        if (i+1) % 10 == 0:
            print(f"Epoch {i+1}/{cfg.PROXY_EPOCHS} | Loss: {loss.item():.4f}")
            
    # 3. 选择最好样本 (Low-Score Samples)
    print("\n[2/3] Selecting Best Samples...")
    y_flat = all_y.view(-1)
    _, best_indices = torch.topk(y_flat, k=cfg.NUM_SAMPLES, largest=True)
    
    x_start = all_x[best_indices]
    original_scores = y_flat[best_indices]
    
    print(f"Selected {cfg.NUM_SAMPLES} samples. Original Mean Score: {original_scores.mean().item():.4f}")
    
    # 4. 执行纯梯度优化
    print("\n[3/3] Running Optimization (No Flow Matching)...")
    x_optimized = optimize_with_proxy(x_start, proxy, steps=cfg.OPT_STEPS, lr=cfg.OPT_LR)
    
    # 5. 评估结果
    print("\n=== Evaluation Results ===")
    
    # (A) Proxy 认为它是多少分？
    with torch.no_grad():
        proxy_score = proxy(x_optimized)
    print(f"Proxy Predicted Score Mean: {proxy_score.mean().item():.4f} (Self-Evaluation)")
    
    # (B) 真实的 Oracle 认为它是多少分？
    if task.is_discrete:
        # 还原形状 (N, 8, 4)
        vocab_size = 4
        seq_len = input_dim // vocab_size
        x_reshaped = x_optimized.view(cfg.NUM_SAMPLES, seq_len, vocab_size)
        
        # Argmax 转回离散索引
        x_indices = torch.argmax(x_reshaped, dim=2).cpu().numpy()
        
        # 真实打分
        true_scores = task.predict(x_indices)
    else:
        true_scores = task.predict(x_optimized.cpu().numpy())
        
    print(f"Oracle True Score Mean:     {true_scores.mean():.4f}")
    print(f"Oracle True Score Max:      {true_scores.max():.4f}")
    sorted_scores = np.sort(np.asarray(true_scores).reshape(-1))  # 小 -> 大
    total = sorted_scores.shape[0]
    def pick_rank_fraction(f):
        idx = max(0, min(total - 1, int(np.ceil(f * total) - 1)))
        return sorted_scores[idx]
    def to_scalar(x):
        return float(np.asarray(x).squeeze())
    p50 = to_scalar(pick_rank_fraction(0.50))
    p80 = to_scalar(pick_rank_fraction(0.80))
    p100 = to_scalar(pick_rank_fraction(1.00))  # 最后一个
    print(f"Percentile Scores (50th / 80th / 100th, asc): {p50:.4f} | {p80:.4f} | {p100:.4f}")
    
    # 分析差异
    gap = proxy_score.mean().item() - true_scores.mean()
    print("-" * 30)
    print(f"Gap (Hallucination):        {gap:.4f}")
    print("-" * 30)
    
    if gap > 0.5:
        print("结论：Proxy 产生了严重的幻觉！它以为优化得很好，但生成的样本实际上是无效的（对抗样本）。")
        print("这证明了 Flow Matching (SP-RFM) 作为流形约束的必要性。")
    elif true_scores.mean() > 0.9:
        print("结论：纯梯度上升也很强？那可能是任务太简单，或者是 Proxy 泛化性极好。")
    else:
        print("结论：优化失败，Proxy 可能没指对方向，或者梯度更新步长不合适。")
def seed_everything(seed=42):
    import random
    import os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    seed_everything()
    main()
import torch
import torch.nn as nn

class VectorFieldNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256): # 保持之前的加宽
        super().__init__()
        self.input_dim = input_dim
        
        # 1. 正常的条件映射
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.y_high_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === CFG 关键修改 A: 定义一个可学习的“空条件”向量 ===
        # 当我们丢弃条件时，就用这个向量代替
        self.null_y_high = nn.Parameter(torch.randn(1, hidden_dim))
        
        self.y_low_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.x_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 联合处理 (保留 Dropout 防止过拟合)
        self.joint_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, y_high, y_low, drop_mask=None): 
        """
        drop_mask: (B, 1) 的布尔张量。True 表示丢弃条件 (Unconditional)。
        """
        if x.dim() > 2: x = x.view(x.size(0), -1)
        if t.dim() == 1: t = t.view(-1, 1)
        if y_high.dim() == 1: y_high = y_high.view(-1, 1)
        if y_low.dim() == 1: y_low = y_low.view(-1, 1)
            
        t_emb = self.time_mlp(t)
        x_emb = self.x_mlp(x)
        y_low_emb = self.y_low_mlp(y_low) # y_low 通常作为 Context 不丢弃，或者你也可以选择丢弃
        
        # === CFG 关键修改 B: 根据 mask 替换条件 ===
        y_high_emb = self.y_high_mlp(y_high)
        
        if drop_mask is not None:
            # 广播 null_y_high 到 batch 大小
            batch_null = self.null_y_high.expand(y_high.shape[0], -1)
            # 如果 mask 为 True，使用 Null Embedding；否则使用真实 Embedding
            y_high_emb = torch.where(drop_mask, batch_null, y_high_emb)
        
        h = torch.cat([x_emb, t_emb, y_high_emb, y_low_emb], dim=-1)
        v = self.joint_mlp(h)
        return v

# models.py (追加在后面)

import torch
import torch.nn as nn

class RankProxy(nn.Module):
    """
    基于 ICLR 2025 RaM 的 ListNet 代理模型。
    输入: DNA (B, D)
    输出: Unnormalized Score (Logit)
    """
    def __init__(self, input_dim, hidden_dim=1024): # 论文用了 2048，这里我们适当加宽
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm 比 BatchNorm 在 Ranking 任务更稳
            nn.SiLU(),
            nn.Dropout(0.1),          # 防止过拟合，增强外推鲁棒性
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, 1)  # 输出标量分数 (无 Sigmoid!)
        )

    def forward(self, x):
        return self.net(x)


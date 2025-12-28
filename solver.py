import torch
import torch.nn.functional as F

class ConditionalFlowMatching:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, x_anchor, x_better, x_worse, y_better, y_worse, y_anchor):
        """
        PA-FDO 核心训练逻辑
        """
        # 数据上设备
        x_anchor = x_anchor.to(self.device)
        x_better = x_better.to(self.device)
        x_worse = x_worse.to(self.device)
        y_better = y_better.view(-1, 1).to(self.device)
        y_worse = y_worse.view(-1, 1).to(self.device)
        y_anchor = y_anchor.view(-1, 1).to(self.device)
        
        B = x_anchor.shape[0]
        
        # === 1. 构建流匹配路径 (Anchor -> Better) ===
        # 我们希望流场将 Anchor 推向 Better
        t = torch.rand(B, 1, device=self.device)
        
        # 线性插值路径
        x_t = (1 - t) * x_anchor + t * x_better
        
        # 目标速度向量 (Flow Matching Target)
        u_t = x_better - x_anchor
        
        # CFG Mask (15% 概率丢弃条件)
        drop_mask = (torch.rand(B, 1, device=self.device) < 0.15)
        
        # 预测速度
        # 注意：这里的条件是 "Target Score" (y_better) 和 "Current Score" (y_anchor)
        # 模型的 forward 参数需要和 models.py 一致：forward(x, t, y_high, y_low, mask)
        v_pred = self.model(x_t, t, y_better, y_anchor, drop_mask=drop_mask)
        
        # === A. 基础流匹配损失 (MSE) ===
        loss_mse = torch.mean((v_pred - u_t) ** 2)
        
        # === B. 偏好对齐损失 (Preference Alignment Loss) ===
        # 核心思想：在 x_t 处，向量场应当更指向 Better，而远离 Worse
        # 仅当 Triplet 有效时计算 (即 Better != Anchor 且 Worse != Anchor)
        
        # 计算方向向量
        dir_better = x_better - x_anchor
        dir_worse = x_worse - x_anchor
        
        # 为了数值稳定性，归一化方向向量
        dir_better_norm = F.normalize(dir_better, p=2, dim=1)
        dir_worse_norm = F.normalize(dir_worse, p=2, dim=1)
        v_pred_norm = F.normalize(v_pred, p=2, dim=1)
        
        # 计算余弦相似度
        cos_better = torch.sum(v_pred_norm * dir_better_norm, dim=1)
        cos_worse = torch.sum(v_pred_norm * dir_worse_norm, dim=1)
        
        # Mask: 只有当确实存在 better/worse 样本时才计算 Loss
        # 如果 better == anchor (Identity), dir_better 为 0, 余弦无意义
        has_better = (torch.norm(dir_better, p=2, dim=1) > 1e-4)
        has_worse = (torch.norm(dir_worse, p=2, dim=1) > 1e-4)
        valid_triplet = has_better & has_worse
        
        loss_pref = torch.tensor(0.0, device=self.device)
        
        if valid_triplet.any():
            # Margin Loss or LogSigmoid
            # 我们希望 cos_better > cos_worse
            # diff = cos_better - cos_worse
            # loss = -log(sigmoid(scale * diff))
            scale = 10.0 # 温度系数
            pref_obj = cos_better[valid_triplet] - cos_worse[valid_triplet]
            loss_pref = -torch.log(torch.sigmoid(scale * pref_obj) + 1e-6).mean()
            
        # 总损失：MSE + lambda * Preference
        # lambda 取 0.1 ~ 0.5 之间，既要生成准确，又要方向对
        return loss_mse + 0.2 * loss_pref

    @torch.no_grad()
    def sample(self, x_start, y_target, y_start, proxy=None, steps=100, cfg_scale=4.0, grad_scale=4.0):
        """
        Uncertainty-Aware Sampling
        """
        self.model.eval()
        
        x_current = x_start.clone().to(self.device)
        y_target = y_target.to(self.device)
        y_start = y_start.to(self.device)
        B = x_current.shape[0]
        
        dt = 1.0 / steps
        
        # 预定义 CFG Masks
        mask_uncond = torch.ones((B, 1), device=self.device, dtype=torch.bool)
        mask_cond = torch.zeros((B, 1), device=self.device, dtype=torch.bool)
        
        # 如果有 Proxy，开启 Train 模式以启用 Dropout
        if proxy is not None:
            proxy.train()
        
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B, 1), t_val, device=self.device)
            
            # === 1. 计算不确定性感知梯度 (Uncertainty-Aware Gradient) ===
            grad_final = torch.zeros_like(x_current)
            
            if proxy is not None and grad_scale > 0:
                with torch.enable_grad():
                    x_in = x_current.detach().clone().requires_grad_(True)
                    
                    # MC Dropout: 采样 5 次
                    mc_preds = []
                    for _ in range(5):
                        mc_preds.append(proxy(x_in))
                    mc_preds = torch.stack(mc_preds) # (5, B, 1)
                    
                    avg_score = mc_preds.mean(dim=0)
                    uncertainty = mc_preds.std(dim=0) # (B, 1)
                    
                    # 计算梯度
                    grad = torch.autograd.grad(avg_score.sum(), x_in)[0]
                    
                    # === 动态阻尼逻辑 ===
                    # 1. 不确定性越高，权重越小 (exp(-sigma))
                    # 2. t 越接近 1，梯度权重越小 (1-t)，防止破坏最终结构
                    # 归一化 uncertainty: 简单的 trick，防止数值范围问题
                    # 这里的 2.0 是敏感度系数
                    damping_factor = torch.exp(-2.0 * uncertainty) * (1.0 - t_val)
                    
                    grad_final = grad * damping_factor * grad_scale
            
            # === 2. 计算 Flow Velocity (CFG) ===
            with torch.no_grad():
                v_cond = self.model(x_current, t, y_target, y_start, drop_mask=mask_cond)
                v_uncond = self.model(x_current, t, y_target, y_start, drop_mask=mask_uncond)
                
                v_flow = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # === 3. 混合更新 (Euler Step) ===
            v_total = v_flow + grad_final
            x_current = x_current + v_total * dt
            
        return x_current
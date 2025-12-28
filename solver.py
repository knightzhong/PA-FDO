import torch
import torch.nn.functional as F

class ConditionalFlowMatching:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_divergence(self, x_in, t, y_target, y_start, drop_mask=None):
        """
        利用 Hutchinson 估计器计算散度 div(v)
        Trace(J) approx e^T J e
        """
        x_in = x_in.detach().requires_grad_(True)
        epsilon = torch.randn_like(x_in)
        
        v_pred = self.model(x_in, t, y_target, y_start, drop_mask=drop_mask)
        v_projected = torch.sum(v_pred * epsilon, dim=1)
        
        grad_x = torch.autograd.grad(v_projected, x_in, create_graph=True)[0]
        div = torch.sum(grad_x * epsilon, dim=1)
        
        return div, v_pred

    def compute_loss(self, x_anchor, x_better, x_worse, y_better, y_worse, y_anchor):
        """
        PA-FDO 核心训练逻辑: MSE + Flow-DPO
        """
        x_anchor = x_anchor.to(self.device)
        x_better = x_better.to(self.device)
        x_worse = x_worse.to(self.device)
        y_better = y_better.view(-1, 1).to(self.device)
        y_worse = y_worse.view(-1, 1).to(self.device)
        y_anchor = y_anchor.view(-1, 1).to(self.device)
        
        B = x_anchor.shape[0]
        t = torch.rand(B, 1, device=self.device)
        
        # === 1. 基础流匹配 (MSE) ===
        x_t_better = (1 - t) * x_anchor + t * x_better
        u_t = x_better - x_anchor
        drop_mask = (torch.rand(B, 1, device=self.device) < 0.15)
        
        v_pred = self.model(x_t_better, t, y_better, y_anchor, drop_mask=drop_mask)
        loss_mse = torch.mean((v_pred - u_t) ** 2)
        
        # === 2. 路径偏好损失 (Flow-DPO) ===
        # 计算 Winner 和 Loser 路径在同一点 t 的散度
        x_t_worse = (1 - t) * x_anchor + t * x_worse
        
        # 注意：计算 DPO 时通常关闭 drop_mask 以衡量条件流场
        div_better, _ = self.compute_divergence(x_t_better, t, y_better, y_anchor)
        div_worse, _ = self.compute_divergence(x_t_worse, t, y_worse, y_anchor)
        
        # Loss: 让 Winner 收缩 (div < 0) 比 Loser 更强，或者 Loser 扩散 (div > 0)
        # diff > 0 意味着 worse 的散度比 better 大 (符合预期)
        diff = div_worse - div_better
        loss_dpo = -torch.nn.functional.logsigmoid(1.0 * diff).mean()
        
        return loss_mse + 0.1 * loss_dpo

    @torch.no_grad()
    def sample(self, x_start, y_target, y_start, 
               proxy=None, 
               centroid=None,          # 新增: 训练集质心
               steps=100, 
               cfg_scale=4.0, 
               grad_scale=4.0, 
               reg_scale=0.05):        # 新增: 回复力强度
        """
        Energy-Guided Sampling: CFG + Proxy Gradient + Manifold Restoration
        """
        self.model.eval()
        
        x_current = x_start.clone().to(self.device)
        y_target = y_target.to(self.device)
        y_start = y_start.to(self.device)
        
        # 如果提供了质心，确保它在设备上
        if centroid is not None:
            centroid = centroid.to(self.device)
            
        B = x_current.shape[0]
        dt = 1.0 / steps
        
        mask_uncond = torch.ones((B, 1), device=self.device, dtype=torch.bool)
        mask_cond = torch.zeros((B, 1), device=self.device, dtype=torch.bool)
        
        if proxy is not None:
            proxy.train() # 开启 Dropout
        
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B, 1), t_val, device=self.device)
            
            # === A. 计算不确定性感知梯度 (Gradient Guidance) ===
            grad_final = torch.zeros_like(x_current)
            
            if proxy is not None and grad_scale > 0:
                with torch.enable_grad():
                    x_in = x_current.detach().clone().requires_grad_(True)
                    
                    # MC Dropout Uncertainty
                    mc_preds = []
                    for _ in range(5):
                        mc_preds.append(proxy(x_in))
                    mc_preds = torch.stack(mc_preds)
                    
                    avg_score = mc_preds.mean(dim=0)
                    uncertainty = mc_preds.std(dim=0)
                    
                    grad = torch.autograd.grad(avg_score.sum(), x_in)[0]
                    
                    # 动态阻尼: 不确定性高或接近终点时减小梯度干扰
                    damping = torch.exp(-2.0 * uncertainty) * (1.0 - t_val)
                    grad_final = grad * damping * grad_scale
            
            # === B. 计算回复力 (Manifold Restoration) ===
            # 防止样本飞出数据流形太远 (OOD)
            # Force = - k * (x - centroid)
            v_reg = torch.zeros_like(x_current)
            if centroid is not None and reg_scale > 0:
                # 假设数据已标准化，centroid 通常接近 0
                # 这种拉力随距离线性增加
                dist_vec = x_current - centroid
                v_reg = - dist_vec * reg_scale
                
                # 可选：随时间 t 增加拉力，确保终点不发散？
                # 或者随时间 t 减小拉力，允许探索？
                # 这里我们保持恒定，作为背景场
            
            # === C. 计算流速度 (CFG) ===
            with torch.no_grad():
                v_cond = self.model(x_current, t, y_target, y_start, drop_mask=mask_cond)
                v_uncond = self.model(x_current, t, y_target, y_start, drop_mask=mask_uncond)
                
                v_flow = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # === D. 欧拉积分 ===
            # Total Velocity = Flow + Gradient + Restoration
            v_total = v_flow + grad_final + v_reg
            x_current = x_current + v_total * dt
            
        return x_current
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'depth_local_loss_weight': 0.0004,
    'depth_global_loss_weight': 0.004,
    'depth_pearson_loss_weight': 0.01,
    'depth_loss_weight': 1.0
}


# depth loss part
class DepthLoss(nn.Module):
    def __init__(self, cfg):
        super(DepthLoss, self).__init__()
        self.cfg = cfg
    def pearson_corrcoef(self, x, y): # 计算皮尔逊相关系数 (Pearson correlation coefficient)
        x = x - x.mean() # 输入: x, y 为 1D 向量 (已展平的预测与GT)
        y = y - y.mean()
        return torch.sum(x * y) / (torch.norm(x) * torch.norm(y) + 1e-8)  # 输出: [-1, 1] 之间的相关系数，越接近1相关性越高
    def normalize(self, input, mean=None, std=None): # 对输入做归一化
        input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
        input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
        return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))
    def patchify(self, input, patch_size):  # 将输入划分为 patch
        patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size) # [B, C*P*P, L]→[B, L, C*P*P]→[B*L, P*P]
        return patches
    
    def patch_norm_mse_loss(self, input, target, patch_size, margin, return_mask=False): # Patch 级别的归一化 + L2损失 (局部)
        input_patches = self.normalize(self.patchify(input, patch_size))  # 对每个patch独立归一化
        target_patches = self.normalize(self.patchify(target, patch_size))
        return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)
        # return pearson_correlation_loss(input_patches, target_patches, margin, return_mask)

    # def patch_norm_mse_loss_global(self, input, target, patch_size, margin, return_mask=False): # Patch 级别的归一化 + L2损失 (全局)
    #     input_patches = self.normalize(self.patchify(input, patch_size), std = input.std().detach()) # 用整张图的std来归一化，而不是每个patch单独的std
    #     target_patches = self.normalize(self.patchify(target, patch_size), std = target.std().detach())
    #     return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)
    #     # return pearson_correlation_loss(input_patches, target_patches, margin, return_mask)

    def patch_norm_mse_loss_global(self, input, target, patch_size, margin, return_mask=False):

        fg_mask = (input > 1e-6)   # [B,1,H,W]

        if fg_mask.sum() == 0:
            fg_mask = torch.ones_like(input).bool()

        global_std_in  = input[fg_mask].std().detach()
        global_std_tgt = target[fg_mask].std().detach()

        input_patches  = self.normalize(self.patchify(input,  patch_size), std=global_std_in)
        target_patches = self.normalize(self.patchify(target, patch_size), std=global_std_tgt)

        patch_mask = F.unfold(fg_mask.float(), kernel_size=patch_size, stride=patch_size)
        patch_mask = patch_mask.any(dim=1).reshape(-1)   # [B*L]

        input_patches  = input_patches[patch_mask]
        target_patches = target_patches[patch_mask]

        return self.margin_l2_loss(input_patches, target_patches, margin, return_mask)

    def margin_l2_loss(self, depth_out, depth_tgt, margin, return_mask=False):  # 带有 margin 的 L2 损失
        mask = (depth_out - depth_tgt).abs() > margin # 仅在 (预测 - GT) > margin 的地方计算平方误差
        if not return_mask:
            return ((depth_out - depth_tgt)[mask] ** 2).mean()
        else:
            return ((depth_out - depth_tgt)[mask] ** 2).mean(), mask

    def forward(self, depth_out, depth_tgt, patch_size, margin=0.02):
        local_loss = self.patch_norm_mse_loss(depth_out, depth_tgt, patch_size, margin)
        global_loss = self.patch_norm_mse_loss_global(depth_out, depth_tgt, patch_size, margin)
        
        # 调用皮尔逊相关系数
        pred_flat = depth_out.view(-1)
        gt_flat = depth_tgt.view(-1)
        pearson_loss = 1.0 - self.pearson_corrcoef(pred_flat, gt_flat)
        
        cfg = self.cfg

        # total_loss = cfg.depth_local_loss_weight * local_loss + cfg.depth_global_loss_weight * global_loss + cfg.depth_pearson_loss_weight * pearson_loss

        total_loss = (
            cfg["depth_local_loss_weight"] * local_loss +
            cfg["depth_global_loss_weight"] * global_loss +
            cfg["depth_pearson_loss_weight"] * pearson_loss
        )
        
        return total_loss
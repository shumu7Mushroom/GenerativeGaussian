import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs


class SyncDreamer:
    def __init__(self, device, cfg_path='configs/syncdreamer.yaml', ckpt_path='ckpt/syncdreamer-pretrain.ckpt'):
        self.device = device
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        
        print(f'[INFO] Loading SyncDreamer model from {ckpt_path}...')
        self.model = self._load_model()
        self.sampler = SyncDDIMSampler(self.model, 50)  # 50 steps for training
        
        # 缓存多视角图像
        self.cached_multiview_images = None
        self.cached_input_image = None
        
        print(f'[INFO] SyncDreamer model loaded successfully!')
    
    def _load_model(self):
        """加载SyncDreamer模型"""
        config = OmegaConf.load(self.cfg_path)
        model = instantiate_from_config(config.model)
        ckpt = torch.load(self.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model = model.to(self.device).eval()
        return model
    
    def get_multiview_images(self, input_img, elevation=0, crop_size=200, cfg_scale=2.0, batch_view_num=8):
        """
        生成多视角图像
        
        Args:
            input_img: 输入图像 tensor [1, 3, H, W]
            elevation: 仰角
            crop_size: 裁剪尺寸
            cfg_scale: CFG缩放因子
            batch_view_num: 批次视角数
        
        Returns:
            torch.Tensor: 多视角图像 [B, N, 3, H, W]
        """
        # 检查是否需要重新生成
        if self.cached_multiview_images is not None and torch.equal(input_img, self.cached_input_image):
            return self.cached_multiview_images
        
        # 转换输入图像格式
        input_img_numpy = input_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        input_img_numpy = (input_img_numpy * 255).astype(np.uint8)
        
        # 准备数据
        data = self._prepare_data(input_img_numpy, elevation, crop_size)
        
        # 生成多视角图像
        with torch.no_grad():
            x_sample = self.model.sample(self.sampler, data, cfg_scale, batch_view_num)
        
        # 处理输出格式
        B, N, _, H, W = x_sample.shape
        x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
        
        # 缓存结果
        self.cached_multiview_images = x_sample
        self.cached_input_image = input_img.clone()
        
        return x_sample
    
    def _prepare_data(self, input_img, elevation, crop_size):
        """准备输入数据"""
        # 这里需要根据您的prepare_inputs函数来实现
        # 临时保存图像
        import tempfile
        import cv2
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
            tmp_path = tmp_file.name
        
        try:
            data = prepare_inputs(tmp_path, elevation, crop_size)
            for k, v in data.items():
                data[k] = v.unsqueeze(0).to(self.device)
            return data
        finally:
            Path(tmp_path).unlink()  # 清理临时文件
    
    def train_step(self, rendered_images, poses, step_ratio=None):
        """
        计算SyncDreamer多视角监督损失
        
        Args:
            rendered_images: 渲染图像 [B*N, 3, H, W]
            poses: 相机姿态 [B*N, 4, 4]
            step_ratio: 训练步数比例 (0-1)
        
        Returns:
            torch.Tensor: 损失值
        """
        # 获取输入图像的多视角生成结果
        if self.cached_multiview_images is None:
            print("[WARNING] No cached multiview images available for SyncDreamer supervision")
            return torch.tensor(0.0, device=self.device)
        
        # 根据DreamGaussian的多视角生成策略调整
        if rendered_images.shape[0] == 1:
            # 单视角模式，不适用SyncDreamer监督
            return torch.tensor(0.0, device=self.device)
        
        # MVDream/ImageDream模式：4个视角 (0°, 90°, 180°, 270°)
        if rendered_images.shape[0] == 4:
            batch_size = 1
            num_views = 4
        else:
            # 多batch的情况
            batch_size = rendered_images.shape[0] // 4
            num_views = 4
            if rendered_images.shape[0] % 4 != 0:
                print(f"[WARNING] Rendered images count {rendered_images.shape[0]} is not divisible by 4")
                return torch.tensor(0.0, device=self.device)
        
        # 将渲染图像重新整理为 [B, N, 3, H, W]
        rendered_images = rendered_images.view(batch_size, num_views, *rendered_images.shape[1:])
        
        # 获取缓存的多视角图像，选择对应的视角
        # SyncDreamer生成16个视角，选择与DreamGaussian匹配的4个视角
        syncdreamer_images = self.cached_multiview_images[:batch_size]
        
        # 选择匹配的视角：0°, 90°, 180°, 270° 对应 SyncDreamer 的 0, 4, 8, 12 视角
        selected_views = [0, 4, 8, 12]
        syncdreamer_selected = syncdreamer_images[:, selected_views]
        
        # 调整尺寸匹配
        if rendered_images.shape[-1] != syncdreamer_selected.shape[-1]:
            syncdreamer_selected = F.interpolate(
                syncdreamer_selected.view(-1, *syncdreamer_selected.shape[2:]),
                size=rendered_images.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).view(batch_size, num_views, *rendered_images.shape[2:])
        
        # 计算多视角一致性损失
        loss = F.mse_loss(rendered_images, syncdreamer_selected)
        
        # 添加感知损失 (可选)
        if hasattr(self, 'use_perceptual_loss') and self.use_perceptual_loss:
            loss = loss + 0.1 * self._perceptual_loss(rendered_images, syncdreamer_selected)
        
        # 根据训练进度调整权重
        if step_ratio is not None:
            # 修改权重调度策略：保持一定的最小权重
            # 早期权重较高，中期保持稳定，后期略微降低但不至于消失
            if step_ratio < 0.3:
                weight = 1.0  # 前30%保持满权重
            elif step_ratio < 0.7:
                weight = 0.8  # 中期30%-70%保持0.8权重
            else:
                weight = 0.5  # 后期70%-100%保持0.5权重
            loss = loss * weight
            
            # 添加调试信息
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
            
            if self.debug_counter % 50 == 0:
                print(f"SyncDreamer loss: {loss.item():.6f}, weight: {weight:.3f}, step_ratio: {step_ratio:.3f}")
        
        return loss
    
    def _perceptual_loss(self, pred, target):
        """
        计算感知损失 (简化版本)
        """
        # 简单的梯度损失
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        return grad_loss
    
    def clear_cache(self):
        """清理缓存"""
        self.cached_multiview_images = None
        self.cached_input_image = None

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import traceback

from realesrgan_wrapper import enhance_single_image
from cam_utils import orbit_camera
from zero123xl_singleview import generate_view
# 导入 SyncDreamer 生成函数
from utils.sync_generate import generate_multiview_images

def run_syncdreamer_prediction(self):
    # 将预测结果保存在输出目录的子文件夹中
    output_dir = os.path.join(self.opt.outdir, f"syncdreamer_pred_{self.opt.input}")
    os.makedirs(output_dir, exist_ok=True)
    
    input_img_path = getattr(self.opt, 'input', None)
    
    if not input_img_path or not os.path.exists(input_img_path):
        raise ValueError(f"input image path is invalid: {input_img_path}")
    
    batch_view_num = 16  # SyncDreamer生成16张
    use_view_num = 8      
    dreamgaussian_elevations = [0, -30, 30]
    syncdreamer_elevations = [-e for e in dreamgaussian_elevations]
    
    # 为每个elevation生成图片
    for batch_idx, syncdreamer_elevation in enumerate(syncdreamer_elevations):
        batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        try:
            generate_multiview_images(
                input_path=input_img_path,
                output_path=batch_output_dir,
                elevation=syncdreamer_elevation,
                cfg_path='configs/syncdreamer.yaml',
                ckpt_path='ckpt/syncdreamer-pretrain.ckpt',
                sample_num=1,
                crop_size=200,
                cfg_scale=2,
                batch_view_num=batch_view_num,
                seed=6033,
                sampler='ddim',
                sample_steps=50,
                save_individual=False
            )
        except Exception as e:
            print(f"[ERROR] SyncDreamer batch {batch_idx} failed: {e}")
            print(traceback.format_exc())
            continue
    
    # 加载并存储生成的图片和视角信息
    multi_view_images = []
    poses = []
    vers_cache, hors_cache, radii_cache = [], [], []
    
    # 收集所有可用的图片
    all_available_images = []
    for batch_idx in range(len(dreamgaussian_elevations)):
        batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
        if not os.path.exists(batch_dir):
            continue
            
        # 选择间隔的8张图片：第2，4，6，8，10，12，14，16个（索引1，3，5，7，9，11，13，15）
        selected_indices = [1, 3, 5, 7, 9, 11, 13, 15]  # 0-based索引
        for view_idx in selected_indices:
            img_path = os.path.join(batch_dir, f"0_{view_idx}.png")
            if os.path.exists(img_path):
                elevation = dreamgaussian_elevations[batch_idx]
                all_available_images.append((batch_idx, view_idx, img_path, elevation))
    
    # 保存所有图片，不进行随机选择
    if len(all_available_images) == 0:
        print("[WARN] No SyncDreamer images available, will use input image as fallback anchor")
        self.predicted_images = None
        return
    
    # 加载所有可用的图片和计算视角信息
    for batch_idx, view_idx, img_path, elevation in all_available_images:
        # 对每张SyncDreamer生成的图片进行超分辨率处理
        enhanced_img_path = img_path.replace('.png', '_enhanced.png')
        
        # 检查是否已经处理过，如果没有则进行超分辨率处理
        if not os.path.exists(enhanced_img_path):
            print(f'[INFO] Applying super-resolution to SyncDreamer image: {img_path}')
            try:
                enhanced_img_path = enhance_single_image(
                    input_image_path=img_path,
                    output_image_path=enhanced_img_path,
                    model_name='RealESRGAN_x4plus_anime_6B',
                    outscale=4,
                    face_enhance=True,  # Enable face enhancement for better quality
                    gpu_id=0
                )
                if enhanced_img_path:
                    print(f'[INFO] Super-resolution successful for: {img_path}')
                else:
                    print(f'[WARN] Super-resolution failed for: {img_path}, using original image')
                    enhanced_img_path = img_path
            except Exception as e:
                print(f'[ERROR] Super-resolution failed for {img_path}: {e}, using original image')
                enhanced_img_path = img_path
        else:
            print(f'[INFO] Using existing enhanced image: {enhanced_img_path}')
        
        # 加载处理后的图片（如果超分辨率成功）或原始图片（如果失败）
        img = cv2.imread(enhanced_img_path)
        if img is None:
            # 如果增强的图片加载失败，尝试加载原始图片
            print(f'[WARN] Failed to load enhanced image {enhanced_img_path}, trying original')
            img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            multi_view_images.append(img)
            
            # 计算视角参数（基于间隔选择的8个视角）
            # view_idx是SyncDreamer的原始索引（1,3,5,7,9,11,13,15）
            # 将其映射到0-7的训练索引
            selected_indices = [1, 3, 5, 7, 9, 11, 13, 15]
            training_view_idx = selected_indices.index(view_idx)
            
            hor = (training_view_idx * 360.0 / use_view_num) % 360
            if hor > 180:
                hor -= 360
            ver = elevation
            radius = 0
            
            vers_cache.append(ver)
            hors_cache.append(hor)
            radii_cache.append(radius)
            
            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
            poses.append(pose)
        else:
            print(f'[ERROR] Failed to load both enhanced and original image for: {img_path}')

    if not multi_view_images:
        self.predicted_images = None
        return

    # 转换为Tensor并存储（保持超分辨率处理后的高分辨率）
    # 由于图片已经通过超分辨率处理并保存到本地，直接使用高分辨率图片，不进行下采样
    predicted_images_torch = [torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device) for img in multi_view_images]
    
    self.predicted_images = torch.cat(predicted_images_torch, dim=0)
    self.predicted_poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
    self.predicted_vers = torch.tensor(vers_cache, device=self.device)
    self.predicted_hors = torch.tensor(hors_cache, device=self.device)
    self.predicted_radii = torch.tensor(radii_cache, device=self.device)

    # print(self.predicted_images.shape)
    
    # print(f"[INFO] Cached {self.predicted_images.shape[0]} predicted views for training.")


def find_closest_anchor(self, target_ver, target_hor):
    """
    Find the closest anchor image from SyncDreamer or Zero123-xl predictions based on viewing angles.
    If the closest match is a front-facing view, use the super-resolution enhanced input image instead.
    
    Args:
        target_ver: target elevation angle
        target_hor: target horizontal angle
        
    Returns:
        closest anchor image tensor
    """
    if self.predicted_images is None or len(self.predicted_images) == 0:
        # print("[INFO] No SyncDreamer predictions available, using enhanced input image as fallback")
        return self.input_img_torch  # fallback to original input image
    
    # Calculate spherical distances instead of simple angle addition
    # Normalize angles to standard ranges first
    # Horizontal: -180 to 180, Vertical: -90 to 90
    def normalize_horizontal_angle(angle):
        """Normalize horizontal angle to [-180, 180] range"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle
    
    def normalize_vertical_angle(angle):
        """Clamp vertical angle to [-90, 90] range"""
        return max(-90, min(90, angle))
    
    # Normalize target angles
    target_hor_norm = normalize_horizontal_angle(target_hor)
    target_ver_norm = normalize_vertical_angle(target_ver)
    
    # Normalize predicted angles
    predicted_hors_norm = torch.tensor([normalize_horizontal_angle(h.item()) for h in self.predicted_hors])
    predicted_vers_norm = torch.tensor([normalize_vertical_angle(v.item()) for v in self.predicted_vers])
    
    # Convert normalized angles to radians for spherical distance calculation
    target_ver_rad = torch.deg2rad(torch.tensor(target_ver_norm))
    target_hor_rad = torch.deg2rad(torch.tensor(target_hor_norm))
    predicted_vers_rad = torch.deg2rad(predicted_vers_norm)
    predicted_hors_rad = torch.deg2rad(predicted_hors_norm)
    
    # Calculate spherical distance using the haversine-like formula
    # d = arccos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2-lon1))
    cos_spherical_dist = (torch.sin(target_ver_rad) * torch.sin(predicted_vers_rad) + 
                            torch.cos(target_ver_rad) * torch.cos(predicted_vers_rad) * 
                            torch.cos(predicted_hors_rad - target_hor_rad))
    
    # Clamp to avoid numerical issues with arccos
    cos_spherical_dist = torch.clamp(cos_spherical_dist, -1.0, 1.0)
    spherical_distances = torch.acos(cos_spherical_dist)
    
    # Convert back to degrees for consistency
    total_diffs = torch.rad2deg(spherical_distances)
    
    # Find the index of the closest match
    closest_idx = torch.argmin(total_diffs)
    
    # Check if the closest match is a front-facing view (horizontal angle close to 0)
    closest_hor = self.predicted_hors[closest_idx].item()
    closest_ver = self.predicted_vers[closest_idx].item()
    
    # Define front-facing threshold (within ±30 degrees of front view)
    front_facing_threshold = 60.0
    
    # If closest match is front-facing and elevation is also close, use the super-resolution enhanced input image
    if abs(closest_hor) <= front_facing_threshold and abs(closest_ver) <= front_facing_threshold:
        # print(f"[INFO] Closest anchor is front-facing (hor={closest_hor:.1f}°), using enhanced input image")
        return self.input_img_torch
    
    # Otherwise, use the closest predicted image
    closest_image = self.predicted_images[closest_idx].unsqueeze(0)  # [1, 3, H, W]
    
    # Keep the high-resolution enhanced anchor images (don't downscale to input size)
    # The training loop will handle the resizing to 1024x1024 if needed
    
    # print(f"[INFO] Using prediction as anchor (ver={self.predicted_vers[closest_idx].item():.1f}°, hor={closest_hor:.1f}°)")
    return closest_image

def generate_zero123xl_anchors(self):
    """
    Pre-generate Zero123xl anchor images for specific viewing angles 
    using the same angles as the commented SyncDreamer logic.
    """

    
    # 使用与SyncDreamer相同的角度配置
    dreamgaussian_elevations = [0, -30, 30]
    use_view_num = 8
    selected_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    
    # 获取超分辨率处理后的输入图片路径
    if hasattr(self.opt, 'input') and self.opt.input:
        # 构建超分辨率后的输入图片路径
        file_path = Path(self.opt.input)
        output_dir = "logs"
        
        # Handle naming convention: xx_rgba.png -> xx_enhanced_rgba.png
        stem = file_path.stem
        if stem.endswith('_rgba'):
            # Remove _rgba suffix and add _enhanced_rgba
            base_name = stem[:-5]  # Remove '_rgba'
            enhanced_stem = f"{base_name}_enhanced_rgba"
        else:
            # For other files, just add _enhanced
            enhanced_stem = f"{stem}_enhanced"
        
        enhanced_input_path = os.path.join(output_dir, f"{enhanced_stem}.png")
        
        # 检查超分辨率图片是否存在，如果不存在则使用原始图片
        if not os.path.exists(enhanced_input_path):
            enhanced_input_path = self.opt.input
            print(f'[WARN] Enhanced image not found, using original: {enhanced_input_path}')
        
        # 创建以输入图片名字命名的输出文件夹
        input_basename = file_path.stem.replace('_rgba', '').replace('_enhanced_rgba', '')
        output_folder = os.path.join(output_dir, f"{input_basename}_zero123xl_outputs")
        os.makedirs(output_folder, exist_ok=True)
        
    else:
        # fallback
        enhanced_input_path = "test/input.png"
        output_folder = "logs/zero123xl_outputs"
        os.makedirs(output_folder, exist_ok=True)

    # Store generated anchor data in the same format as SyncDreamer
    multi_view_images = []
    poses = []
    vers_cache, hors_cache, radii_cache = [], [], []
    
    # Count for batch progress tracking
    total_anchors = len(dreamgaussian_elevations) * len(selected_indices)
    generated_count = 0
    existing_count = 0
    
    print(f"[INFO] Generating Zero123xl anchor images for {total_anchors} viewing angles...")
    
    # 为每个elevation和每个选定的视角索引生成图片
    for elevation in dreamgaussian_elevations:
        for view_idx in selected_indices:
            hor = (view_idx * 360.0 / 16) % 360
            if hor > 180:
                hor -= 360
            ver = elevation

            # 生成锚点图片
            output_path = os.path.join(output_folder, f"anchor_{ver}_{hor}.png")
            enhanced_output_path = output_path.replace('.png', '_enhanced.png')

            # 检查是否已存在原图
            if os.path.exists(output_path):
                existing_count += 1
            else:
                try:
                    result = generate_view(
                        input_path=enhanced_input_path,
                        output_path=output_path,
                        elevation=ver,
                        azimuth=hor
                    )
                    generated_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to generate anchor for ver={ver}, hor={hor}: {e}")
                    continue

            # 超分辨率处理，优先加载enhanced
            if not os.path.exists(enhanced_output_path):
                try:
                    enhanced_path = enhance_single_image(
                        input_image_path=output_path,
                        output_image_path=enhanced_output_path,
                        model_name='RealESRGAN_x4plus_anime_6B',
                        outscale=4,
                        face_enhance=True,
                        gpu_id=0
                    )
                    if enhanced_path:
                        pass
                    else:
                        enhanced_output_path = output_path
                except Exception as e:
                    print(f"[ERROR] Super-resolution failed for {output_path}: {e}, using original image")
                    enhanced_output_path = output_path

            # 读取处理后的图片（优先enhanced，没有则用原图）
            img_path_to_load = enhanced_output_path if os.path.exists(enhanced_output_path) else output_path
            try:
                img = cv2.imread(img_path_to_load)
                if img is None:
                    print(f"[WARN] Failed to load enhanced image {img_path_to_load}, trying original")
                    img = cv2.imread(output_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    multi_view_images.append(img)
                    vers_cache.append(ver)
                    hors_cache.append(hor)
                    radii_cache.append(0)
                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + 0)
                    poses.append(pose)
                else:
                    print(f"[ERROR] Failed to load both enhanced and original image for: {output_path}")
            except Exception as e:
                print(f"[ERROR] Exception loading anchor image {img_path_to_load}: {e}")
                continue

    if not multi_view_images:
        print("[WARN] No Zero123xl anchors generated, will use input image as fallback")
        self.predicted_images = None
        return

    # 转换为Tensor并存储
    predicted_images_torch = [torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device) for img in multi_view_images]
    
    self.predicted_images = torch.cat(predicted_images_torch, dim=0)
    self.predicted_poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
    self.predicted_vers = torch.tensor(vers_cache, device=self.device)
    self.predicted_hors = torch.tensor(hors_cache, device=self.device)
    self.predicted_radii = torch.tensor(radii_cache, device=self.device)
    
    print(f"[INFO] Zero123xl anchor generation complete: {generated_count} new, {existing_count} existing, {len(multi_view_images)} total loaded for training.")

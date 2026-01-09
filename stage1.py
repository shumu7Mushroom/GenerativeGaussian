import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
import rembg
from pathlib import Path

import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from omegaconf import OmegaConf

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam
import math
from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import lpips
from random import randint

from PIL import Image
from torchmetrics.functional import peak_signal_noise_ratio as psnr 
import datetime

from Depth_Anything_V2.run_api import generate_depth  
from Real_ESRGAN.realesrgan_wrapper import enhance_single_image
from utils.depth_utils import DepthLoss
from utils.anchors_utils import generate_zero123xl_anchors
from utils.anchors_utils import run_syncdreamer_prediction


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_zero123xl = False # Enable zero123xl prediction 
        self.enable_syncdreamer = True # Enable SyncDreamer prediction 

        ## depth
        self.cfg = {
            'depth_local_loss_weight': 0.0004,
            'depth_global_loss_weight': 0.004,
            'depth_pearson_loss_weight': 0.01,
            'depth_loss_weight': 1.0
        }

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # Initial LPIPS
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.device)

        # Similarity threshold of generated samples.
        self.similarity_threshold = 0.3
        
        #  predicted multi-view images and poses
        self.predicted_images = None
        self.predicted_poses = None
        self.predicted_vers = None
        self.predicted_hors = None
        self.predicted_radii = None
        
        # Zero123xl - no need to pre-generate, generate on-demand
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            print(f"[INFO] loading SD...")
            from guidance.sd_utils import StableDiffusion
            self.guidance_sd = StableDiffusion(self.device)
            print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            # 统一使用本地的zero123-xl-diffusers模型
            self.guidance_zero123 = Zero123(
                self.device,
                model_key='/media/work/E/data_aigc/cache/models--ashawkey--zero123-xl-diffusers'
            )
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

        if self.enable_syncdreamer:
            run_syncdreamer_prediction(self)
            
        # Generate Zero123xl anchor images for specific viewing angles
        if self.enable_zero123xl:
            generate_zero123xl_anchors(self)

    def cleanup_temp_files(self):
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"[WARN] Failed to delete temp file {f}: {e}")

        self.temp_files.clear()

    def compute_depth_loss(self, out, image_resized, bg_color, ver, hor):
        self.temp_files = []
        depth_map = out.get("depth", None)
        if depth_map is not None and self.step >= 150:
            input_tag = getattr(self.opt, 'save_path', None)
            save_dir = os.path.join(self.opt.outdir, input_tag, "depth_maps")
            os.makedirs(save_dir, exist_ok=True)
            if depth_map.dim() == 3:
                depth_map_np = depth_map.squeeze().detach().cpu().numpy()
            else:
                depth_map_np = depth_map.detach().cpu().numpy()
            depth_map_norm = (depth_map_np - depth_map_np.min()) / (depth_map_np.ptp() + 1e-8)
            depth_map_uint8 = (depth_map_norm * 255).astype(np.uint8)
            filename = f"step{self.step}_ver{ver}_hor{hor}.png"
            depth_map_path = os.path.join(save_dir, filename)

            # cv2.imwrite(depth_map_path, depth_map_uint8)

            self.temp_files.append(depth_map_path)

            # 把 image_resized 统一转成 numpy HWC uint8
            image_np = image_resized
            if isinstance(image_np, torch.Tensor):
                image_np = image_np.detach().cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]
            elif image_np.ndim != 3:
                raise ValueError(f"Unexpected image_resized shape: {image_np.shape}")
            image_np = image_np.transpose(1, 2, 0)
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            image_resized_bgr = image_np
            rgb_filename = f"step{self.step}_ver{ver}_hor{hor}_rgb.png"
            rgb_path = os.path.join(save_dir, rgb_filename)

            # cv2.imwrite(rgb_path, image_resized_bgr)

            self.temp_files.append(rgb_path)

            predict_dir = Path(self.opt.outdir) / input_tag / "depth_predict"
            predict_dir.mkdir(parents=True, exist_ok=True)
            predict_filename = Path(filename).stem + "_predict.png"
            depth_pred_img = generate_depth(
                img=image_resized_bgr,
                input_size=518,
                outdir=str(predict_dir),
                encoder='vits',
                pred_only=True,
                grayscale=True
            )
            bg_rgb = (bg_color[:3].detach().cpu().numpy() * 255).astype(np.uint8)
            br, bgc, bb = bg_rgb
            b, g, r = cv2.split(image_resized_bgr)
            eps = 5
            mask_bg = (
                (np.abs(b.astype(int) - bb) < eps) &
                (np.abs(g.astype(int) - bgc) < eps) &
                (np.abs(r.astype(int) - br) < eps)
            )
            depth_np = np.array(depth_pred_img)
            if depth_np.ndim == 3:
                depth_np = depth_np[:, :, 0]
            depth_np[mask_bg] = 0

            # 保存
            # fixed_temp_path = predict_dir / "depth_fixed.png"
            # cv2.imwrite(str(fixed_temp_path), depth_np)
            # fixed_final_path = predict_dir / f"{Path(filename).stem}_predict_fixed.png"

            criterion = DepthLoss(self.cfg)

            gt_depth_np = depth_map_norm.astype(np.float32)
            depth_pred_np = depth_np.astype(np.float32)
            if depth_pred_np.ndim == 3:
                depth_pred_np = cv2.cvtColor(depth_pred_np, cv2.COLOR_BGR2GRAY)
            depth_pred_np = cv2.resize(
                depth_pred_np,
                (gt_depth_np.shape[1], gt_depth_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            depth_pred_np /= 255.0
            depth_out = torch.from_numpy(gt_depth_np).unsqueeze(0).unsqueeze(0)
            depth_tgt = torch.from_numpy(depth_pred_np).unsqueeze(0).unsqueeze(0)
            patch_size = randint(5, 17)
            dloss = criterion(depth_out, depth_tgt, patch_size, margin=0.02)

            # if self.step % 50 != 0:
            #     self.cleanup_temp_files()  
            
            return dloss
        
        return 0.0

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0
            depthloss = 0.0

            ### known view
            if self.input_img_torch is not None:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(image, self.input_img_torch)

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.input_mask_torch)

            ### novel view (manual batch)
            render_resolution = 512 if step_ratio < 0.4 else 1024
            images = []
            # Define the solution of LPIPS
            lpips_resolution = (1024, 1024)
            # Similarities of samples
            similarities = [] 
            
            # Store both samples and their corresponding anchors
            positive_samples = []
            negative_samples = []
            positive_anchors = []  # Corresponding anchors for positive samples
            negative_anchors = []  # Corresponding anchors for negative samples
            
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                
                if image.shape[-2:] != (1024, 1024):
                    image_resized = F.interpolate(image, size=lpips_resolution, mode="bilinear", align_corners=False)
                else:
                    image_resized = image
                
                # Find the closest pre-generated anchor for this specific viewing angle
                from utils.anchors_utils import find_closest_anchor
                closest_anchor = find_closest_anchor(self, ver, hor)

                if closest_anchor.shape[-2:] != (1024, 1024):
                    closest_anchor = F.interpolate(
                        closest_anchor, 
                        size=lpips_resolution, 
                        mode="bilinear", 
                        align_corners=False
                    )

                # Calculate LPIPS similarity with the closest anchor instead of fixed input
                similarity = self.lpips_loss_fn(image_resized, closest_anchor).item()

                similarities.append(similarity)
                
                if self.step == 500:
                    print(f"[DEBUG] Step {self.step}: ver={ver}, hor={hor}, LPIPS similarity={similarity:.4f}")
                    import datetime
                    record_path = os.path.join(os.path.dirname(__file__), 'stage1record.txt')
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    lpips_values = ', '.join([f'{v:.6f}' for v in similarities])
                    avg_lpips = sum(similarities) / len(similarities) if similarities else 0.0
                    input_info = getattr(self.opt, 'input', 'unknown')
                    with open(record_path, 'a') as f:
                        f.write(f'{now} | input: {input_info} | step: {self.step} | LPIPS: [{lpips_values}] | avg: {avg_lpips:.6f}\n')
                
                # Select positive and negative samples based on LPIPS similarity
                # Store both the sample and its corresponding anchor
                if similarity < self.similarity_threshold:
                    positive_samples.append(image_resized)
                    positive_anchors.append(closest_anchor)
                else:
                    negative_samples.append(image_resized)
                    negative_anchors.append(closest_anchor)

                images.append(image)
                
                dloss = self.compute_depth_loss(out, image_resized, bg_color, ver, hor)
                depthloss = dloss if depthloss == 0.0 else depthloss + dloss
                    
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # Compute QA-Triplet Loss
            # Ensure that all positive and negative samples are converted to PyTorch tensors.
            positive_samples = [torch.tensor(sample).to(self.device) if not isinstance(sample, torch.Tensor) else sample for sample in positive_samples]
            negative_samples = [torch.tensor(sample).to(self.device) if not isinstance(sample, torch.Tensor) else sample for sample in negative_samples]

            count = 0

            # Calculate the weighting factors N(p) and N(n)
            N_p = len(positive_samples)  # Number of positive samples
            N_n = len(negative_samples)  # Number of negative samples

            # Compute log_2(1 + N(p)) and log_2(1 + N(n))
            log_Np = torch.log2(torch.tensor(1.0 + N_p))
            log_Nn = torch.log2(torch.tensor(1.0 + N_n))

            # Initial QA_triplet_loss
            qa_triplet_loss = 0

            if log_Np == 0:
                # No positive samples case
                d_ap = 0
                for i, negative in enumerate(negative_samples):
                    # Use corresponding anchor for each negative sample
                    anchor = negative_anchors[i]
                    d_an = torch.norm(anchor - negative, p=2)
                    # print(f"d(anchor, negative): {d_an.item()}")
                    lpips_loss = self.lpips_loss_fn(negative, anchor).item() * 100
                    loss_value = torch.max(torch.tensor(0.0), log_Np * d_ap - log_Nn * (d_an + lpips_loss) + 2500)
                    qa_triplet_loss += loss_value
                    count += 1
                    
            elif log_Nn == 0:
                # No negative samples case
                d_an = 0
                for i, positive in enumerate(positive_samples):
                    # Use corresponding anchor for each positive sample
                    anchor = positive_anchors[i]
                    d_ap = torch.norm(anchor - positive, p=2)
                    # print(f"d(anchor, positive): {d_ap.item()}")
                    lpips_loss = self.lpips_loss_fn(positive, anchor).item() * 100
                    loss_value = torch.max(torch.tensor(0.0), log_Np * (d_ap + lpips_loss) - log_Nn * d_an + 0.2)
                    qa_triplet_loss += loss_value
                    count += 1

            else:
                # Both positive and negative samples exist
                # Strategy: Use angle-aware anchor selection for more meaningful comparison
                for i, positive in enumerate(positive_samples):
                    positive_anchor = positive_anchors[i]
                    
                    for j, negative in enumerate(negative_samples):
                        negative_anchor = negative_anchors[j]
                        
                        # Method 1: Use the anchor that is geometrically closer to both samples
                        # Calculate which anchor provides better "central" reference
                        pos_to_pos_anchor = torch.norm(positive_anchor - positive, p=2)
                        pos_to_neg_anchor = torch.norm(negative_anchor - positive, p=2)
                        neg_to_pos_anchor = torch.norm(positive_anchor - negative, p=2)
                        neg_to_neg_anchor = torch.norm(negative_anchor - negative, p=2)
                        
                        # Choose anchor based on minimum total distance (more balanced)
                        total_dist_pos_anchor = pos_to_pos_anchor + neg_to_pos_anchor
                        total_dist_neg_anchor = pos_to_neg_anchor + neg_to_neg_anchor
                        
                        if total_dist_pos_anchor <= total_dist_neg_anchor:
                            # Use positive anchor as reference
                            # chosen_anchor = positive_anchor
                            d_ap = pos_to_pos_anchor
                            d_an = neg_to_pos_anchor
                            lpips_neg_loss = self.lpips_loss_fn(positive, negative_anchor).item() * 100
                            lpips_pos_loss = self.lpips_loss_fn(positive, positive_anchor).item() * 100
                        else:
                            # Use negative anchor as reference
                            # chosen_anchor = negative_anchor
                            d_ap = pos_to_neg_anchor
                            d_an = neg_to_neg_anchor
                            lpips_neg_loss = self.lpips_loss_fn(negative, negative_anchor).item() * 100
                            lpips_pos_loss = self.lpips_loss_fn(negative, positive_anchor).item() * 100

                        loss_value = torch.max(torch.tensor(0.0), log_Np * (d_ap + lpips_pos_loss) - log_Nn * (d_an + lpips_neg_loss) + 1450)
                        qa_triplet_loss += loss_value
                        count += 1

            # Calculate average loss 
            if count > 0:
                qa_triplet_loss = qa_triplet_loss / count
                # print("qa_triplet_loss:", qa_triplet_loss)
                loss += qa_triplet_loss    
            
            # depth loss
            if depthloss > 0 and self.step >= 150:
                # print("depthloss:",depthloss)
                depthloss = depthloss / self.opt.batch_size
                loss += depthloss

            # guidance loss
            if self.enable_sd:
                loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        
        # Store the original file path to correctly find the prompt file later.
        original_file_for_prompt = file

        # If zero123 is enabled, perform super-resolution on the input image first.
        if self.opt.lambda_zero123 > 0:
            print(f'[INFO] Zero123 is enabled, applying super-resolution to the input image...')
            
            # Define output path for the enhanced image
            file_path = Path(file)
            output_dir = file_path.parent
            
            # Handle naming convention: xx_rgba.png -> xx_enhanced_rgba.png
            stem = file_path.stem
            if stem.endswith('_rgba'):
                # Remove _rgba suffix and add _enhanced_rgba
                base_name = stem[:-5]  # Remove '_rgba'
                enhanced_stem = f"{base_name}_enhanced_rgba"
            else:
                # For other files, just add _enhanced
                enhanced_stem = f"{stem}_enhanced"
            
            enhanced_file_path = output_dir / f"{enhanced_stem}.png"

            # Call the enhancement function
            enhanced_image_path = enhance_single_image(
                input_image_path=str(file_path),
                output_image_path=str(enhanced_file_path),
                model_name='RealESRGAN_x4plus_anime_6B',
                outscale=4,
                face_enhance=True,  # Enable face enhancement for better quality
                gpu_id=0
            )

            if enhanced_image_path:
                # print(f'[INFO] Super-resolution successful. Loading enhanced image from: {enhanced_image_path}')
                file = enhanced_image_path  # Use the enhanced image for processing
            else:
                print(f'[WARN] Super-resolution failed. Using original image: {file}')


        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'[ERROR] Failed to read image: {file}')
            return
        
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        # img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt from the original file path
        file_prompt = original_file_for_prompt.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r", encoding="utf-8") as f:
                self.prompt = f.read().strip()
        else:
            print(f'[INFO] prompt file {file_prompt} not found, using empty prompt.')
            self.prompt = ""
    
    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            # 只导出三维几何网格（mesh）
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)# 在image.yaml中设置的密度阈值在train_step中用gaussians.add_densification_stats更新
            mesh.write_ply(path)

        elif mode == 'geo+tex': # 带有纹理贴图的三维网格
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh) # 点云每个点都有一个密度density属性(跟opacity有关)表示该点在空间的重要度。density_thresh 的作用是在将高斯点云转换为三角网格（mesh）时，只保留密度大于该阈值的点
            # 将训练后的高斯点云生成体素图通过 marching cubes 等方法从体素图提取三角网格，得到mesh
            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv() # 自动为 mesh 展开 UV 坐标，生成 mesh.vt（顶点UV）和 mesh.ft（面对应的UV索引）
            mesh.auto_normal() # 计算法线信息

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32) # 创建一个全零的三通道张量，用于存储每个像素的RGB颜色
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32) # 用于统计每个像素被多少次投影覆盖，后续做平均融合

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            # vers 和 hors 分别定义了一组仰角（垂直方向）和方位角（水平方向），这些角度组合起来代表了26个不同的观察视角。这样做是为了从多个方向采样颜色，保证纹理贴图的完整性和细节
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr
            # 选择光栅化上下文将三维空间中的几何体投影到2D网格，如果没有强制使用 CUDA 光栅化且没有启用 GUI 或操作系统为 Windows为第一个if语句里的OpenGL光栅化
            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()
            # 遍历每个烘焙视角，刚才的26个视角
            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius) # radius 控制了相机在三维空间中环绕物体时的距离
                # 创建了一个临时的、完整的相机对象
                cur_cam = MiniCam(
                    pose, # 相机位姿，刚才的orbit_camera函数计算的
                    render_resolution, # 渲染分辨率宽度
                    render_resolution, # 渲染分辨率高度
                    self.cam.fovy, # 垂直视场角
                    self.cam.fovx, # 水平视场角
                    self.cam.near, # 最近剪裁面
                    self.cam.far, # 最远剪裁面
                )
                
                cur_out = self.renderer.render(cur_cam) # 相机视角渲染出一张图像，结果包含多种输出，image，alpha，depth，viewspace_points，visibility_filter，radii
                # image 渲染得到的RGB彩色图像，类型为tensor
                # alpha 渲染得到的透明度图像，类型为tensor
                # depth 渲染得到的深度图像，类型为tensor 每个像素的值表示该像素对应的3D空间深度（距离相机的远近）
                # viewspace_points 当前视角下所有高斯点投影到屏幕空间的2D坐标，类型为tensor
                # visibility_filter 可见性掩码，长度等于高斯点数量。每个元素表示对应高斯点在当前视角下是否可见,类型为tensor
                # radii 每个高斯点在屏幕空间的投影半径，用于判断点的有效性、剪枝和密度化 类型为tensor
                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1] 取出渲染结果中的RGB图像，并增加一个batch维度

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device) #位姿矩阵从numpy数组转为float32的PyTorch张量
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device) #投影矩阵从numpy数组转为float32的PyTorch张量
                # mesh.v是一个Nx3的张量，表示网格顶点的三维坐标
                # mesh.f是一个Mx3的张量，表示网格面的顶点索引，每个三角面的三个顶点索引
                # mesh.vt是一个Nx2的张量，表示网格顶点的纹理坐标
                # F.pad 补一列1，变成齐次坐标
                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0) # 将世界坐标系下的点变换到相机坐标系下
                v_clip = v_cam @ proj.T #把相机空间的顶点进一步变换到裁剪空间（clip space），为光栅化做准备
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution)) # 用 nvdiffrast 库进行光栅化，经过相机变换和投影变换后将三维网格投影到二维屏幕像素网格上
                # dr.rasterize 是 nvdiffrast 的光栅化函数。
                # glctx 是光栅化上下文，v_clip 是裁剪空间的顶点坐标，mesh.f 是网格面的顶点索引，(render_resolution, render_resolution) 是渲染分辨率，rast包含每个像素的三角形索引和重心坐标
                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1] 取出所有顶点在相机坐标系下的z轴（深度）分量，并取负号
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()#透明度掩码rast[0, ..., 3:] 取出每个像素对应的三角面索引（face id，> 0 判断该像素是否被三角面覆盖
                # 这行代码将网格顶点的UV坐标（二维纹理坐标）插值到每个像素上，得到每个像素对应的UV坐标
                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1] interpolate是插值用的，使用rast进行了像素级的映射
                # mesh.vt 是所有顶点的UV坐标，mesh.ft 是每个三角面的UV索引
                # dr.interpolate 是 nvdiffrast 的插值函数，用于将顶点属性（如纹理坐标）插值到光栅化后的像素上
                # use normal to produce a back-project mask
                # mesh.vn 是所有顶点的法线
                # mesh.fn 是每个三角面的法线索引
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera) 法线旋转到世界空间 可见性判断
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]] # 计算法线与相机视线夹角余弦，取旋转后法线的z分量

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1] alpha大于0表示像素被三角面覆盖,在mesh表面上，viewcos大于0.5表示法线大致朝向相机
                mask = mask.view(-1) # 展平成一维形状为 [H, W, 1] 的布尔张量，表示哪些像素满足上述条件

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask] # 把所有像素的UV坐标展平成二维数组，clamp(0, 1)：确保UV坐标在合法范围内
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous() # 展平成像素列表，并用mask筛选出有效像素的颜色
                
                # update texture image 把rgbs里的采集到的像素颜色根据其uv坐标uvs融合投影到albedo上
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1, # [0, 1] 线性变换到 [-1, 1]，适配网格采样的标准坐标系
                    rgbs,
                    min_resolution=256,
                    return_count=True,# 返回每个纹理像素被写入的次数
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1 # 选出写入次数小于0.1的像素
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask] # 直接累加当前视角采样到的颜色直到次数大于0.1

            mask = cnt.squeeze(-1) > 0 # 选出所有被写入过的像素
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3) #累加的颜色值除以写入次数，实现加权平均，得到最终的纹理颜色cnt[mask].repeat(1, 3) 保证每个通道都除以相同的次数
            
                # === 第二部分：新增高斯属性烘焙 ===
            print(f"[INFO] baking gaussian attributes (opacity & offset)...")
            
            gaussians = self.renderer.gaussians
            
            # 为每个网格顶点找到最近的高斯点
            print(f"[INFO] finding nearest gaussians for {mesh.v.shape[0]} vertices...")
            batch_size = 10240
            nearest_gauss_indices = []
            
            for i in tqdm.trange(0, mesh.v.shape[0], batch_size):
                v_batch = mesh.v[i:i+batch_size]
                dist = torch.cdist(v_batch, gaussians.get_xyz)
                nearest_gauss_indices.append(torch.argmin(dist, dim=-1))
            
            nearest_gauss_indices = torch.cat(nearest_gauss_indices, dim=0)

            # 提取高斯属性
            raw_opacity = gaussians.get_opacity[nearest_gauss_indices].squeeze(-1)
            baked_opacity = torch.sigmoid(raw_opacity)

            # 对opacity进行预处理和过滤
            opacity_median = torch.median(baked_opacity)
            opacity_std = torch.std(baked_opacity)
            opacity_threshold = opacity_median + 2 * opacity_std
            baked_opacity = torch.clamp(baked_opacity, 0.0, min(0.8, opacity_threshold.item()))
            baked_opacity = baked_opacity * 0.6 + 0.2

            baked_offset = gaussians.get_xyz[nearest_gauss_indices] - mesh.v

            # 将顶点属性光栅化到UV纹理空间
            v_clip = torch.cat([
                mesh.vt * 2 - 1,
                torch.zeros_like(mesh.vt[:, :1]),
                torch.ones_like(mesh.vt[:, :1])
            ], dim=-1).unsqueeze(0).to(self.device)

            rast, _ = dr.rasterize(glctx, v_clip, mesh.ft, (h, w))#拍照？

            # 光栅化opacity属性
            opacity_map, _ = dr.interpolate(
                baked_opacity.unsqueeze(0).unsqueeze(-1).contiguous(),
                rast.contiguous(), 
                mesh.f.contiguous()
            )
            opacity_map = opacity_map.squeeze(0).squeeze(-1)

            # 应用高斯模糊
            opacity_map_tensor = opacity_map.unsqueeze(0).unsqueeze(0)
            blur_kernel_size = 5
            sigma = 1.0
            kernel_size = blur_kernel_size
            kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32)**2 / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernel_2d = kernel[:, None] * kernel[None, :]
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).to(self.device)
            opacity_map_blurred = F.conv2d(opacity_map_tensor, kernel_2d, padding=kernel_size//2)
            opacity_map = opacity_map_blurred.squeeze(0).squeeze(0)

            # 光栅化offset属性
            offset_map, _ = dr.interpolate(
                baked_offset.unsqueeze(0).contiguous(),
                rast.contiguous(), 
                mesh.f.contiguous()
            )
            offset_map = offset_map.squeeze(0)

            # === 第三部分：统一的空洞修复 ===
            print(f"[INFO] inpainting all attribute maps...")
            
            # 转换为numpy进行修复
            albedo = albedo.detach().cpu().numpy()
            opacity_map = opacity_map.detach().cpu().numpy()
            offset_map = offset_map.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # 统一的修复函数
            def inpaint_attribute_map(attr_map, mask, is_opacity=False):
                from sklearn.neighbors import NearestNeighbors
                from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

                if is_opacity:
                    valid_region = mask.astype(float)
                    smoothed_attr = gaussian_filter(attr_map * valid_region, sigma=1.0)
                    valid_smoothed = gaussian_filter(valid_region, sigma=1.0)
                    valid_mask = valid_smoothed > 0.1
                    attr_map[valid_mask] = smoothed_attr[valid_mask] / valid_smoothed[valid_mask]

                inpaint_region = binary_dilation(mask, iterations=32)
                inpaint_region[mask] = 0

                search_region = mask.copy()
                not_search_region = binary_erosion(search_region, iterations=3)
                search_region[not_search_region] = 0

                search_coords = np.stack(np.nonzero(search_region), axis=-1)
                inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

                if len(search_coords) == 0 or len(inpaint_coords) == 0:
                    return attr_map

                n_neighbors = min(5, len(search_coords))
                knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(search_coords)
                distances, indices = knn.kneighbors(inpaint_coords)
                
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                for i, coord in enumerate(inpaint_coords):
                    neighbor_values = attr_map[tuple(search_coords[indices[i]].T)]
                    
                    if len(neighbor_values.shape) > 1 and neighbor_values.shape[1] > 1:
                        weighted_value = np.sum(neighbor_values * weights[i:i+1].T, axis=0)
                    else:
                        weighted_value = np.sum(neighbor_values * weights[i])
                    
                    if is_opacity:
                        weighted_value = np.clip(weighted_value, 0.2, 0.8)
                        
                    attr_map[tuple(coord)] = weighted_value
                
                return attr_map

            # 修复所有属性图
            albedo = inpaint_attribute_map(albedo, mask, is_opacity=False)
            opacity_map = inpaint_attribute_map(opacity_map, mask, is_opacity=True)
            opacity_map = np.clip(opacity_map, 0.15, 0.85)
            
            from scipy.ndimage import gaussian_filter
            opacity_map = gaussian_filter(opacity_map, sigma=0.8)
            opacity_map = np.clip(opacity_map, 0.2, 0.8)
            
            for i in range(3):
                offset_map[..., i] = inpaint_attribute_map(offset_map[..., i], mask)

            # === 第四部分：保存所有属性图 ===
            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)
            print(f"[INFO] save enhanced mesh to {path}")
        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path) # 多视角优化后保存高斯点云模型到PLY文件

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            
            # # Render front view and calculate LPIPS
            self.evaluate_front_view()
            
        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        
    @torch.no_grad()
    def evaluate_front_view(self):
        print(f"[INFO] Evaluating front view...")
        
        # Create front-facing camera (elevation=0, azimuth=0)
        pose = orbit_camera(self.opt.elevation + 0, 0, self.opt.radius + 0)
        front_cam = MiniCam(
            pose,
            1024,  # High resolution for evaluation
            1024,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        
        # Render front view with white background (consistent evaluation)
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        out = self.renderer.render(front_cam, bg_color=bg_color)
        rendered_image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
        
        # Get input image at same resolution
        input_image = self.input_img_torch
        print("input shape:",input_image.shape)
        if input_image.shape[-2:] != (1024, 1024):
            input_image = F.interpolate(
                input_image,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False
            )
        input_image = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        print("input shape:", input_image.shape)

        lpips_score = self.lpips_loss_fn(rendered_image, input_image).item()
        psnr_score = psnr(rendered_image, input_image, data_range=1.0).item()

        # Save rendered front view
        os.makedirs(self.opt.outdir, exist_ok=True)
        rendered_image_np = rendered_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rendered_image_np = (rendered_image_np * 255).astype(np.uint8)

        # Save as image
        rendered_pil = Image.fromarray(rendered_image_np)
        front_view_path = os.path.join(self.opt.outdir, f"{self.opt.save_path}_front_view.png")
        # rendered_pil.save(front_view_path)
        print(f"[INFO] Front view saved to: {front_view_path}")

        # Save psnr and lpips to txt, filename contains input image info
        input_image_name = getattr(self.opt, 'input', None)
        if input_image_name is None:
            input_image_name = getattr(self, 'input_img_path', None)
        if input_image_name is None and hasattr(self, 'input_img'):
            input_image_name = getattr(self, 'input_img', None)
        if input_image_name is not None:
            base_name = os.path.basename(str(input_image_name))
            base_name = os.path.splitext(base_name)[0]
        else:
            base_name = 'input_image'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record_path = os.path.join(os.path.dirname(__file__), "stage1record.txt")
        with open(record_path, 'a') as f:
            f.write(f"time: {now}, input_image: {base_name}, LPIPS: {lpips_score:.6f}, PSNR: {psnr_score:.6f}\n")
        print(f"[INFO] Metrics appended to: {record_path}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
        
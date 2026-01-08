import os
from random import randint
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
import trimesh
import rembg
from pathlib import Path
import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim
 
from cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer 

import lpips 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips

class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        # self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.lpips = lpips.LPIPS(net='alex').cuda()

    def forward(self, img_out, img_target):
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_target)
        return loss

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

        # renderer
        self.renderer = Renderer(opt).to(self.device)

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
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        
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
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)

        self.fixed_cam = (pose, self.cam.perspective)
        

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
            self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

        # 训练前计算16视角LPIPS
        self.calculate_stage1_lpips()

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0
            ssim_loss = 0.0
            lpipsloss = 0.0

            ### known view
            if self.input_img_torch is not None:

                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(*self.fixed_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                # rgb loss
                image = out["image"] # [H, W, 3] in [0, 1]
                valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()
                loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)

            ### novel view (manual batch)
            # render_resolution = 512
            render_resolution = 1024
            images = []
            poses = []
            vers, hors, radii = [], [], []

            # novel_albedos = []
            # novel_masks = []

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

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                ## 计算lpips loss
                if hasattr(self, 'input_img_torch') and self.input_img_torch is not None and self.step % 50 == 0:
                    reference_img = F.interpolate(self.input_img_torch, size=(render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    lpips_similarity = self.lpips_loss(image, reference_img).item()
                    print(f"LPIPS similarity: {lpips_similarity:.4f}")
                    # 直接写入stage2record.txt
                    import datetime
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    input_image_name = getattr(self.opt, 'input', None)
                    if input_image_name is not None:
                        base_name = os.path.basename(str(input_image_name))
                        base_name = os.path.splitext(base_name)[0]
                    else:
                        base_name = 'input_image'
                    record_path = os.path.join(os.path.dirname(__file__), "stage2record.txt")
                    # 收集所有batch的lpips值
                    if not hasattr(self, '_lpips_batch_values') or self.step % 50 == 0 and _ == 0:
                        self._lpips_batch_values = []
                    self._lpips_batch_values.append(lpips_similarity)
                    # 最后一个batch时写入
                    if _ == self.opt.batch_size - 1:
                        avg_lpips = sum(self._lpips_batch_values) / len(self._lpips_batch_values)
                        with open(record_path, 'a') as f:
                            f.write(f"time: {now}, input_image: {base_name}, LPIPS_batch: {self._lpips_batch_values}, LPIPS_batch_avg: {avg_lpips:.6f}\n")

               

                if hasattr(self, 'input_img_torch') and self.input_img_torch is not None:
                    reference_img = F.interpolate(self.input_img_torch, size=(render_resolution, render_resolution), mode="bilinear", align_corners=False)

                    lambda_dssim = 0.2   # 和 PGSR 一样

                    Ll1 = l1_loss(image, reference_img)
                    ssimloss = 1.0 - ssim(image, reference_img)

                    ssimloss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * ssimloss

                    ssim_loss += ssimloss

                    img_lpips = image * 2 - 1
                    ref_lpips = reference_img * 2 - 1
                    lpipsloss += self.lpips_loss(img_lpips, ref_lpips).mean()

                images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # guidance loss
            strength = step_ratio * 0.15 + 0.8
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                    refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
                else:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                    refined_images = self.guidance_sd.refine(images, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)

            if self.enable_zero123:
                # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
                # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

            if ssim_loss > 0:
                ssim_loss = ssim_loss / self.opt.batch_size
                loss = loss + 0.001 * ssim_loss

            if lpipsloss > 0:
                lpipsloss = lpipsloss / self.opt.batch_size
                # loss = loss + 0.0001 * lpipsloss

            # if self.step % 10 == 0:
            #     print("ssim_loss:", ssim_loss)
            #     print("loss:", loss)
            #     print("lpips_loss:",lpipsloss)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

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

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()
            
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

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'[ERROR] Failed to read image: {file}')
            return

        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (
            1 - self.input_mask
        )
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
    
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        
    def calculate_stage1_lpips(self):
        render_resolution = 1024
        if not hasattr(self, 'input_img_torch') or self.input_img_torch is None:
            print("[WARN] No input image available for LPIPS computation")
            return

        lpips_scores = []
        for i in range(16):
            azimuth = i * 22.5
            pose = orbit_camera(self.opt.elevation, azimuth, self.opt.radius)
            out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=1.0)
            rendered_image = out["image"]  # [H, W, 3] in [0, 1]
            rendered_image = rendered_image.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, 3, H, W]
            reference_img = F.interpolate(self.input_img_torch, size=(render_resolution, render_resolution), mode="bilinear", align_corners=False)
            # lpips expects [-1,1]
            img_lpips = rendered_image * 2 - 1
            ref_lpips = reference_img * 2 - 1
            lpips_score = self.lpips_loss(img_lpips, ref_lpips).mean().item()
            lpips_scores.append(lpips_score)
        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        print(f"[INFO] 16视角LPIPS均值: {avg_lpips:.6f}")
        # 可选：写入日志
        try:
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            input_image_name = getattr(self.opt, 'input', None)
            if self.step==0:
                stage = "stage1"
            else:
                stage = "stage2"
            if input_image_name is not None:
                base_name = os.path.basename(str(input_image_name))
                base_name = os.path.splitext(base_name)[0]
            else:
                base_name = 'input_image'
            record_path = os.path.join(os.path.dirname(__file__), "stage2record.txt")
            with open(record_path, 'a') as f:
                f.write(f"Stage: {stage}, time: {now}, input_image: {base_name}, LPIPS_16views: {lpips_scores}, LPIPS_16views_avg: {avg_lpips:.6f}\n")
        except Exception as e:
            print(f"[WARN] 写入16视角LPIPS日志失败: {e}")
        return avg_lpips
        
        

    @torch.no_grad()
    def compute_final_lpips(self):
        """
        Compute final LPIPS score between rendered image and super-resolution enhanced input image
        """
        if not hasattr(self, 'input_img_torch') or self.input_img_torch is None:
            print("[WARN] No input image available for LPIPS computation")
            return
        
        # Use the same rendering resolution as in training
        render_resolution = 1024
        
        # Render from the default front-facing view (consistent with train_step logic)
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        
        # Render right 45 degree view
        back_pose = orbit_camera(self.opt.elevation, 180, self.opt.radius)             
        out_back = self.renderer.render(back_pose, self.cam.perspective,
                                    render_resolution, render_resolution, ssaa=1.0)
        rendered_back = out_back["image"]
        rendered_back_np = (rendered_back.cpu().numpy() * 255).astype(np.uint8)
        rendered_back_np = cv2.cvtColor(rendered_back_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.opt.outdir, f"{self.opt.save_path}_back.png"),
                    rendered_back_np)
        
        ## Render left 45 degree view
        pose_left45 = orbit_camera(self.opt.elevation, -45, self.opt.radius)
        out_left45 = self.renderer.render(pose_left45, self.cam.perspective,
                                        render_resolution, render_resolution, ssaa=1.0)
        rendered_left45 = out_left45["image"]
        rendered_left45_np = (rendered_left45.cpu().numpy() * 255).astype(np.uint8)
        rendered_left45_np = cv2.cvtColor(rendered_left45_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.opt.outdir, f"{self.opt.save_path}_left45.png"),
                    rendered_left45_np)
        
        ## Render left 70 degree view
        pose_left70 = orbit_camera(self.opt.elevation, -70, self.opt.radius)
        out_left70 = self.renderer.render(pose_left70, self.cam.perspective,
                                        render_resolution, render_resolution, ssaa=1.0)
        rendered_left70 = out_left70["image"]
        rendered_left70_np = (rendered_left70.cpu().numpy() * 255).astype(np.uint8)
        rendered_left70_np = cv2.cvtColor(rendered_left70_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.opt.outdir, f"{self.opt.save_path}_left70.png"),
                    rendered_left70_np)
        
        ## Render left 120 degree view
        pose_left120 = orbit_camera(self.opt.elevation, -120, self.opt.radius)
        out_left120 = self.renderer.render(pose_left120, self.cam.perspective,
                                        render_resolution, render_resolution, ssaa=1.0)
        rendered_left120 = out_left120["image"]
        rendered_left120_np = (rendered_left120.cpu().numpy() * 255).astype(np.uint8)
        rendered_left120_np = cv2.cvtColor(rendered_left120_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.opt.outdir, f"{self.opt.save_path}_left120.png"),
                    rendered_left120_np)

        ## Render right 45 degree view
        pose_right45 = orbit_camera(self.opt.elevation, 45, self.opt.radius)
        out_left45 = self.renderer.render(pose_right45, self.cam.perspective,
                                        render_resolution, render_resolution, ssaa=1.0)
        rendered_right45 = out_left45["image"]
        rendered_right45_np = (rendered_right45.cpu().numpy() * 255).astype(np.uint8)
        rendered_right45_np = cv2.cvtColor(rendered_right45_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.opt.outdir, f"{self.opt.save_path}_right45.png"),
                    rendered_right45_np)
        
        ## Render left view
        pose_left = orbit_camera(self.opt.elevation, -90, self.opt.radius)
        out_left = self.renderer.render(pose_left, self.cam.perspective,
                                    render_resolution, render_resolution, ssaa=1.0)
        rendered_left = out_left["image"]
        rendered_left_np = (rendered_left.cpu().numpy() * 255).astype(np.uint8)
        rendered_left_np = cv2.cvtColor(rendered_left_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.opt.outdir, f"{self.opt.save_path}_left.png"),
                rendered_left_np)
        
        # Render the final result (consistent with train_step render call)
        out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=1.0)
        rendered_image = out["image"]  # [H, W, 3] in [0, 1]
        rendered_image = rendered_image.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, 3, H, W]
        
        # Prepare reference image (super-resolution enhanced input)
        reference_img = F.interpolate(self.input_img_torch, size=(render_resolution, render_resolution), mode="bilinear", align_corners=False)
        

        # Compute LPIPS similarity
        lpips_score = self.lpips_loss(rendered_image, reference_img).item()
        print(f"[INFO] Final LPIPS score: {lpips_score:.4f}")

        from torchmetrics.functional import peak_signal_noise_ratio as psnr 
        psnr_score = psnr(rendered_image, reference_img, data_range=1.0).item()
        print(f"[INFO] Front view PSNR score: {psnr_score:.4f}")

        # Optionally save the rendered image for visual comparison
        save_path = os.path.join(self.opt.outdir, f"{self.opt.save_path}_final_render.png")
        rendered_np = rendered_image[0].permute(1, 2, 0).cpu().numpy()
        rendered_np = (rendered_np * 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Final rendered image saved to {save_path}")

        # 追加保存psnr和lpips到stage2record.txt，记录时间、input_image、LPIPS和PSNR
        import datetime
        input_image_name = getattr(self.opt, 'input', None)
        if input_image_name is not None:
            base_name = os.path.basename(str(input_image_name))
            base_name = os.path.splitext(base_name)[0]
        else:
            base_name = 'input_image'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record_path = os.path.join(os.path.dirname(__file__), "stage2record.txt")

        # 新增：收集train_step第50轮的lpips值（不再保存平均值）
        lpips_batch_values = getattr(self, 'lpips_batch_values', None)
        with open(record_path, 'a') as f:
            f.write(f"time: {now}, input_image: {base_name}, ")
            if lpips_batch_values is not None and len(lpips_batch_values) > 0:
                f.write(f"LPIPS_batch: {lpips_batch_values}, ")
            f.write(f"LPIPS: {lpips_score:.6f}, PSNR: {psnr_score:.6f}\n")
        print(f"[INFO] Metrics appended to: {record_path}")

        return lpips_score

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

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

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
                    ("image", "depth", "alpha", "normal"),
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
        
        # Final LPIPS evaluation at the end of stage2
        print("[INFO] Computing final LPIPS evaluation...")
        
        self.calculate_stage1_lpips()
        # self.compute_final_lpips()
        
        # save
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    
    time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)
        
    time_end = time.time()
    print(f"[INFO] total time: {time_end - time_start:.2f} seconds")

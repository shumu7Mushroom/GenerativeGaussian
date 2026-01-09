# zero123xl_singleview.py
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2

# 固定默认参数
MODEL_KEY = "/media/work/E/data_aigc/cache/models--ashawkey--zero123-xl-diffusers"
SIZE = 256
STEPS = 75
GUIDANCE = 7.0
RADIUS = 0.1
SEED = 0
GFPGAN_MODEL = "gfpgan/weights/GFPGANv1.3.pth"
GFPGAN_UPSCALE = 1
GFPGAN_CENTER = False

def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    else:
        img = img.convert("RGB")
    return img

def _sharpness_score(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"))
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _gfpgan_restore_if_face(pil_img: Image.Image, device: str = "cuda") -> Image.Image:
    """仅在检测到人脸时进行GFPGAN修复"""
    from gfpgan import GFPGANer
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    restorer = GFPGANer(
        model_path=GFPGAN_MODEL,
        upscale=GFPGAN_UPSCALE,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    cropped_faces, restored_faces, restored_bgr = restorer.enhance(
        img_bgr, has_aligned=False, only_center_face=GFPGAN_CENTER, paste_back=True
    )

    if not restored_faces:
        return pil_img
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(restored_rgb)

def generate_view(input_path: str, output_path: str, elevation: float, azimuth: float) -> str:
    """
    根据输入图片生成指定仰角和水平角的视角图（功能全默认开启）

    Args:
        input_path: 输入图片路径
        output_path: 生成图片保存路径
        elevation: 仰角 (°)
        azimuth: 水平角 (°)

    Returns:
        保存图片的绝对路径
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    from diffusers import DDIMScheduler
    from zero123 import Zero123Pipeline
    pipe = Zero123Pipeline.from_pretrained(MODEL_KEY, torch_dtype=dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    image = _load_image(input_path)
    
    # 检查图片尺寸，如果不是256x256则自动调整
    if image.size != (SIZE, SIZE):
        print(f"[INFO] 输入图片尺寸为 {image.size}，自动调整为 {SIZE}x{SIZE}")
        image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device).manual_seed(SEED)
    kw = dict(
        image=image,
        elevation=float(elevation),
        azimuth=float(azimuth),
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
        output_type="pil",
    )

    try:
        result = pipe(distance=RADIUS, **kw)
    except TypeError:
        result = pipe(radius=RADIUS, **kw)

    out = result.images[0]
    try:
        out = _gfpgan_restore_if_face(out, device=device)
    except Exception as e:
        print(f"[GFPGAN WARNING] 修复异常，跳过：{e}")
        # pass

    score = _sharpness_score(out)
    out.save(output_path)
    print(f"[DONE] sharpness={score:.1f} -> {Path(output_path).resolve()}")

    return str(Path(output_path).resolve())

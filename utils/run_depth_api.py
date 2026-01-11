import os
import cv2
import numpy as np
import matplotlib
import torch
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


def generate_depth(img, input_size=518, outdir="./vis_depth", encoder='vitl', pred_only=True, grayscale=False, device='auto'):
    """Generate depth images for a single image path or a list file.

    Args:
        img_path: path to an image file or a text file listing images.
        input_size: model input size
        outdir: output directory
        encoder: which encoder to use ('vits','vitb','vitl','vitg')
        pred_only: if True, only save the depth map; otherwise concat with RGB
        grayscale: if True, save grayscale depth map

    Returns:
        list of saved file paths
    """
    # device selection: 'auto' uses CUDA if available, 'cpu' forces CPU, or pass torch device string like 'cuda:0'
    if device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif device == 'cpu':
        DEVICE = 'cpu'
    else:
        DEVICE = device

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    base_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(base_dir, 'checkpoints', f'depth_anything_v2_{encoder}.pth')

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    # load checkpoint (load to CPU first to avoid CUDA OOM when loading large files)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        depth_anything.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # move model to requested device
    try:
        depth_anything = depth_anything.to(DEVICE).eval()    
    except Exception:
        # fallback to cpu if requested device not available
        depth_anything = depth_anything.to('cpu').eval()

    os.makedirs(outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # saved = []

    # 这里假设 img 是 numpy，HWC，BGR/uint8，如果是 tensor 你可以在外面先转好
    raw_image = img
    if raw_image is None:
        # return saved  # 或者直接 raise
        raise ValueError("Input img is None in generate_depth")
    
    # 前向推理得到深度
    depth = depth_anything.infer_image(raw_image, input_size)

    # 归一化到 0–255
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    depth = depth.astype(np.uint8)

    # 灰度 or 伪彩
    if grayscale:
        depth_out = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth_out = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # 因为现在没有 filename，可以用固定名字，反正你外面还会重命名
    out_name = "depth.png"
    save_path = os.path.join(outdir, out_name)

    if pred_only:
        # cv2.imwrite(save_path, depth_out)
        # saved.append(save_path)
        return depth_out   
    else:
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth_out])
        # cv2.imwrite(save_path, combined_result)
        # saved.append(save_path)
        return depth_out
    
    # return saved
    
if __name__ == '__main__':
    # keep the original CLI behavior for backward compatibility
    import argparse

    parser = argparse.ArgumentParser(description='Depth Anything V2 API wrapper')
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()

    generate_depth(args.img_path, input_size=args.input_size, outdir=args.outdir, encoder=args.encoder, pred_only=args.pred_only, grayscale=args.grayscale)

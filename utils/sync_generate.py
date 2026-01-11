# import sys
# import os
# syncdreamer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SyncDreamer'))
# if syncdreamer_path not in sys.path:
#     sys.path.insert(0, syncdreamer_path)
    
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from SyncDreamer.ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from SyncDreamer.ldm.util import instantiate_from_config, prepare_inputs
# from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
# from ldm.util import instantiate_from_config, prepare_inputs


def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def generate_multiview_images(
    input_path: str,
    output_path: str,
    elevation: float,
    cfg_path: str = 'configs/syncdreamer.yaml',
    ckpt_path: str = 'ckpt/syncdreamer-pretrain.ckpt',
    sample_num: int = 4,
    crop_size: int = 200,
    cfg_scale: float = 2.0,
    batch_view_num: int = 8,
    seed: int = 6033,
    sampler: str = 'ddim',
    sample_steps: int = 50,
    save_individual: bool = False
):
    """
    生成多视角图像的主函数
    
    Args:
        input_path: 输入图像路径
        output_path: 输出目录路径
        elevation: 仰角
        cfg_path: 配置文件路径
        ckpt_path: 模型权重路径
        sample_num: 采样数量
        crop_size: 裁剪尺寸
        cfg_scale: CFG 缩放因子
        batch_view_num: 批次视角数
        seed: 随机种子
        sampler: 采样器类型
        sample_steps: 采样步数
        save_individual: 是否保存拼接图像（默认只保存单独的视角图像）
    
    Returns:
        np.ndarray: 生成的图像数组，形状为 (B, N, H, W, C)
    """
    torch.random.manual_seed(seed)
    np.random.seed(seed)


    model = load_model(cfg_path, ckpt_path, strict=True)
    
    # 正确的模型加载方式：先实例化，再加载权重
    # config = OmegaConf.load(cfg_path)
    # model = instantiate_from_config(config.model)
    # print(f'loading model from {ckpt_path} ...')
    # ckpt = torch.load(ckpt_path, map_location='cpu')
    # if 'state_dict' in ckpt:
    #     state_dict = ckpt['state_dict']
    # else:
    #     state_dict = ckpt
    # model.load_state_dict(state_dict, strict=True)
    # model = model.cuda().eval()

    assert isinstance(model, SyncMultiviewDiffusion)
    Path(output_path).mkdir(exist_ok=True, parents=True)

    # prepare data
    data = prepare_inputs(input_path, elevation, crop_size)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], sample_num, dim=0)

    if sampler == 'ddim':
        sampler_obj = SyncDDIMSampler(model, sample_steps)
    else:
        raise NotImplementedError
    x_sample = model.sample(sampler_obj, data, cfg_scale, batch_view_num)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    print(f"x_sample.shape: {x_sample.shape}")

    # 单独保存每个视角（默认行为）
    for bi in range(B):
        for ni in range(N):
            print(f"Saving {bi}_{ni}.png, shape: {x_sample[bi, ni].shape}")
            output_fn = Path(output_path) / f'{bi}_{ni}.png'
            imsave(output_fn, x_sample[bi, ni])

    # 拼装保存（可选）
    if save_individual:
        for bi in range(B):
            output_fn = Path(output_path) / f'{bi}_combined.png'
            imsave(output_fn, np.concatenate([x_sample[bi, ni] for ni in range(N)], 1))

    alpha_mask = x_sample[...,3]

    return x_sample, alpha_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt', type=str, default='ckpt/syncdreamer-pretrain.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--elevation', type=float, required=True)
    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=200)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)
    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--save_individual', action='store_true', help='是否额外保存拼接图像（默认只保存单独的视角图像）')
    flags = parser.parse_args()

    # 调用新的函数
    generate_multiview_images(
        input_path=flags.input,
        output_path=flags.output,
        elevation=flags.elevation,
        cfg_path=flags.cfg,
        ckpt_path=flags.ckpt,
        sample_num=flags.sample_num,
        crop_size=flags.crop_size,
        cfg_scale=flags.cfg_scale,
        batch_view_num=flags.batch_view_num,
        seed=flags.seed,
        sampler=flags.sampler,
        sample_steps=flags.sample_steps,
        save_individual=flags.save_individual
    )

if __name__=="__main__":
    main()


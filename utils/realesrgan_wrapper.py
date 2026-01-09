"""
Real-ESRGAN 包装器
提供简单的函数调用接口来使用Real-ESRGAN进行图像超分辨率处理
"""

import argparse
import cv2
import glob
import os
import shutil
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def enhance_image(
    input_path,
    output_path=None,
    model_name='RealESRGAN_x4plus_anime_6B',
    denoise_strength=0.5,
    outscale=4,
    model_path=None,
    suffix='out',
    tile=0,
    tile_pad=10,
    pre_pad=0,
    face_enhance=False,
    fp32=False,
    alpha_upsampler='realesrgan',
    ext='auto',
    gpu_id=None
):
    """
    使用Real-ESRGAN增强图像
    
    参数:
        input_path (str): 输入图像路径或文件夹路径
        output_path (str, optional): 输出文件夹路径，默认为'results'
        model_name (str): 模型名称，支持的模型:
            - RealESRGAN_x4plus_anime_6B (默认，适合动漫图像)
            - RealESRGAN_x4plus
            - RealESRNet_x4plus  
            - RealESRGAN_x2plus
            - realesr-animevideov3
            - realesr-general-x4v3
        denoise_strength (float): 去噪强度 (0-1)，仅用于realesr-general-x4v3模型
        outscale (float): 最终放大倍数
        model_path (str, optional): 自定义模型路径
        suffix (str): 输出图像后缀
        tile (int): 分块大小，0表示不分块
        tile_pad (int): 分块填充大小
        pre_pad (int): 预填充大小
        face_enhance (bool): 是否使用GFPGAN增强人脸
        fp32 (bool): 是否使用fp32精度
        alpha_upsampler (str): Alpha通道上采样器 ('realesrgan' 或 'bicubic')
        ext (str): 图像扩展名 ('auto', 'jpg', 'png')
        gpu_id (int, optional): GPU设备ID
    
    返回:
        bool: 处理是否成功
    """
    
    if output_path is None:
        output_path = 'results'
    
    # 确定模型配置
    model_name = model_name.split('.')[0]
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 确定模型路径
    if model_path is not None:
        model_path = model_path
    else:
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # 使用dni控制去噪强度
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # 创建增强器
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    if face_enhance:  # 使用GFPGAN进行人脸增强
        from gfpgan import GFPGANer

        # 优先使用本地模型路径
        gfpgan_model_path = 'gfpgan/weights/GFPGANv1.3.pth'
        
        # 检查本地模型是否存在，如果不存在则回退到URL下载
        if not os.path.isfile(gfpgan_model_path):
            print(f"本地 GFPGAN 模型未在 {gfpgan_model_path} 找到，尝试从网络下载...")
            gfpgan_model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        else:
            print(f"找到本地 GFPGAN 模型: {gfpgan_model_path}")

        face_enhancer = GFPGANer(
            model_path=gfpgan_model_path,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 获取输入路径列表
    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted(glob.glob(os.path.join(input_path, '*')))

    success_count = 0
    total_count = len(paths)
    
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print(f'处理中 {idx+1}/{total_count}: {imgname}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'无法读取图像: {path}')
            continue
            
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print(f'错误: {error}')
            print('如果遇到CUDA内存不足，请尝试设置更小的tile值。')
            continue
        
        # 确定输出文件扩展名
        if ext == 'auto':
            extension = extension[1:]
        else:
            extension = ext
        if img_mode == 'RGBA':  # RGBA图像应该保存为png格式
            extension = 'png'
        
        # 生成输出文件路径
        if suffix == '':
            save_path = os.path.join(output_path, f'{imgname}.{extension}')
        else:
            save_path = os.path.join(output_path, f'{imgname}_{suffix}.{extension}')
        
        # 保存图像
        cv2.imwrite(save_path, output)
        success_count += 1
        print(f'已保存: {save_path}')

    print(f'处理完成! 成功: {success_count}/{total_count}')
    return success_count == total_count


def enhance_single_image(
    input_image_path,
    output_image_path=None,
    model_name='RealESRGAN_x4plus_anime_6B',
    outscale=4,
    **kwargs
):
    """
    增强单张图像的简化接口
    
    参数:
        input_image_path (str): 输入图像路径
        output_image_path (str, optional): 输出图像路径，如果不指定则自动生成
        model_name (str): 模型名称
        outscale (float): 放大倍数
        **kwargs: 其他参数传递给enhance_image函数
    
    返回:
        str: 输出图像路径，如果失败则返回None
    """
    
    if not os.path.isfile(input_image_path):
        print(f'输入文件不存在: {input_image_path}')
        return None
    
    # 如果没有指定输出路径，自动生成
    if output_image_path is None:
        input_dir = os.path.dirname(input_image_path)
        input_name, input_ext = os.path.splitext(os.path.basename(input_image_path))
        output_image_path = os.path.join(input_dir, f'{input_name}_enhanced_rgba{input_ext}')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # 为了避免覆盖原始文件，使用临时目录和唯一后缀
    temp_output_dir = os.path.dirname(output_image_path)
    temp_suffix = 'temp_enhanced'  # 使用临时后缀避免与原文件冲突
    
    # 调用主函数，使用临时后缀确保不会覆盖原始文件
    success = enhance_image(
        input_path=input_image_path,
        output_path=temp_output_dir,
        model_name=model_name,
        outscale=outscale,
        suffix=temp_suffix,
        **kwargs
    )
    
    if success:
        # 临时输出文件路径（enhance_image函数生成的）
        input_name = os.path.splitext(os.path.basename(input_image_path))[0]
        temp_output_path = os.path.join(temp_output_dir, f'{input_name}_{temp_suffix}.png')
        
        # 将临时文件移动到最终位置
        if os.path.exists(temp_output_path):
            shutil.move(temp_output_path, output_image_path)  # 移动文件到最终位置
            return output_image_path
    
    return None


# 示例使用
if __name__ == '__main__':
    # 示例1: 增强单张图像
    input_img = 'inputs/test.jpg'
    output_img = 'results/test_enhanced.jpg'
    
    result = enhance_single_image(
        input_image_path=input_img,
        output_image_path=output_img,
        model_name='RealESRGAN_x4plus_anime_6B',
        outscale=4
    )
    
    if result:
        print(f'图像增强成功: {result}')
    else:
        print('图像增强失败')
    
    # 示例2: 批量处理文件夹中的图像
    success = enhance_image(
        input_path='inputs',
        output_path='results',
        model_name='RealESRGAN_x4plus_anime_6B',
        outscale=4,
        suffix='enhanced'
    )
    
    if success:
        print('批量处理成功')
    else:
        print('批量处理中有失败的图像')

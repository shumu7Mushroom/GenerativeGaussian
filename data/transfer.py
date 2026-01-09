from PIL import Image
import numpy as np

# 读取原图像
img = Image.open('./anya1_rgba.png')
print(f"原图像模式: {img.mode}")
print(f"原图像尺寸: {img.size}")

# 确保转换为RGBA模式
if img.mode != 'RGBA':
    img = img.convert('RGBA')
    print("已转换为RGBA模式")

img_array = np.array(img)
print(f"数组形状: {img_array.shape}")

# 确保有4个通道
if len(img_array.shape) == 3 and img_array.shape[2] == 4:
    # 保存为RGBA图像
    Image.fromarray(img_array).save('./anya_rgba_fixed.png')
    print("✅ 已保存RGBA图像为 'anya_rgba_fixed.png'")
elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
    # 添加alpha通道
    alpha = np.ones((img_array.shape[0], img_array.shape[1], 1), dtype=img_array.dtype) * 255
    img_with_alpha = np.concatenate([img_array, alpha], axis=2)
    Image.fromarray(img_with_alpha).save('./anya_rgba_fixed.png')
    print("✅ 已添加alpha通道并保存为 'anya_rgba_fixed.png'")
else:
    print(f"❌ 意外的图像格式，数组形状: {img_array.shape}")

test_img = Image.open('./anya_rgba_fixed.png')
print(f"保存后的图像模式: {test_img.mode}")
print(f"保存后的图像通道数: {np.array(test_img).shape}")
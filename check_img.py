from PIL import Image
import numpy as np

# 加载图片
img_path = "data/test/1.png"
img = Image.open(img_path)

# 转为 NumPy 数组
img_array = np.array(img)

# 检查图像形状和通道数
print(f"Image shape: {img_array.shape}")  # 输出形状 (H, W, C)，C 表示通道数
if len(img_array.shape) == 3:  # 如果是彩色图像
    print(f"Number of channels: {img_array.shape[2]}")

# 检查整体像素值范围
print(f"Overall Min pixel value: {img_array.min()}")
print(f"Overall Max pixel value: {img_array.max()}")

# 检查每个通道的值范围
if len(img_array.shape) == 3:  # 检查是否有多个通道
    for i in range(img_array.shape[2]):  # 遍历每个通道
        channel_min = img_array[..., i].min()
        channel_max = img_array[..., i].max()
        print(f"Channel {i}: Min value = {channel_min}, Max value = {channel_max}")
else:  # 如果是灰度图像
    print("This is a grayscale image.")

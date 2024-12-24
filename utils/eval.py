import torch
from PIL import Image
import torchvision.transforms as transforms
from evaluation import psnr, ssim  # 替换为你的模块路径

# 1. 加载图像并转换为张量
def load_image(image_path):
    """
    Load an image and convert it to a normalized PyTorch tensor.

    Args:
        image_path (str): Path to the image.

    Returns:
        torch.Tensor: Image tensor with shape (1, 3, H, W) and values in [0, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts to [C, H, W] and normalizes to [0, 1]
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    return transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

# 2. End-to-end测试函数
def test_images(image1_path, image2_path):
    """
    Compare two images and calculate PSNR and SSIM.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        tuple: PSNR and SSIM values.
    """
    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Ensure shapes match
    assert img1.shape == img2.shape, "Images must have the same dimensions!"

    # Move images to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Compute PSNR and SSIM
    psnr_value = psnr(img1, img2).item()  # Batch of size 1, take the scalar value
    ssim_value = ssim(img1, img2).item()

    return psnr_value, ssim_value

# 3. 测试
if __name__ == "__main__":
    image1_path = "/home/gagagk16/Rain/Derain/TransWeather/imgs/2.png"  # Path to the first image
    image2_path = "/home/gagagk16/Rain/Derain/TransWeather/imgs/1.png"  # Path to the second image

    psnr_value, ssim_value = test_images(image1_path, image2_path)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

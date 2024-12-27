import torch
import torch.nn.functional as F
from torchvision.utils import save_image

def psnr(pred_image, gt, max_val=1.0):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between predicted and ground truth images.

    Args:
        pred_image (torch.Tensor): Predicted image tensor with shape (B, C, H, W).
        gt (torch.Tensor): Ground truth image tensor with the same shape as pred_image.
        max_val (float): Maximum pixel value of the images. Default is 1.0 for normalized images.

    Returns:
        torch.Tensor: Batch of PSNR values.
    """
    # Ensure input tensors have the same shape
    assert pred_image.shape == gt.shape, "Predicted and ground truth images must have the same shape."

    # Compute Mean Squared Error (MSE)
    mse = torch.mean((pred_image - gt) ** 2, dim=(-3, -2, -1))  # Average over C, H, W dimensions

    # Avoid division by zero
    mse = torch.clamp(mse, min=1e-10)

    # Compute PSNR
    psnr_value = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_value

def ssim(pred_image, gt, max_val=1.0, window_size=11, k1=0.01, k2=0.03):
    """
    Compute SSIM (Structural Similarity Index Measure) between predicted and ground truth images.

    Args:
        pred_image (torch.Tensor): Predicted image tensor with shape (B, C, H, W).
        gt (torch.Tensor): Ground truth image tensor with the same shape as pred_image.
        max_val (float): Maximum pixel value of the images. Default is 1.0 for normalized images.
        window_size (int): Size of the Gaussian kernel for local statistics. Default is 11.
        k1 (float): Constant for SSIM formula. Default is 0.01.
        k2 (float): Constant for SSIM formula. Default is 0.03.

    Returns:
        torch.Tensor: Batch of SSIM values.
    """
    # Ensure input tensors have the same shape
    assert pred_image.shape == gt.shape, "Predicted and ground truth images must have the same shape."
    B, C, H, W = pred_image.shape

    # Gaussian kernel
    def create_window(window_size, channel):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords**2) / (2 * 1.5**2))
        g /= g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
        return window.to(pred_image.device)

    window = create_window(window_size, C)
    padding = window_size // 2

    # Local mean
    mu1 = F.conv2d(pred_image, window, padding=padding, groups=C)
    mu2 = F.conv2d(gt, window, padding=padding, groups=C)

    # Local variance
    sigma1_sq = F.conv2d(pred_image * pred_image, window, padding=padding, groups=C) - mu1**2
    sigma2_sq = F.conv2d(gt * gt, window, padding=padding, groups=C) - mu2**2
    sigma12 = F.conv2d(pred_image * gt, window, padding=padding, groups=C) - mu1 * mu2

    # SSIM calculation
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean(dim=(-3, -2, -1))  # Reduce spatial dimensions

def validation(net, dataloader, device):
    """
    Perform validation and compute average PSNR and SSIM over the dataset.

    Args:
        net (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
        tuple: Average PSNR and SSIM across the dataset.
    """
    net.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():  # Disable gradient computation
        for batch_id, (input_image, gt_image) in enumerate(dataloader):
            input_image = input_image.to(device)
            
            gt_image = gt_image.to(device)

            # Model prediction
            pred_image = net(input_image)

            # Compute PSNR and SSIM
            batch_psnr = psnr(pred_image, gt_image)
            batch_ssim = ssim(pred_image, gt_image)

            psnr_list.extend(batch_psnr.cpu().numpy())
            ssim_list.extend(batch_ssim.cpu().numpy())

            # if batch_id % 100 == 0:
            #     print(f"Validation Batch {batch_id}: PSNR = {torch.mean(batch_psnr).item():.2f}, SSIM = {torch.mean(batch_ssim).item():.4f}")

    # Calculate averages
    avg_psnr = torch.tensor(psnr_list).mean().item()
    avg_ssim = torch.tensor(ssim_list).mean().item()

    return avg_psnr, avg_ssim
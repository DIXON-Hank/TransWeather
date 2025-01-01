import math

def adjust_learning_rate(optimizer, epoch, initial_lr, eta_min, use_cos=True, T_max=50, warmup_epochs=0, warmup_lr_start=1e-4):
    """
    Args:
        initial_lr (float): starting learning rate
        T_max (int): T, a.k.a. epoch numbers
        eta_min (float): min learning rate
    """
    if eta_min is None:
        eta_min = initial_lr / 10
    else:
        eta_min = eta_min
    
    if use_cos:
        if epoch < warmup_epochs:
            new_lr = warmup_lr_start + (initial_lr - warmup_lr_start) * (epoch / warmup_epochs)
        else:
        # calculate current lr 
            cos_epoch = epoch - warmup_epochs
            cos_T_max = T_max - warmup_epochs
            new_lr = eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * cos_epoch / cos_T_max)) / 2

        # update new lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
        return new_lr
    else:
        return initial_lr

def print_log(epoch, total_epochs, epoch_time, train_psnr, train_ssim, val_psnr, val_ssim, exp_name):
    print(f"Epoch [{epoch}/{total_epochs}] - Time: {epoch_time:.2f}s")
    print(f"Train PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
    print(f"Validation PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
    log_path = f"./exp/{exp_name}/training_log.txt"
    with open(log_path, "a") as f:
        f.write(f"Epoch [{epoch}/{total_epochs}] - Time: {epoch_time:.2f}s\n")
        f.write(f"Train PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}\n")
        f.write(f"Validation PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}\n")

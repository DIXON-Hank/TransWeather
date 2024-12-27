import math

def adjust_learning_rate(optimizer, epoch, initial_lr, T_max=50, eta_min=1e-5):
    """
    Args:
        initial_lr (float): starting learning rate
        T_max (int): T, a.k.a. epoch numbers
        eta_min (float): min learning rate
    """
    # calculate current lr 
    new_lr = eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

    # update new lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    return new_lr

def print_log(epoch, total_epochs, epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    print(f"Epoch [{epoch}/{total_epochs}] - Time: {epoch_time:.2f}s")
    print(f"Train PSNR: {train_psnr:.2f}")
    print(f"Validation PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
    log_path = f"./exp/{exp_name}/training_log.txt"
    with open(log_path, "a") as f:
        f.write(f"Epoch [{epoch}/{total_epochs}] - Time: {epoch_time:.2f}s\n")
        f.write(f"Train PSNR: {train_psnr:.2f}\n")
        f.write(f"Validation PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}\n")
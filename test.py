import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import myDataloader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import Compose
import os
import numpy as np
import random
from transweather_model import Transweather,UNet

# --- Parse hyper-parameters --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-weight_path', help='Directory for the used weight', type=str, required=True)
parser.add_argument('-data_dir', help='Directory for validation/test data', default='./data/test/', type=str)
parser.add_argument('-output_dir', help='Directory for your output data', default='./data/output', type=str)
parser.add_argument('-seed', help='Set random seed', default=19, type=int)
args = parser.parse_args()

# --- Assign parameters --- #
batch_size = args.batch_size
weight_path = args. weight_path
data_dir = args.data_dir
output_dir = args.output_dir
seed = args.seed

# --- Set random seed --- #
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f'Seed set to: {seed}')
else:
    print('No loaded seed')

# --- GPU device setup --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- data loader --- #
data_loader = DataLoader(
    myDataloader(data_dir, mode="test"),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# --- Define the network --- #
net = Transweather().cuda()
net = nn.DataParallel(net, device_ids=device_ids)
net = net.to(device)

# --- Load the network weight --- #

if not os.path.exists(weight_path):
    raise FileNotFoundError(f"Weight file not found at: {weight_path}")
net.load_state_dict(torch.load(weight_path, map_location=device))
print(f"Loaded weights from {weight_path}")

# --- Create output dir --- #
os.makedirs(output_dir, exist_ok=True)
# os.makedirs(os.path.join(output_dir, "raindrop"), exist_ok=True)

# --- Begin testing --- #
net.eval()
start_time = time.time()
with torch.no_grad():  # Disable gradient computation
    for idx, (input_image, _) in enumerate(data_loader):
        input_image = input_image.to(device)
        # Forward pass
        output_image = net(input_image).clamp(-1, 1) 
        
        # Save input and output images
        for i in range(input_image.size(0)):  # Handle batch size > 1
            input_save_path = os.path.join(args.output_dir, f"{idx}_input.png")
            output_save_path = os.path.join(args.output_dir, f"{idx}_output.png")
            visualization_path = os.path.join(args.output_dir, f"{idx}_visualization.png")
            inverse_transform = Compose([
                lambda tensor: tensor * torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)  # 将张量转换回 PIL 图像
            ])
            # Save images
            input_image_0 = inverse_transform(input_image[i].cpu())
            output_image_0 = inverse_transform(output_image[i].cpu())
            save_image(input_image_0, input_save_path)
            save_image(output_image_0, output_save_path)

            # Visualize input and output side by side
            # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # # axes[0].imshow(input_image[i].cpu().permute(1, 2, 0).numpy())
            # axes[0].set_title("Input")
            # axes[0].axis("off")
            # # axes[1].imshow(output_image[i].cpu().permute(1, 2, 0).numpy())
            # axes[1].set_title("Output")
            # axes[1].axis("off")
            # plt.savefig(visualization_path)
            # plt.close()

        print(f"Processed batch {idx + 1}/{len(data_loader)}")

end_time = time.time() - start_time
print(f"Inference completed in {end_time:.2f} seconds")

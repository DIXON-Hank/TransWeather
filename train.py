import time
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import myDataloader, create_random_subset
from utils.tools import print_log, adjust_learning_rate
from utils.evaluation import psnr, ssim, validation
from torchvision.models import vgg16
from perceptual import LossNetwork

import numpy as np
import random

from mymodel import UNetTransformerWithAttentionFusion

plt.switch_backend('agg')


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[224, 224], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=8, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.05, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', default="debug", help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=39, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))

os.makedirs("./exp/{}".format(exp_name), exist_ok=True)
os.makedirs("./exp/{}/.logs".format(exp_name), exist_ok=True)
# set tensorboard writer
writer = SummaryWriter(log_dir="./exp/{}/.logs".format(exp_name))

root_dir = '/home/gagagk16/Rain/Derain/Dataset/RainDS/RainDS_syn/'
DEBUG = False

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = UNetTransformerWithAttentionFusion()
denormalize = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.5, 1/0.5, 1/0.5]),  # 假设训练时使用的 std=[0.5, 0.5, 0.5]
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),       # 假设训练时使用的 mean=[0.5, 0.5, 0.5]
])

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('./exp/{}/'.format(exp_name))==False:     
    os.mkdir('./exp/{}/'.format(exp_name))  
try:
    net.load_state_dict(torch.load('./exp/{}/best'.format(exp_name)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
train_types = ["raindrop"]
val_types = ["raindrop"]
train_dataset = myDataloader(root_dir, crop_size, mode="train", data_types=train_types)
val_dataset = myDataloader(root_dir, crop_size, mode="val", data_types=val_types) # TODO: add different validation set on different stage

if DEBUG:
    train_dataset = create_random_subset(train_dataset, 0.1)
    val_dataset = create_random_subset(val_dataset, 0.1)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8)

print("Using {} train pairs".format(len(train_dataset)))
print("Using {} val pairs".format(len(val_dataset)))


# --- Previous PSNR and SSIM in testing --- #
net.eval()
# old_val_psnr, old_val_ssim = validation(net, val_dataloader, device)
# print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
old_val_psnr = 0
old_val_ssim = 0
val_psnr = 0
val_ssim = 0

print("-" * 10, "Starting Training", "-" * 10)
net.train()
try:
    for epoch in range(epoch_start, num_epochs):
        psnr_list = []
        ssim_list = []
        start_time = time.time()
        current_lr = adjust_learning_rate(optimizer, epoch, learning_rate, None, T_max=200, warmup_epochs=5, use_cos=True)
        print("Current lr:", current_lr)
    #-------------------------------------------------------------------------------------------------------------
        for batch_id, train_data in enumerate(train_dataloader):

            input_image, gt = train_data  
            input_image = input_image.to(device)
            gt = gt.to(device)

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            pred_image = net(input_image)

            smooth_loss = F.smooth_l1_loss(pred_image, gt)
            perceptual_loss = loss_network(pred_image, gt)

            loss = smooth_loss + lambda_loss * perceptual_loss 

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # --- To calculate average PSNR --- #
            psnr_list.extend(psnr(pred_image.detach().cpu(), gt.detach().cpu()))
            ssim_list.extend(ssim(pred_image.detach().cpu(), gt.detach().cpu()))

            if not (batch_id % 100):
                print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
                print('Loss:{}'.format(loss))

            # --- TensorBoard: 记录训练损失 --- #
            writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_dataloader) + batch_id)

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)
        train_ssim = sum(ssim_list) / len(ssim_list)

        # --- Save the network parameters --- #
        torch.save(net.state_dict(), './exp/{}/latest'.format(exp_name))

        # --- Use the evaluation model in testing --- #
        net.eval()
        if  epoch >= 0 : 
            val_psnr, val_ssim = validation(net, val_dataloader, device)
            one_epoch_time = time.time() - start_time
            print("Validation on {}".format(val_types))
            print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, train_ssim, val_psnr, val_ssim, exp_name)

        # --- TensorBoard: 记录 PSNR 和 SSIM --- #
        writer.add_scalar("PSNR/Train", train_psnr, epoch)
        writer.add_scalar("SSIM/Train", train_ssim, epoch)
        writer.add_scalar("PSNR/Validation", val_psnr, epoch)
        writer.add_scalar("SSIM/Validation", val_ssim, epoch)
        writer.add_scalar("learning_rate", current_lr, epoch)

        # --- TensorBoard: 每 10 个 epoch 记录图像 --- #
        if epoch % 10 == 0 and epoch != 0:
            writer.add_images("Input Images", denormalize(input_image), epoch)
            writer.add_images("Predicted Images", denormalize(pred_image.clamp(-1, 1)), epoch)
            writer.add_images("Ground Truth", denormalize(gt), epoch)

        # --- update the network weight --- #
        if val_psnr >= old_val_psnr:
            torch.save(net.state_dict(), './exp/{}/best'.format(exp_name))
            print('model saved')
            old_val_psnr = val_psnr
finally:
    writer.close()
    torch.cuda.empty_cache()
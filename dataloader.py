from torch.utils.data import Dataset
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
import os
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class myDataloader(Dataset):
    def __init__(self, crop_size, root_dir, mode, data_types):
        super().__init__()
        self.crop_size = crop_size
        self.mode = mode # train or val
        self.data_types = data_types
        self.input_paths = []
        self.gt_paths = []
        
        for data_type in data_types:
            input_dir = os.path.join(root_dir, data_type, mode)
            gt_dir = os.path.join(root_dir, 'gt')

            input_files = sorted(os.listdir(input_dir))
            for input_file in input_files:
                gt_name = self.get_gt_file(input_file)
                gt_file = os.path.join(gt_dir, mode, gt_name)
                self.input_paths.append(os.path.join(input_dir, input_file))
                self.gt_paths.append(gt_file)
                
        print("Sucessfully loaded {} {} images.".format(len(input_files), mode))

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_path = self.input_paths[index]
        gt_path = self.gt_paths[index]

        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        width, height = input_img.size

        #zoom if the input image is smaller than the crop size
        if width < crop_width or height < crop_height:
            input_img = input_img.resize((max(crop_width, width), max(crop_height, height)), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((max(crop_width, width), max(crop_height, height)), Image.Resampling.LANCZOS)

        # random cropping
        # TODO: add data augmentation? complete val 
        if self.mode == "train":
            x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
            input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
            gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        elif self.mode == "val":
            wd_new,ht_new = input_img.size
            if ht_new>wd_new and ht_new>1024:
                wd_new = int(np.ceil(wd_new*1024/ht_new))
                ht_new = 1024
            elif ht_new<=wd_new and wd_new>1024:
                ht_new = int(np.ceil(ht_new*1024/wd_new))
                wd_new = 1024
            wd_new = int(16*np.ceil(wd_new/16.0))
            ht_new = int(16*np.ceil(ht_new/16.0))
            input_crop_img = input_img.resize((wd_new,ht_new), Image.Resampling.LANCZOS)
            gt_crop_img = gt_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_tensor = transform_input(input_crop_img)
        gt_tensor = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(input_tensor.shape)[0] != 3 or list(gt_tensor.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_path))

        return input_tensor, gt_tensor

    def get_gt_file(self, input_file):
        gt_match = re.search(r"(\w+_\d+_\d+)", input_file)
        if not gt_match:
            raise RuntimeError("Gt name extraction failed!!")
        gt_file = str(gt_match.group(1)) + "_leftImg8bit.png"
        return gt_file

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_paths)
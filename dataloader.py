from torch.utils.data import Dataset, Subset
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
import os
import numpy as np
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class myDataloader(Dataset):
    def __init__(self, root_dir, crop_size=[224, 224], mode="train", data_types="all"):
        super().__init__()
        self.crop_size = crop_size
        self.mode = mode # train or val
        self.data_types = data_types
        self.input_paths = []
        self.gt_paths = []
        if mode in {"train", "val", "val_all"}:
            for data_type in data_types:
                input_dir = os.path.join(root_dir, data_type, mode)
                gt_dir = os.path.join(root_dir, 'gt')

                input_files = sorted(os.listdir(input_dir))
                for input_file in input_files:
                    # gt_name = self.get_gt_file(input_file)
                    gt_name = input_file
                    gt_path = os.path.join(gt_dir, mode, gt_name)
                    self.input_paths.append(os.path.join(input_dir, input_file))
                    self.gt_paths.append(gt_path)
        elif mode == "test":
            input_files = sorted(os.listdir(root_dir))
            self.input_paths = [os.path.join(root_dir, fname) for fname in input_files]
            self.gt_paths = self.input_paths
        
        print("Sucessfully loaded {} {} images.".format(len(self.input_paths), mode))

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_path = self.input_paths[index]
        gt_path = self.gt_paths[index]

        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        width, height = input_img.size

        # zoom if the input image is smaller than the crop size
        if width < crop_width or height < crop_height and self.mode == "train":
            input_img = input_img.resize((max(crop_width, width), max(crop_height, height)), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((max(crop_width, width), max(crop_height, height)), Image.Resampling.LANCZOS)

        # random cropping
        width, height = input_img.size
        # TODO: add data augmentation? flipping lightning
        if self.mode == "train":
            x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
            input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
            gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        else:
            target_width, target_height = (768, 384) if width >= 768 and height >= 384 else (512, 256)
            input_img = input_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            center_x, center_y = input_img.width // 2, input_img.height // 2
            left = center_x - target_width // 2
            top = center_y - target_height // 2
            right = center_x + target_width // 2
            bottom = center_y + target_height // 2

            input_crop_img = input_img.crop((left, top, right, bottom))
            gt_crop_img = gt_img.crop((left, top, right, bottom))
        
        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform_gt = Compose([ToTensor()])
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
    
def create_random_subset(dataset, subset_ratio=0.1):
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    subset_indices = random.sample(range(total_size), subset_size)  # 随机选择索引
    return Subset(dataset, subset_indices)
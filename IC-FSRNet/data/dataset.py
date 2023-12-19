from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch
import glob
class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        self.args = args
        self.train = train
        self.imgs_HR_path = os.path.join(root, 'HR')

        self.imgs_HR = sorted(
            glob.glob(os.path.join(self.imgs_HR_path, '*.png'))
        )

        self.imgs_LR_path = os.path.join(root, 'LLR_x16')

        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))
        )
      
        self.transform = transforms.ToTensor()
        self.train = train


    def __getitem__(self, item):
        img_path_HR = os.path.join(self.imgs_HR_path, self.imgs_HR[item])
        (filepath, tempfilename) = os.path.split(img_path_HR)
        img_path_LR = os.path.join(self.imgs_LR_path, tempfilename)
        LR = Image.open(img_path_LR)
        HR = Image.open(img_path_HR)
        HR = np.array(HR)
        LR = np.array(LR)
        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        HR = ToTensor()(HR)
        LR = ToTensor()(LR)
        filename = os.path.basename(img_path_HR)
        return LR, HR, filename

    def __len__(self):
        return len(self.imgs_HR)


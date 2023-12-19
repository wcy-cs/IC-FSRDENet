from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os

class LRHRDataset(Dataset):
    def __init__(self, data_LR, data_HR,l_resolution=16, r_resolution=256, split='train', data_len=-1):

        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split


        self.sr_path = Util.get_paths_from_images(
                '{}'.format(data_LR))
        self.hr_path = Util.get_paths_from_images(
                '{}'.format(data_HR))

        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.dataset_len#self.data_len #

    def __getitem__(self, index):
        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        filepath, tempfilename = os.path.split(self.hr_path[index])

        [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return {'HR': img_HR, 'SR': img_SR, 'Index': index, 'filename': tempfilename}

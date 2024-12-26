import random

import cv2
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from albumentations import (Compose,
    HorizontalFlip,  Resize,
    Normalize)

def get_train_transforms():
    # p:使用此转换的概率，默认值为 0.5
    return Compose([
        # HorizontalFlip(p=0.5),  # 水平翻转
        Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0), # 归一化，将像素值除以255，减去每个通道的平均值并除以每个通道的std
    ], p=1.)


def get_valid_transforms():
    return Compose([
        Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], p=1.)


class MyDataset(Dataset):
    def __init__(self, data_path, data_list,
                 transforms=None,
                 ):
        super().__init__()
        self.transforms = transforms
        self.data_list = data_list
        self.data_path = data_path

    def __getitem__(self, index: int):
        npy_data = np.load(self.data_path + '/' + self.data_list[index])
        enhance_data = npy_data[:8, ...]
        normal_data = npy_data[8:, ...]

        enhance_data_list = []
        normal_data_list = []

        for i in range(enhance_data.shape[0]):
            data = enhance_data[i]
            data = cv2.resize(data, (64, 128))
            if self.transforms:
                data = self.transforms(image=data)['image']
            img = torch.from_numpy(data).type(torch.FloatTensor)
            img = img.permute(2, 1, 0)
            enhance_data_list.append(img)
        data_enhance = torch.stack(enhance_data_list, 0)

        for i in range(normal_data.shape[0]):
            data = normal_data[i]
            data = cv2.resize(data, (64, 128))
            if self.transforms:
                data = self.transforms(image=data)['image']
            img = torch.from_numpy(data).type(torch.FloatTensor)
            img = img.permute(2, 1, 0)
            normal_data_list.append(img)
        data_normal = torch.stack(normal_data_list, 0)

        # if random.uniform(0, 1) > 0.5:
        #     return data_enhance, data_normal
        # else:
        #     return data_normal, data_enhance

        return data_enhance, data_normal

    def __len__(self):
        return len(self.data_list)







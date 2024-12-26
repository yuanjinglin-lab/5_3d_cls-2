import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch.nn.functional as F
from tqdm import tqdm


class MyDataset(Dataset):

    def __init__(self, data_root="data/pkl", phase="train"):

        self.data_root = data_root
        self.total_data_list = []
        self.label_list = []

        for pkl_name in tqdm(sorted(os.listdir(data_root))):
            self.total_data_list.append(pkl_name)

            # pkl_path = os.path.join(self.data_root, pkl_name)
            # with open(pkl_path, 'rb') as f:
            #     data = pickle.load(f)
            # data = torch.tensor(data) / 255.0
            # data = data.permute(0, 4, 1, 2, 3)
            # _, _, D, H, W = data.shape
            # data = data.reshape(4 * 3, D, H, W)
            # self.total_data_list.append((data, pkl_name))

        total_num = len(self.total_data_list)
        total_idx = np.arange(0, total_num)
        np.random.seed(2022)
        np.random.shuffle(total_idx)

        self.data_idx = []
        if phase == "train":
            self.data_idx = total_idx[:int(total_num * 0.7)]
        elif phase == "val":
            self.data_idx = total_idx[int(total_num * 0.7): int(total_num * 1.0)]

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):

        pkl_name = self.total_data_list[self.data_idx[idx]]
        pkl_path = os.path.join(self.data_root, pkl_name)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        data = torch.tensor(data) / 255.0
        data = data.permute(0, 4, 1, 2, 3)

        data = random_noise(data)
        data = random_crop_resize(data)
        data = random_flip(data)

        _, _, T, H, W = data.shape
        data = data.reshape(4 * 3, T, H, W)
        # data, pkl_name = self.total_data_list[self.data_idx[idx]]

        label = pkl_name.split(".")[0].split("_")[-1]
        if label == "normal":
            label = 0
        elif label == "abnormal":
            label = 1
        label = torch.tensor(label)

        return data.float(), label.long()


def random_flip(data):
    flip_flag = torch.rand(1)
    if flip_flag < 0.2:
        flip_dims = []
        for flip_i in range(2, 5):
            flip_i_flag = torch.rand(1)
            if flip_i_flag < 0.5:
                flip_dims.append(int(flip_i))
        if len(flip_dims) > 0:
            data = torch.flip(data, dims=flip_dims)

    return data


def random_crop_resize(data):
    shape = data.shape[2:]
    crop_flag = torch.rand(1)
    if crop_flag < 0.3:
        x1 = np.random.randint(low=1, high=3)
        y1 = np.random.randint(low=1, high=10)
        z1 = np.random.randint(low=1, high=5)
        x2 = np.random.randint(low=1, high=3)
        y2 = np.random.randint(low=1, high=10)
        z2 = np.random.randint(low=1, high=5)
        data = data[:, :, x1:-x2, y1:-y2, z1:-z2]
        data = F.interpolate(data, shape)

    return data


def random_noise(data):
    noise_flag = torch.rand(1)
    if noise_flag < 0.3:
        max_mean = 0.01
        max_std = 0.05
        mean = torch.rand(1) * max_mean
        std = torch.rand(1) * max_std
        data = add_gaussian_noise(data, mean.item(), std.item())
    
    return data


def add_gaussian_noise(input, mean=0, std=1):
    # 生成与输入形状相同的高斯随机数张量
    noise = torch.normal(mean=mean, std=std, size=input.size()).to(input.device)
    # 将噪声加到输入上
    return input + noise


if __name__ == "__main__":
    from tqdm import tqdm
    dataset = MyDataset()
    for i in range(5):
        for _ in tqdm(dataset):
            pass



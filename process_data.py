import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import pickle


def load_video(video_path):
    clip = VideoFileClip(video_path)
    # print(clip.size)
    print(video_path, clip.fps)
    frame_list = []
    for frame_number, frame in enumerate(clip.iter_frames()):
        frame_array = np.array(frame)
        frame_list.append(frame_array)
    clip.close()

    video_data = np.stack(frame_list)

    print(video_data.shape)
    return video_data

def load_normal_user_data(user_dir, sample_frame_num):
    user_data = []
    for video_name in ["A1.avi", "B1.avi", "C1.avi", "D1.avi"]:
        video_path = os.path.join(user_dir, video_name)
        if os.path.exists(video_path):
            video_data = load_video(video_path)
        else:
            print("%s empty" % video_path)
            video_data = np.zeros((sample_frame_num, 708, 508, 3))
        # total_num = len(video_data)
        # uniform_sample_ids = [round(total_num / sample_frame_num * i) for i in range(sample_frame_num)]
        # # print(uniform_sample_ids)
        # video_data = video_data[uniform_sample_ids]

        video_data = torch.tensor(video_data).float()
        video_data = video_data.permute(3, 0, 1, 2).unsqueeze(dim=0)
        # 708 508
        video_data = F.interpolate(video_data, (20, 96, 64), mode="trilinear")[0].permute(1, 2, 3, 0)
        video_data = video_data.numpy().astype("uint8")

        user_data.append(video_data)
    user_data = np.stack(user_data)
    print(user_data.shape)
            
    return user_data


def load_abnormal_user_data(user_dir, sample_frame_num):
    user_data = []
    for video_name in ["A1.avi", "B1.avi", "C1.avi", "D1.avi"]:
        video_path = os.path.join(user_dir, video_name)
        if os.path.exists(video_path):
            video_data = load_video(video_path)
        else:
            print("%s empty" % video_path)
            video_data = np.zeros((sample_frame_num, 872, 1896, 3))
        # total_num = len(video_data)
        # uniform_sample_ids = [round(total_num / sample_frame_num * i) for i in range(sample_frame_num)]
        # # print(uniform_sample_ids)
        # video_data = video_data[uniform_sample_ids]

        video_data = torch.tensor(video_data).float()
        video_data = video_data.permute(3, 0, 1, 2).unsqueeze(dim=0)
        # 872 1896
        video_data = F.interpolate(video_data, (20, 96, 64), mode="trilinear")[0].permute(1, 2, 3, 0)
        video_data = video_data.numpy().astype("uint8")

        user_data.append(video_data)
    user_data = np.stack(user_data)
    print(user_data.shape)
            
    return user_data


data_root = "data/all"
save_dir = "data/pkl"
normal_sample_frame_num = 2

os.makedirs(save_dir, exist_ok=True)

normal_dir = os.path.join(data_root, "normal")
abnormal_dir = os.path.join(data_root, "abnormal")

for user_name in sorted(os.listdir(normal_dir)):
    user_dir = os.path.join(normal_dir, user_name)
    save_path = os.path.join(save_dir, user_name + "_normal.pkl")

    user_data = load_normal_user_data(user_dir, sample_frame_num=normal_sample_frame_num)

    with open(save_path, "wb") as f:
        pickle.dump(user_data, f)

for user_name in sorted(os.listdir(abnormal_dir)):
    user_dir = os.path.join(abnormal_dir, user_name)
    save_path = os.path.join(save_dir, user_name + "_abnormal.pkl")

    user_data = load_abnormal_user_data(user_dir, sample_frame_num=normal_sample_frame_num)

    with open(save_path, "wb") as f:
        pickle.dump(user_data, f)
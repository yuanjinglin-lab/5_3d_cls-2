import os
import random
import cv2
import numpy as np

fa_dir = r'E:\data\MRI_data\dault_image_crop'
dst_dir = r'E:\data\MRI_data\dault_image_npy'

count = 0
while True:
    if len(os.listdir(dst_dir)) > 2000:
        break

    son_idx = random.randint(0, 57)
    son_dir = fa_dir + '/' + os.listdir(fa_dir)[son_idx]
    son_enhance_dir = fa_dir + '/' + os.listdir(fa_dir)[son_idx] + '/' + 'enhance'
    son_normal_dir = fa_dir + '/' + os.listdir(fa_dir)[son_idx] + '/' + 'normal'
    total_num = len(os.listdir(son_enhance_dir))
    random_idx = random.randint(0, total_num-8)
    ran_list = os.listdir(son_enhance_dir)[random_idx:random_idx+8]
    # print(ran_list)
    enhance_data_list = []
    normal_data_list = []
    for img_name in ran_list:
        img_data_enhance = cv2.imread(son_enhance_dir + '/' + img_name)
        img_data_enhance = np.expand_dims(img_data_enhance, 0)
        enhance_data_list.append(img_data_enhance)
        img_data_normal = cv2.imread(son_normal_dir + '/' + img_name)
        img_data_normal = np.expand_dims(img_data_normal, 0)
        normal_data_list.append(img_data_normal)
    enhance_data_cat = np.concatenate(enhance_data_list, 0)
    normal_data_cat = np.concatenate(normal_data_list, 0)
    # print(enhance_data_cat.shape, normal_data_cat.shape)
    last_cat = np.concatenate([enhance_data_cat, normal_data_cat], 0)
    print(last_cat.shape)
    np.save(dst_dir + '/' + str(count).rjust(5, '0') + '.npy', last_cat)
    count += 1

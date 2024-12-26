import os
import random
import cv2
import numpy as np

fa_dir = r'E:\data\MRI_data\dault_image_crop'
dst_dir = r'E:\data\MRI_data\0105_0161'

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

count = 0
son_enhance_dir = fa_dir + '/0105_0161/' + 'enhance'
son_normal_dir = fa_dir + '/0105_0161/' + 'normal'
total_num = len(os.listdir(son_enhance_dir))
tt = total_num // 8
for i in range(tt):
    ran_list = os.listdir(son_enhance_dir)[i * 8: (i+1) * 8]
    enhance_data_list = []
    normal_data_list = []
    for img_name in ran_list:
        if img_name != 'Thumbs.db':
            print(son_enhance_dir + '/' + img_name)
            img_data_enhance = cv2.imread(son_enhance_dir + '/' + img_name)
            img_data_enhance = np.expand_dims(img_data_enhance, 0)
            print(img_data_enhance.shape)
            enhance_data_list.append(img_data_enhance)
            img_data_normal = cv2.imread(son_normal_dir + '/' + img_name)
            img_data_normal = np.expand_dims(img_data_normal, 0)
            normal_data_list.append(img_data_normal)
    enhance_data_cat = np.concatenate(enhance_data_list, 0)
    normal_data_cat = np.concatenate(normal_data_list, 0)
    last_cat = np.concatenate([enhance_data_cat, normal_data_cat], 0)
    np.save(dst_dir + '/' + str(count).rjust(5, '0') + '.npy', last_cat)
    count += 1

ran_list = os.listdir(son_enhance_dir)[tt * 8:]
if len(ran_list) != 0:
    enhance_data_list = []
    normal_data_list = []
    for img_name in ran_list:
        if img_name != 'Thumbs.db':
            img_data_enhance = cv2.imread(son_enhance_dir + '/' + img_name)
            img_data_enhance = np.expand_dims(img_data_enhance, 0)
            enhance_data_list.append(img_data_enhance)
            img_data_normal = cv2.imread(son_normal_dir + '/' + img_name)
            img_data_normal = np.expand_dims(img_data_normal, 0)
            normal_data_list.append(img_data_normal)
    enhance_data_cat = np.concatenate(enhance_data_list, 0)
    normal_data_cat = np.concatenate(normal_data_list, 0)
    last_cat = np.concatenate([enhance_data_cat, normal_data_cat], 0)
    np.save(dst_dir + '/' + str(count).rjust(5, '0') + '.npy', last_cat)



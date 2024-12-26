import os
import cv2
import random
import numpy as np

## step 1
def video2frame(mri_video_dir, mri_image_dir):
    # mri_video_dir = r'E:\data\MRI_data\dault'
    # mri_image_dir = r'E:\data\MRI_data\dault_image'
    for video_name in os.listdir(mri_video_dir):
        if not os.path.exists(mri_image_dir + '/' + video_name):
            os.mkdir(mri_image_dir + '/' + video_name.split('.')[0])
        cap = cv2.VideoCapture(mri_video_dir + '/' + video_name)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_num == 10:
                    print(frame.shape)
                image_name = str(frame_num).rjust(5, '0') + '.jpg'
                cv2.imwrite(mri_image_dir + '/' + video_name.split('.')[0] + '/' + image_name, frame)
                frame_num += 1
            else:
                break
        cap.release()

## step2
def crop_img(fa_img_dir, save_img_dir):
    # fa_img_dir = r'E:\data\MRI_data\dault_image'
    # save_img_dir = r'E:\data\MRI_data\dault_image_crop'
    for son_img_dir in os.listdir(fa_img_dir):
         if not os.path.exists(save_img_dir+'/'+son_img_dir):
             os.mkdir(save_img_dir+'/'+son_img_dir)
         if not os.path.exists(save_img_dir+'/'+son_img_dir+'/'+'enhance'):
             os.mkdir(save_img_dir+'/'+son_img_dir+'/'+'enhance')
         if not os.path.exists(save_img_dir+'/'+son_img_dir+'/'+'normal'):
             os.mkdir(save_img_dir + '/' + son_img_dir + '/' + 'normal')
         for img_name in os.listdir(fa_img_dir + '/' + son_img_dir):
             img_path = fa_img_dir + '/' + son_img_dir + '/' + img_name
             img_data = cv2.imread(img_path)
             h, w, c = img_data.shape
             left_data = img_data[:, :w//2, :]
             right_data = img_data[:, w//2:, :]
             h_left, w_left, c = left_data.shape
             h_right, w_right, c = right_data.shape

             center_left_x, center_left_y = w//2, h//2
             # print(left_data.shape, center_left_x, center_left_y)
             crop_left_data = left_data[center_left_y - int(center_left_x * 0.4): center_left_y + int(center_left_x * 0.4),
                                        center_left_x - int(center_left_x * 0.4): center_left_x + int(center_left_x * 0.4), :]
             center_right_x, center_right_y = w//2, h//2
             crop_right_data = right_data[center_right_y - int(center_right_x * 0.4): center_right_y + int(center_right_x * 0.4),
                                        center_right_x - int(center_right_x * 0.4): center_right_x + int(center_right_x * 0.4), :]

             # (crop_left_data.shape, crop_right_data.shape)
             crop_left_data = cv2.resize(crop_left_data, (128, 256))
             crop_right_data = cv2.resize(crop_right_data, (128, 256))
             cv2.imwrite(save_img_dir + '/' + son_img_dir + '/' + 'enhance' + '/' + img_name, crop_left_data)
             cv2.imwrite(save_img_dir + '/' + son_img_dir + '/' + 'normal' + '/' + img_name, crop_right_data)


## step3 ehance img
def ehance_img(fa_dir,dst_dir):
    # fa_dir = r'E:\data\MRI_data\dault_image_crop'
    # dst_dir = r'E:\data\MRI_data\dault_image_npy'
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
## step4
def split_img_npy(fa_dir,dst_dir):
    # fa_dir = r'E:\data\MRI_data\dault_image_crop'
    # dst_dir = r'E:\data\MRI_data\0105_0161'

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    count = 0
    son_enhance_dir = fa_dir + '/add_data/' + 'enhance'
    son_normal_dir = fa_dir + '/add_data/' + 'normal'

    if not os.path.exists(son_enhance_dir):
        os.makedirs(son_enhance_dir)
    if not os.path.exists(son_normal_dir):
        os.makedirs(son_normal_dir)

    total_num = len(os.listdir(son_enhance_dir))
    num = 8
    tt = total_num // num
    for i in range(tt):
        ran_list = os.listdir(son_enhance_dir)[i * num: (i+1) * num]
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

    ran_list = os.listdir(son_enhance_dir)[tt * num:]
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

if __name__ == '__main__':
    mri_video_dir = None # r'E:\data\MRI_data\dault'
    mri_image_dir = None # r'E:\data\MRI_data\dault_image'

    ## 1 video 2 img
    video2frame(mri_video_dir,mri_image_dir)

    ## 2 crop
    crop_img_dir = None # r'E:\data\MRI_data\dault_image_crop'
    crop_img(mri_image_dir, crop_img_dir)

    ## 3
    ehance_img_dst = None # r'E:\data\MRI_data\dault_image_npy'
    ehance_img(crop_img_dir, ehance_img_dst)

    ## 4 get npy
    split_dst = None # r'E:\data\MRI_data\0105_0161'
    split_img_npy(ehance_img_dst, split_dst)


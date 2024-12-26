import os
import cv2

fa_img_dir = r'E:\data\MRI_data\dault_image'
save_img_dir = r'E:\data\MRI_data\dault_image_crop'

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


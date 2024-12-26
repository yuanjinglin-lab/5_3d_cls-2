import cv2
import os

mri_video_dir = r'E:\data\MRI_data\dault'
mri_image_dir = r'E:\data\MRI_data\dault_image'

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

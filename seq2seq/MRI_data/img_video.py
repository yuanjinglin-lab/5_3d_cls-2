import cv2
import os
from PIL import Image

def Pic2Video(imgPath, videoPath):
    # imgPath = "youimgPath"  # 读取图片路径
    # videoPath = "youvideoPath"  # 保存视频路径
 
    images = os.listdir(imgPath)
    fps = 15  # 每秒25帧数
 
    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
 
    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
        print(im_name)
        videoWriter.write(frame)
    print("图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    videoPath = "result_images/enhance2normal/sr.mp4"
    imgPath = 'result_images/enhance2normal/sr/'

    Pic2Video(imgPath, videoPath)

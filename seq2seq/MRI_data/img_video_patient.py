import numpy as np

from models.seq2seq.seq2seq import Generator
import yaml
import os
from dataloader.mydata import *
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


with open('config_examples/my_data.yaml', 'r') as f:
    config = yaml.safe_load(f)

net = Generator(config)
net.cuda()
net.load_state_dict(torch.load('ckpt/seq2seq/brats/2d/ckpt_best.pth'))

data_path = r'E:\data\MRI_data\0105_0161'
data_list = os.listdir(data_path)

train_data = MyDataset(data_path, data_list, get_valid_transforms())

num_id = 0
for t in range(len(data_list)):
    data = train_data.__getitem__(t)

    source_seq = data[0]
    target_seq = data[1]

    source_img = source_seq.cuda()
    target_img = target_seq.cuda()
    # print(source_img.shape, target_img.shape)
    source_img = source_img.unsqueeze(0)
    target_img = target_img.unsqueeze(0)

    c_s = 64
    target_code = torch.from_numpy(np.ones((1, c_s))).to(device='cuda:0', dtype=torch.float32)
    output_target = net(source_img, target_code, n_outseq=target_img.shape[1])
    print(output_target.shape)

    output_target = output_target.squeeze()
    output_target = output_target.permute(0, 3, 2, 1)

    for i in range(output_target.shape[0]):
        predict = output_target[i]
        #print(predict.shape)
        predict = predict.detach().cpu().numpy()[..., 0]
        predict = predict * 255.
        predict = np.clip(predict, 0 , 255)
        predict[np.where(predict > 220.)] = 0.
        predict = predict.astype(np.uint8)
        cv2.imwrite('result_images/0105_0161/pred/' + str(num_id).rjust(5, '0') + '.jpg', predict)

    target_img = target_img.squeeze()
    target_img = target_img.permute(0, 3, 2, 1)
    for i in range(target_img.shape[0]):
        gt = target_img[i]
        print(gt.shape)
        gt = gt.detach().cpu().numpy()[..., 0]
        gt = gt * 255.
        gt = gt.astype(np.uint8)
        cv2.imwrite('result_images/0105_0161/gt/gt_' + str(num_id).rjust(5, '0') + '.jpg', gt)

    source_img = source_img.squeeze()
    source_img = source_img.permute(0, 3, 2, 1)
    for i in range(source_img.shape[0]):
        sr = source_img[i]
        print(sr.shape)
        sr = sr.detach().cpu().numpy()[..., 0]
        sr = sr * 255.
        sr = sr.astype(np.uint8)
        cv2.imwrite('result_images/0105_0161/sr/sr_' + str(num_id).rjust(5, '0') + '.jpg', sr)

    num_id += 1


videoPath_sr = "result_images/0105_0161/sr.mp4"
imgPath_sr = 'result_images/0105_0161/sr/'
Pic2Video(imgPath_sr, videoPath_sr)

videoPath_pred = "result_images/0105_0161/pred.mp4"
imgPath_pred = 'result_images/0105_0161/pred/'
Pic2Video(imgPath_pred, videoPath_pred)

videoPath_gt = "result_images/0105_0161/gt.mp4"
imgPath_gt = 'result_images/0105_0161/gt/'
Pic2Video(imgPath_gt, videoPath_gt)
from models.seq2seq.seq2seq import Generator
import yaml
import os
from dataloader.mydata import *
with open('config_examples/my_data.yaml', 'r') as f:
    config = yaml.safe_load(f)

net = Generator(config)
net.cuda()
net.load_state_dict(torch.load('ckpt/seq2seq/brats/2d/ckpt_best.pth'))

data_path = r'E:\data\MRI_data\dault_image_npy'
data_list = os.listdir(data_path)[:200]

train_data = MyDataset(data_path, data_list, get_valid_transforms())

data = train_data.__getitem__(0)

print(data[0].shape, data[1].shape)

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
    print(predict.shape)
    predict = predict.detach().cpu().numpy()[..., 0]
    predict = predict * 255.
    predict = predict.astype(np.uint8)
    cv2.imwrite('result_images/' + str(i) + '.jpg', predict)

target_img = target_img.squeeze()
target_img = target_img.permute(0, 3, 2, 1)
for i in range(target_img.shape[0]):
    gt = target_img[i]
    print(gt.shape)
    gt = gt.detach().cpu().numpy()[..., 0]
    gt = gt * 255.
    gt = gt.astype(np.uint8)
    cv2.imwrite('result_images/gt_' + str(i) + '.jpg', gt)

source_img = source_img.squeeze()
source_img = source_img.permute(0, 3, 2, 1)
for i in range(source_img.shape[0]):
    sr = source_img[i]
    print(sr.shape)
    sr = sr.detach().cpu().numpy()[..., 0]
    sr = sr * 255.
    sr = sr.astype(np.uint8)
    cv2.imwrite('result_images/sr_' + str(i) + '.jpg', sr)
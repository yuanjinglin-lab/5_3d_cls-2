import argparse
import os
import sys
import logging
import yaml
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

sys.path.append('./src/')
sys.path.append('../../mri_seq2seq')

from utils import poly_lr, Recorder, Plotter, save_grid_images, torch_PSNR, torch_SSIM
from losses import PerceptualLoss
from dataloader.mydata import MyDataset, get_train_transforms, get_valid_transforms
from models.seq2seq.seq2seq import Generator


def train(args, net, device):
    data_path = r'../MRI_data/dault_image_npy'
    data_list = os.listdir(data_path)[:200]

    train_data = MyDataset(data_path, data_list, get_train_transforms())
    valid_data = MyDataset(data_path, data_list, get_valid_transforms())

    n_train = len(train_data)  # len(dataset) - n_val
    n_valid = len(valid_data)  # len(dataset) - n_val

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    c_in = args['seq2seq']['c_in']
    c_s = args['seq2seq']['c_s']
    epochs = args['train']['epochs']
    lr = np.float32(args['train']['lr'])
    dir_visualize = args['train']['vis']
    dir_checkpoint = args['train']['ckpt']
    lambda_rec = args['train']['lambda_rec']
    lambda_per = args['train']['lambda_per']
    lambda_cyc = args['train']['lambda_cyc']

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {1}
        Learning rate:   {lr}
        Training size:   {n_train}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-6) / lr)
    perceptual = PerceptualLoss().to(device=device)

    recorder = Recorder(['train_loss', 'psnr', 'ssim'])
    plotter = Plotter(dir_visualize, keys1=['train_loss'], keys2=['psnr'])

    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,train_loss,psnr\n')

    total_step = 0
    best_psnr = 0
    best_ssim = 0
    nan_times = 0
    for epoch in range(epochs):
        if epoch != 0:
            scheduler.step()
        net.train()
        train_losses = []
        for batch in tqdm(train_loader, total=len(train_loader)):
            source_seq = batch[0]
            target_seq = batch[1]

            source_img = source_seq.cuda()
            target_img = target_seq.cuda()

            nw, nh = source_img.shape[-2], source_img.shape[-1]

            # source_code = torch.from_numpy(np.array([1 if i == source_seq else 0 for i in range(c_s)])).reshape(
            #     (1, c_s)).to(device=device, dtype=torch.float32)
            # target_code = torch.from_numpy(np.array([1 if i == target_seq else 0 for i in range(c_s)])).reshape(
            #     (1, c_s)).to(device=device, dtype=torch.float32)

            source_code = torch.from_numpy(np.ones((1, c_s))).to(device=device, dtype=torch.float32)
            target_code = torch.from_numpy(np.ones((1, c_s))).to(device=device, dtype=torch.float32)

            output_source = net(source_img, source_code, n_outseq=source_img.shape[1])
            output_target = net(source_img, target_code, n_outseq=target_img.shape[1])
            output_cyc = net(output_target, source_code, n_outseq=source_img.shape[1])

            loss_rec = nn.SmoothL1Loss()(output_target, target_img) + nn.SmoothL1Loss()(output_source,
                                                                                        source_img)
            loss_cyc = nn.SmoothL1Loss()(output_cyc, source_img)
            loss_per = perceptual(output_target[0].reshape(-1, 1, nw, nh) / 2 + 0.5,
                                  target_img[0].reshape(-1, 1, nw, nh) / 2 + 0.5) + \
                       perceptual(output_source[0].reshape(-1, 1, nw, nh) / 2 + 0.5,
                                  source_img[0].reshape(-1, 1, nw, nh) / 2 + 0.5)

            loss = lambda_rec * loss_rec + lambda_per * loss_per + lambda_cyc * loss_cyc

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2)
            optimizer.step()

            train_losses.append(loss_rec.item())

            if (total_step % args['train']['ckpt_steps']) == 0:
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))

            total_step += 1
            if total_step > args['train']['total_steps']:
                # torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
                return
        print(sum(train_losses), len(train_losses))
        print('train loss epoch: ', sum(train_losses)/len(train_losses))
        # break

        net.eval()
        valid_psnrs = []
        valid_ssim = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in valid_loader:
                source_seq = batch[0]
                target_seq = batch[1]

                source_img = source_seq.cuda()
                target_img = target_seq.cuda()


                # target_code = torch.from_numpy(np.array([1 if i == target_seq else 0 for i in range(c_s)])).reshape(
                #     (1, c_s)).to(device=device, dtype=torch.float32)
                target_code = torch.from_numpy(np.ones((1, c_s))).to(device=device, dtype=torch.float32)

                output_target = net(source_img, target_code, n_outseq=target_img.shape[1])
                psnr = torch_PSNR(output_target, target_img, data_range=2.).item()
                ssim = torch_SSIM(output_target[0], target_img[0]).item()
                valid_psnrs.append(psnr)
                valid_ssim.append(ssim)
                # break

        valid_psnrs = np.mean(valid_psnrs)
        valid_ssims = np.mean(valid_ssim)
        train_losses = np.mean(train_losses)
        recorder.update({'train_loss': train_losses, 'psnr': valid_psnrs, 'ssim': valid_ssims})
        plotter.send(recorder.call())
        if best_psnr < valid_psnrs:
            best_psnr = valid_psnrs
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{},{},{},{}\n'.format(epoch + 1, train_losses, valid_psnrs, valid_ssims))
        # torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG on images and target label',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='config_examples/my_data.yaml',
                        help='config file')
    parser.add_argument('-l', '--load', dest='load', type=str, default='ckpt/seq2seq_brats_2d_missing.pth', # 'ckpt/seq2seq_brats_2d_missing.pth'
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cuda:0',
                        help='cuda or cpu')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dir_checkpoint = config['train']['ckpt']
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    dir_visualize = config['train']['vis']
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = Generator(config)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)

    try:
        train(
            config,
            net=net,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
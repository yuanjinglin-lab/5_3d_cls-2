import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from encoder import Encoder
from convlstm import ConvLSTM2d, ConvLSTM3d


class Resnet18(nn.Module):
    def __init__(self, in_channel=12, pretrained=False):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(norm_layer=nn.InstanceNorm2d, pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x


class Model(nn.Module):

    def __init__(self, ndims=2, c_in=2, c_enc=[64, 128, 256], k_enc=[7, 3, 3], 
            s_enc=[1, 2, 2], nres_enc=6, norm="InstanceNorm", num_classes=2):
        super(Model, self).__init__()
        self.ndims = ndims
        self.c_in = c_in
        self.c_enc = c_enc
        self.k_enc = k_enc
        self.s_enc = s_enc
        self.nres_enc = nres_enc

        # self.encoder = Encoder(ndims=ndims, c_in=c_in, c_enc=c_enc, 
        #     k_enc=k_enc, s_enc=s_enc, nres_enc=nres_enc, norm=norm)
        self.encoder = Resnet18(in_channel=c_in)

        assert ndims in [2, 3]
        ConvLSTM = ConvLSTM2d if ndims == 2 else ConvLSTM3d
        conv_lstm_dim = 512#c_enc[-1]
        self.conv_lstm = ConvLSTM(
            input_dim=conv_lstm_dim,
            hidden_dim=[conv_lstm_dim, conv_lstm_dim, conv_lstm_dim],
            kernel_size=(3, 3) if self.ndims == 2 else (3, 3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        self.fc = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, input):
        B, C, T, H, W = input.shape
        input = input.permute(0, 2, 1, 3, 4)
        input = input.reshape(B * T, C, H, W)
        x = self.encoder(input)
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        # x = x.permute(0, 2, 1, 3, 4)
        # B, T, C, H, W
        x = self.conv_lstm(x)[0][0]
        x = torch.mean(x, dim=1)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)

        return out


if __name__ == "__main__":
    model = Model(ndims=2, c_in=12, c_enc=[64, 128, 256], k_enc=[7, 3, 3], 
            s_enc=[1, 2, 2], nres_enc=6, norm="InstanceNorm", num_classes=2)
    x = torch.zeros(2, 12, 20, 64, 64)
    out = model(x)
    print(out.shape)
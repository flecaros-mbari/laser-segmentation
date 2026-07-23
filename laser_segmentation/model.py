"""U-Net architecture (encoder/decoder with skip connections)."""

import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Module, ModuleList, ReLU
from torchvision.transforms import CenterCrop


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.enc_blocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        for block in self.enc_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]
        )
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, enc_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    @staticmethod
    def crop(enc_features, x):
        (_, _, h, w) = x.shape
        return CenterCrop([h, w])(enc_features)


class UNet(Module):
    def __init__(
        self,
        enc_channels=(3, 16, 32, 64),
        dec_channels=(64, 32, 16),
        num_classes=1,
        retain_dim=True,
        out_size=(1080, 1920),
    ):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.head = Conv2d(dec_channels[-1], num_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.head(dec_features)

        if self.retain_dim:
            out = F.interpolate(out, self.out_size)
        return out

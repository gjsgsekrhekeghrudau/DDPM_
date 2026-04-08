import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, unet, num_classes):
        super().__init__()
        self.unet = unet
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Flatten(),

            nn.Linear(12288, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    @torch.no_grad()
    def dift(self, x, t):
        t_emb = self.unet.time_mlp(self.unet.time_embedding(t))

        x1 = self.unet.enc1(x, t_emb)
        x2 = self.unet.enc2(self.unet.downsample(x1), t_emb)
        x2 = self.unet.attn2(x2)
        x3 = self.unet.enc3(self.unet.downsample(x2), t_emb)
        x3 = self.unet.attn3(x3)

        b = self.unet.bottleneck(self.unet.downsample(x3), t_emb)
        b = self.unet.attn_bottleneck(b)

        b = self.pool(b)
        b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-6)
        return b

    def forward(self, x, t):
        x = self.dift(x, t)
        return self.main(x)

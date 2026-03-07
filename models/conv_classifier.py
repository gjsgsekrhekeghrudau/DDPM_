import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, unet, num_classes):
        super().__init__()
        self.unet = unet
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(128, num_classes)
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
        return self.head(x)

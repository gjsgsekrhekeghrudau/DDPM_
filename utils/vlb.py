from models import attention_unet, diffusion
from datasets_.val_dataset import ImageDataset

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def VLB(diff, model, dataloader, device):
    model.eval()
    total_vlb = 0.0
    total_images = 0

    T = diff.diffusion_steps

    for x, _ in tqdm(dataloader):
        x = x.to(device)
        b = x.size(0)

        t = torch.randint(1, T, (b,), device=device)

        noise = torch.randn_like(x)
        x_t, noise = diff.noise_images(x, t, noise)

        pred_noise = model(x_t, t)

        beta_t = diff.beta[t][:, None, None, None]
        alpha_t = diff.alpha[t][:, None, None, None]
        alpha_hat_t = diff.alpha_hat[t][:, None, None, None]

        sigma2 = beta_t

        weight = (beta_t**2) / (2 * sigma2 * alpha_t * (1 - alpha_hat_t))

        mse = F.mse_loss(pred_noise, noise, reduction='none')
        mse = mse.sum(dim=(1, 2, 3))

        kl = weight.view(b) * mse

        vlb = T * kl

        total_vlb += vlb.sum().item()
        total_images += b

    return total_vlb / total_images


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 64
TIME_DIM = 256
DIFFUSION_STEPS = 1000

dataset = ImageDataset(IMG_SIZE, num_classes=100, dataset_path='../datasets_/archive')
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

vlbs = []
for epoch in range(1, 100):
    MODEL_PATH = f'../trained_models/ddpm_epoch{epoch}.pth'

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])

    torch.manual_seed(0)
    vlb = VLB(diff, model, dataloader, DEVICE)
    vlbs.append(vlb)

torch.save({'vlbs': torch.FloatTensor(vlbs)}, 'files/vlb.pth')

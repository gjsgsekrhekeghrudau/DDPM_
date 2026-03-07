from models import attention_unet, diffusion

import torch


def GL(samples, dataset_images):
    samples = samples.view(samples.size(0), -1)
    dataset_images = dataset_images.view(dataset_images.size(0), -1)

    dist = torch.cdist(samples, dataset_images)

    nn_dist = dist.min(dim=1).values
    norms = torch.norm(samples, dim=1)

    return torch.mean(nn_dist / norms)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = f'../trained_models/ddpm_epoch99.pth'
IMG_SIZE = 64
TIME_DIM = 256
DIFFUSION_STEPS = 1000

dataset_images = torch.load('../datasets_/train.pth', map_location=DEVICE)['data']

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

samples = diff.sample(model, 10)
print(GL(samples, dataset_images))

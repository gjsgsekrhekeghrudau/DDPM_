from models import attention_unet, diffusion
from metrics.fid import calculate_fid
from datasets_.train_dataset import ImageDataset

import torch
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 97
MODEL_PATH = f'../trained_models/ddpm_epoch{EPOCH}.pth'
IMG_SIZE = 64
TIME_DIM = 256
DIFFUSION_STEPS = 1000
NUM_SAMPLES = 128

dataset = ImageDataset(IMG_SIZE, num_classes=100, dataset_path='../datasets_/archive')
dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=True, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

samples = diff.sample(model, NUM_SAMPLES)

for real, _ in dataloader:
    fid = calculate_fid(real, samples, DEVICE)
    print(fid)
    break

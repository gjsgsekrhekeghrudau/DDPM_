from models import attention_unet, diffusion, classifier1
from datasets_.train_dataset import ImageDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# lss = torch.load('../utils/files/ls.pth', map_location=DEVICE)['lss']
# EPOCH = torch.argmin(lss)  # Diffusion epoch with min LS
# EPOCH = torch.argmax(lss)  # Diffusion epoch with max LS
EPOCH = 199
# MODEL_PATH = f'../trained_models/ddpm_epoch{EPOCH}.pth'
MODEL_PATH = f'../overfit/ddpm_epoch{EPOCH}.pth'

IMG_SIZE = 64
TIME_DIM = 256
T = 750

EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

DIFFUSION_STEPS = 1000

dataset = ImageDataset(IMG_SIZE, num_classes=3, dataset_path='../datasets_/archive')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

classifier = classifier1.Classifier(model, num_classes=len(dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

loss_history = []
report = {}

for epoch in range(EPOCHS):
    losses = []
    total_loss = 0

    progress = tqdm(dataloader, desc=f'Epoch {epoch}')

    for imgs, labels in progress:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        t = torch.full((BATCH_SIZE, ), T, device=DEVICE)

        noisy, _ = diff.noise_images(imgs, t)
        pred = classifier(noisy, t)

        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

        report['loss'] = losses[-1]
        progress.set_postfix(report)

    loss_history.append(losses)

    report['avg_loss'] = total_loss / len(dataloader)

    torch.save({
        'model_state': classifier.state_dict(),
        'loss_history': loss_history
    }, f'classifiers/t={T}.pth')

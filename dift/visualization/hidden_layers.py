from models import attention_unet, diffusion, classifier2
from datasets_.train_dataset import ImageDataset

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 199
MODEL_PATH = f'../../overfit/ddpm_epoch{EPOCH}.pth'
CLASSIFIER_PATH = f'../classifiers/conv_classifier_{EPOCH}.pth'

IMG_SIZE = 64
TIME_DIM = 256
T = 20

EPOCHS = 10
LEARNING_RATE = 1e-3

DIFFUSION_STEPS = 1000

dataset = ImageDataset(IMG_SIZE, num_classes=3, dataset_path='../../datasets_/archive')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

classifier = classifier2.Classifier(model, num_classes=len(dataset.classes)).to(DEVICE)
checkpoint = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
classifier.load_state_dict(checkpoint['model_state'])
classifier.eval()

for imgs, labels in dataloader:
    imgs = imgs.to(DEVICE)
    labels = labels.to(DEVICE)
    t = torch.full((1, ), T, device=DEVICE)

    with torch.no_grad():
        noisy, _ = diff.noise_images(imgs, t)
        dift = classifier.dift(noisy, t)
        hidden = classifier.main[:16](dift)

        from torchvision.utils import save_image
        save_image(hidden, 'hidden_activation.png', normalize=True)

        hidden = hidden.squeeze(0).permute(1, 2, 0).numpy()

    plt.imshow((hidden - hidden.min()) / (hidden.max() - hidden.min()))
    plt.show()

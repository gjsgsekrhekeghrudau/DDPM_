import matplotlib.pyplot as plt
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 97
CLASSIFIER_PATH = f'classifiers/on_epoch_{EPOCH}.pth'

loss_history = torch.load(CLASSIFIER_PATH)['loss_history']
plt.title(f'Classifier Loss History (on Diffusion Epoch {EPOCH})', fontweight='bold', rotation=0)
plt.xlabel('Epoch', rotation=0)
plt.ylabel('Loss', rotation=0)
for i in range(len(loss_history)):
    epoch_losses = loss_history[i]
    plt.plot(
        torch.arange(i * len(epoch_losses), (i + 1) * len(epoch_losses)),
        epoch_losses,
        label=f'Epoch {i}'
    )
plt.legend()
plt.grid()
plt.savefig(f'classifier_losses_on_diffusion_epoch_{EPOCH}.png')
plt.show()

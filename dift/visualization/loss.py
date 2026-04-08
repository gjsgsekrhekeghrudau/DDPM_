import matplotlib.pyplot as plt
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

T = 750
CLASSIFIER_PATH = f'../classifiers/t={T}.pth'

loss_history = torch.load(CLASSIFIER_PATH)['loss_history']
plt.title(f'Classifier Loss History (on T={T})', fontweight='bold', rotation=0)
plt.xlabel('Iteration', rotation=0)
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
plt.savefig(f'../pictures/classifier_losses_on_t={T}.png')
plt.show()

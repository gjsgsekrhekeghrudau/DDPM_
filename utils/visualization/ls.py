import matplotlib.pyplot as plt
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

lss = torch.load('../files/ls_at_high_t.pth', map_location=DEVICE)['lss']
plt.title('LS depending on epoch', fontweight='bold', rotation=0)
plt.xlabel('Diffusion epoch', rotation=0)
plt.ylabel('LS', rotation=0)
plt.plot(lss)
plt.grid()
plt.savefig('../pictures/ls_at_high_t.png')
plt.show()

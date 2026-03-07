import matplotlib.pyplot as plt
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

lss = torch.load('../files/vlb.pth', map_location=DEVICE)['vlb']
plt.title('VLB depending on epoch', fontweight='bold', rotation=0)
plt.xlabel('Diffusion epoch', rotation=0)
plt.ylabel('VLB', rotation=0)
plt.plot(lss)
plt.grid()
plt.savefig('../pictures/vlb.png')
plt.show()

import torch
import os
from torchvision.utils import save_image
import ssl
import io
import contextlib
from pytorch_fid import fid_score


class FID:
    def __init__(self, device: torch.device):
        ssl._create_default_https_context = ssl._create_unverified_context
        self.device = device

    def __call__(self, real_images, fake_images):
        real_dir = 'temp_real'
        fake_dir = 'temp_fake'
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        self._save_tensors_as_images(real_images, real_dir)
        self._save_tensors_as_images(fake_images, fake_dir)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir],
                batch_size=50,
                device=self.device,
                dims=2048,
                num_workers=0
            )

        self._cleanup_temp_dirs(real_dir, fake_dir)

        return fid_value

    def _save_tensors_as_images(self, tensors, save_dir):
        for i, tensor in enumerate(tensors):
            if tensor.min() < 0:
                img = (tensor + 1) / 2
            else:
                img = tensor
            img = img.clamp(0, 1)
            save_image(img, os.path.join(save_dir, f'image_{i:06d}.png'))

    def _cleanup_temp_dirs(self, *dirs):
        import shutil
        for dir_path in dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)


def calculate_fid(real, fake, device):
    fid_calculator = FID(device)
    fid_score = fid_calculator(real.detach(), fake.detach())
    return fid_score

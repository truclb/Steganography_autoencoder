import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. Định nghĩa Dataset tùy chỉnh
class ECGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for filename in os.listdir(root_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# 2. Định nghĩa Autoencoder
class ECGAutoencoder(nn.Module):
    def __init__(self):
        super(ECGAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8 -> 4
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='nearest') # 32 -> 256
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
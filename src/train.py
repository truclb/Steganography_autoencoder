import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==== Thiết lập ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MSG_LEN = 256  # tối đa 32 ký tự

# ==== Hàm chuyển đổi văn bản <-> tensor ảnh ====
def text_to_binary(text):
    bits = ''.join([format(byte, '08b') for byte in text.encode('utf-8')])  # Dùng UTF-8 thay vì ASCII
    bits = bits.ljust(MAX_MSG_LEN * 8, '0')  # pad cho đủ độ dài cố định
    return bits

def binary_to_text(bits):
    byte_array = bytearray([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
    return byte_array.decode('utf-8', errors="ignore")  # Dùng UTF-8

def text_to_tensor(text, H, W):
    bits = text_to_binary(text)
    arr = np.array([int(b) for b in bits], dtype=np.float32)
    arr = np.resize(arr, (H, W))  # reshape thành ảnh
    return torch.tensor(arr).unsqueeze(0)  # (1, H, W)

def tensor_to_text(tensor):
    bits = tensor.detach().cpu().numpy().flatten()
    bits = ['1' if b > 0.5 else '0' for b in bits[:MAX_MSG_LEN * 8]]
    return binary_to_text(''.join(bits))

# ==== Mô hình Encoder ====
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def forward(self, image, message):
        combined = torch.cat([image, message], dim=1)
        encoded_image = self.encoder(combined)
        return encoded_image

# ==== Mô hình Decoder ====
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, encoded_image):
        return self.decoder(encoded_image)
    
# ==== Dataset ====


class CocoStuffDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        try:
            img = Image.open(img_path).convert("RGB")  # Đảm bảo ảnh không bị lỗi
        except Exception as e:
            print(f"❌ Lỗi khi tải ảnh: {img_path} | {e}")
            return None, None  # Tránh lỗi batch

        if self.transform:
            img = self.transform(img)
        return img, 0  # Trả về ảnh + nhãn giả (để giữ cú pháp CIFAR)

# Định nghĩa dataset CocoStuff
img_dir = r"F:\Master_Course\HK1\AnThongTin\DoAnCuoiKy\Dataset\images"
transform = transforms.Compose([
    transforms.Resize((256,256)),  # Resize về kích thước 32x32 như CIFAR
    transforms.ToTensor(),        # Chuyển ảnh sang tensor (C, H, W) ∈ [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa ảnh
])

trainset = CocoStuffDataset(img_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

# Kiểm tra batch đầu tiên
for images, _ in trainloader:
    print(f"✅ Batch hợp lệ, shape: {images.shape}")
    break

encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.BCELoss()

# ==== Huấn luyện ====
alpha = 0.90  # ưu tiên độ trung thực ảnh
for epoch in range(3):
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{3}", unit="batch")
    total_loss_epoch = 0.0
    for images, _ in progress_bar:
        images = images.to(device)
        B, C, H, W = images.shape

        texts = ["Xin chào @ Stegano! Hồ Hải Triều 16/05/1981. Chúc Bạn thành công"] * B
        messages = torch.stack([text_to_tensor(txt, H, W) for txt in texts]).to(device)

        optimizer.zero_grad()
        encoded_images = encoder(images, messages)
        decoded_messages = decoder(encoded_images)

        loss_msg = criterion(decoded_messages, messages)
        loss_img = F.mse_loss(encoded_images, images)
        loss = loss_msg + alpha * loss_img

        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss_epoch / len(trainloader)
    print(f"Epoch [{epoch+1}/{3}] - Avg Loss: {avg_loss:.4f}")    
    print(f"Epoch {epoch+1} | Loss msg: {loss_msg.item():.4f} | Loss img: {loss_img.item():.4f}")
print("Training complete!")
torch.save(encoder.state_dict(), '.\Deploy\Model\Save_Model\encoder_v2.pth')
torch.save(encoder.state_dict(), '.\Deploy\Model\Save_Model\dencoder_v2.pth')



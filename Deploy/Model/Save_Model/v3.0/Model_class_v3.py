import zlib, os
import torch
import torch.nn as nn
import torch.optim as optim
from math import exp
from reedsolo import RSCodec
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
# ---------------------------------------------------------------------
# Utility Class (unchanged)
class SteganographyUtils:
    def __init__(self, max_msg_len=256):
        self.MAX_MSG_LEN = max_msg_len  # Độ dài tối đa của tin nhắn (tính theo ký tự)

    def text_to_binary(self, text):
        bits = ''.join([format(byte, '08b') for byte in text.encode('utf-8')])
        bits = bits.ljust(self.MAX_MSG_LEN * 8, '0')  # Pad đến độ dài cố định
        return bits

    def binary_to_text(self, bits):
        byte_array = bytearray([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
        return byte_array.decode('utf-8', errors='ignore')

    def text_to_tensor(self, text, H, W):
        bits = self.text_to_binary(text)
        arr = np.array([int(b) for b in bits], dtype=np.float32)
        arr = np.resize(arr, (H, W))  # reshape thành ảnh
        return torch.tensor(arr).unsqueeze(0)  # (1, H, W)

    def tensor_to_text(self, tensor):
        bits = tensor.detach().cpu().numpy().flatten()
        bits = ['1' if b > 0.5 else '0' for b in bits[:self.MAX_MSG_LEN * 8]]
        return self.binary_to_text(''.join(bits))
# ---------------------------------------------------------------------

# ==== Mô hình Encoder ====
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
# === ENCODER ===
        self.conv1 = self.conv_block(4, 64)  # 4 channels: image (3) + message (1)
        self.pool1 = nn.MaxPool2d(2)         # giảm kích thước ảnh xuống H/2

        self.conv2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)         # giảm kích thước ảnh xuống H/4

        self.conv3 = self.conv_block(128, 256)  # Tầng sâu nhất

        # === DECODER ===
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # H/2
        self.dec2 = self.conv_block(256, 128)  # Gộp skip connection

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # H
        self.dec1 = self.conv_block(128, 64)   # Gộp skip connection

        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)  # đầu ra ảnh RGB
        

    def conv_block(self, in_ch, out_ch):
        """ Tầng Conv chuẩn với ReLU activation """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, image, message):
        """ Chạy dữ liệu qua Encoder và Decoder với Skip Connections """
        x = torch.cat([image, message], dim=1)  # Kết hợp ảnh và thông điệp (B, 4, H, W)

        # === ENCODER ===
        e1 = self.conv1(x)          # (B, 64, H, W)
        p1 = self.pool1(e1)         # (B, 64, H/2, W/2)

        e2 = self.conv2(p1)         # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)         # (B, 128, H/4, W/4)

        e3 = self.conv3(p2)         # (B, 256, H/4, W/4)

        # === DECODER ===
        d2 = self.up2(e3)           # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection: concat với đặc trưng từ encoder
        d2 = self.dec2(d2)

        d1 = self.up1(d2)           # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection: concat với đặc trưng từ encoder
        d1 = self.dec1(d1)

        encoded_image = self.out_conv(d1)     # (B, 3, H, W)
        return encoded_image
        
# ==== Mô hình Decoder ====
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # === ENCODER === (giống với Encoder nhưng chỉ nhận ảnh đầu vào)
        self.conv1 = self.conv_block(3, 64)  # 3 channels: RGB image
        self.pool1 = nn.MaxPool2d(2)         # giảm kích thước ảnh xuống H/2

        self.conv2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)         # giảm kích thước ảnh xuống H/4

        self.conv3 = self.conv_block(128, 256)  # Tầng sâu nhất

        # === DECODER ===
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # H/2
        self.dec2 = self.conv_block(256, 128)  # Gộp skip connection

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # H
        self.dec1 = self.conv_block(128, 64)   # Gộp skip connection

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)  # đầu ra thông điệp (1 channel)
        self.sigmoid = nn.Sigmoid()  # đảm bảo đầu ra trong khoảng [0,1]

    def conv_block(self, in_ch, out_ch):
        """ Tầng Conv chuẩn với ReLU activation (giống Encoder) """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, encoded_image):
        """ Chạy dữ liệu qua Encoder và Decoder với Skip Connections """
        # === ENCODER ===
        e1 = self.conv1(encoded_image)      # (B, 64, H, W)
        p1 = self.pool1(e1)                 # (B, 64, H/2, W/2)

        e2 = self.conv2(p1)                 # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)                 # (B, 128, H/4, W/4)

        e3 = self.conv3(p2)                 # (B, 256, H/4, W/4)

        # === DECODER ===
        d2 = self.up2(e3)                   # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)     # Skip connection
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                   # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)     # Skip connection
        d1 = self.dec1(d1)

        decoded_message = self.sigmoid(self.out_conv(d1))  # (B, 1, H, W)
        return decoded_message
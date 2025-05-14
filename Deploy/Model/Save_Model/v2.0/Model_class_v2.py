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
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64
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
    # Hàm mã hóa AES
def aes_encrypt(plaintext, key="abc"):
        # Đảm bảo key có độ dài 32 byte (AES-256)
        key = key.ljust(32)[:32].encode('utf-8')
        iv = get_random_bytes(16)  # Khởi tạo vector ngẫu nhiên
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode('utf-8'), AES.block_size))
        return base64.b64encode(iv + ciphertext).decode('utf-8')  # Trả về base64 để dễ lưu trữ

    # Hàm giải mã AES
def aes_decrypt(encrypted_text, key="abc"):
        key = key.ljust(32)[:32].encode('utf-8')
        encrypted_data = base64.b64decode(encrypted_text)
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return plaintext.decode('utf-8')
# ---------------------------------------------------------------------
def calculate_actual_bpp(message_tensor: torch.Tensor, image_tensor: torch.Tensor) -> float:
    """
    Tính bpp thực tế từ message_tensor và image_tensor.

    Args:
        message_tensor: Tensor chứa dữ liệu nhúng, shape (B, N, H, W) — thường N = 1.
        image_tensor: Tensor ảnh gốc, shape (B, 3, H, W).

    Returns:
        float: Giá trị bpp trung bình thực tế.
    """
    batch_size = image_tensor.size(0)
    _, _, H, W = image_tensor.shape

    # Tổng số bit nhúng mỗi ảnh
    bits_per_message = message_tensor.numel() // batch_size
    pixels_per_image = H * W

    bpp = bits_per_message / pixels_per_image
    return bpp


# ==== Mô hình Encoder ====
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # thêm tầng
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 3, kernel_size=1)  # giữ nguyên đầu ra

    def forward(self, image, message):
        combined = torch.cat([image, message], dim=1)  # (B, 4, H, W)

        x1 = self.relu1(self.conv1(combined))          # (B, 64, H, W)
        x2 = self.relu2(self.conv2(x1))                # (B, 64, H, W)

        x2 = x2 + x1  # simple skip connection

        encoded_image = self.conv3(x2)                 # (B, 3, H, W)
        return encoded_image

# ==== Mô hình Decoder ====
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # đảm bảo đầu ra trong khoảng [0,1]
        )

    def forward(self, encoded_image):
        return self.decoder(encoded_image)
    
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
import reedsolo
#-----------------------------------------
# Utility Class (unchanged)
class SteganographyUtils:
    def __init__(self, max_msg_len=256):
        self.MAX_MSG_LEN = max_msg_len  # Độ dài tối đa của tin nhắn (tính theo ký tự)
        self.rs = reedsolo.RSCodec(nsym=10)
    def text_to_binary(self, text):
        bits = ''.join([format(byte, '08b') for byte in text.encode('utf-8')])
        bits = bits.ljust(self.MAX_MSG_LEN * 8, '0')  # Pad đến độ dài cố định
        return bits

    def binary_to_text(self, bits):
        byte_array = bytearray([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
        return byte_array.decode('utf-8', errors='ignore')

    # Hàm chuyển đổi văn bản thành tensor với mã hóa Reed-Solomon
    def text_to_tensor(self,text, H, W):
        # Bước 1: Chuyển văn bản thành nhị phân
        bits = self.text_to_binary(text)
        
        # Bước 2: Chuyển nhị phân thành mảng byte
        byte_array = np.array([int(b) for b in bits], dtype=np.uint8)
        byte_array = np.packbits(byte_array)  # Chuyển bits thành bytes
        
        # Bước 3: Áp dụng mã hóa Reed-Solomon
        encoded_bytes = self.rs.encode(byte_array.tobytes())  # Mã hóa bằng Reed-Solomon
        
        # Bước 4: Chuyển đổi byte đã mã hóa thành chuỗi nhị phân
        encoded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8)).astype(np.float32)

        # Bước 5: Resize cho phù hợp với kích thước H, W
        total_len = H * W
        if len(encoded_bits) < total_len:
            encoded_bits = np.pad(encoded_bits, (0, total_len - len(encoded_bits)))  # Padding nếu thiếu
        else:
            encoded_bits = encoded_bits[:total_len]  # Cắt nếu dài hơn
        encoded_bits = encoded_bits.reshape((1, H, W))
        return torch.tensor(encoded_bits)  # Trả về tensor với shape (1, H, W)
    def tensor_to_text(self, tensor):
        # Chuyển tensor thành mảng numpy và phẳng ra thành 1 chiều
        bits = tensor.detach().cpu().numpy().flatten()

        # Không cắt bits ở đây, giữ nguyên độ dài
        bits = ['1' if b > 0.5 else '0' for b in bits]
        binary_str = ''.join(bits)

        # Chuyển chuỗi nhị phân thành mảng byte
        byte_array = np.array([int(b) for b in binary_str], dtype=np.uint8)
        byte_array = np.packbits(byte_array)

        try:
            # Giải mã Reed-Solomon
            decoded_bytes = self.rs.decode(byte_array.tobytes())[0]
        except Exception as e:
            print("Reed-Solomon decode failed:", e)
            return "Không thể giải mã thành công hệ thống"

        # Chuyển lại thành chuỗi nhị phân
        decoded_bits = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8)).astype(np.float32)

        # Cắt decoded_bits sau khi giải mã (nếu cần)
        decoded_bits = decoded_bits[:self.MAX_MSG_LEN * 8]

        # Chuyển thành văn bản
        return self.binary_to_text(''.join([str(int(b)) for b in decoded_bits]))

# ---------------------------------------------------------------------
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
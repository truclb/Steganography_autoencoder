#Model Pipeline.
import base64
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from reedsolo import RSCodec
import zlib
from PIL import Image
import torchvision.transforms.functional as TF
# Load the trained SteganoModel

from Model.Model_class_v2 import Encoder, Decoder, SteganographyUtils 
#from Model.Model_class_v3 import Encoder, Decoder, SteganographyUtils 

import torchvision.utils as vutils
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_PATH = ".\Deploy\Model\Save_Model\encoder_v2.pth"
DECODER_PATH = ".\Deploy\Model\Save_Model\dencoder_v2.pth"
STEGO_PATH = ".\Deploy\Storage\Stego_images\stego_image_v2.png"
if not os.path.exists(ENCODER_PATH) or not os.path.exists(DECODER_PATH):
    raise FileNotFoundError("❌ One or both model files not found!")

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))

encoder.to(device).eval()
decoder.to(device).eval()


# Define image transformation (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((1024,1024)),  # Resize giống với dataset đã train
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
utils = SteganographyUtils(max_msg_len=256)
# Embedding function: Encodes secret data into cover image using the loaded model.
def encode(cover_input, secret_data):
    print('------This is ENCODE function------')
    if isinstance(cover_input, str):
        cover_image = Image.open(cover_input).convert("RGB")
    elif isinstance(cover_input, Image.Image):
        cover_image = cover_input.convert("RGB")
    else:
        # Trường hợp còn lại là buffer (ví dụ BytesIO)
        cover_image = Image.open(cover_input).convert("RGB")
    test_img = transform(cover_image).unsqueeze(0).to(device)  # (1, C, H, W)

    # Mã hóa chuỗi thành tensor
    H, W = test_img.shape[-2], test_img.shape[-1]
    test_msg_tensor = utils.text_to_tensor(secret_data, H, W).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Dùng encoder để mã hóa
    with torch.no_grad():
        stego_image = encoder(test_img, test_msg_tensor)
    vutils.save_image(stego_image[0], STEGO_PATH)
    stego_pil = Image.open(STEGO_PATH).convert("RGB")
    print(f"✅ Đã mã hóa chuỗi '{secret_data[:30]}...' vào ảnh")
    return stego_pil #Hàm return trả về hệ thống để hiển thị lên màn hình

# Decoding function: Extracts secret data from the stego image using the loaded model.
def decode(stego_input):
    print("--------This is DECODE function-------")
    
    if isinstance(stego_input, str):
        stego_image = Image.open(stego_input).convert("RGB")
    elif isinstance(stego_input, Image.Image):
        stego_image = stego_input.convert("RGB")
    else:
        # Trường hợp còn lại là buffer (ví dụ BytesIO)
        stego_image = Image.open(stego_input).convert("RGB")
    recovered_text = 'ERROR - cannot recovery'
    recovered_img = stego_image

    stego_tensor = transform(stego_image).unsqueeze(0).to(device)  # Convert to tensor
    # Gọi hàm model để chạy và lấy đặc trưng ra.
    with torch.no_grad():
       recovered_secret = decoder(stego_tensor)

    recovered_text = utils.tensor_to_text(recovered_secret[0])
    print(f"✅ Giải mã thành công chuỗi: ",recovered_text)
    return recovered_text,recovered_img

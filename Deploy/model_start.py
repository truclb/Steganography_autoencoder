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
import base64
from Model.Model_class_v3 import Encoder, Decoder, SteganographyUtils, aes_encrypt,aes_decrypt, calculate_actual_bpp,CoverReconstructor
import torch.nn.functional as F
#from Model.Model_class_v3 import Encoder, Decoder, SteganographyUtils,calculate_actual_bpp 
import torchvision.utils as vutils


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_PATH = ".\Deploy\Model\Save_Model\encoder_v3.pth"
DECODER_PATH = ".\Deploy\Model\Save_Model\decoder_v3.pth"
RECONSTRUCT_PATH = ".\Deploy\Model\Save_Model\c_reconstruct_v3.pth"
STEGO_PATH = ".\Deploy\Storage\Stego_images\stego_image_v2_1.png"
RECOVER_PATH=".\Deploy\Storage\Stego_images\cover_image_v2_1.png"
if not os.path.exists(ENCODER_PATH) or not os.path.exists(DECODER_PATH):
    raise FileNotFoundError("❌ One or both model files not found!")

encoder = Encoder()
decoder = Decoder()
reconstruct = CoverReconstructor()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
reconstruct.load_state_dict(torch.load(RECONSTRUCT_PATH, map_location=device))

encoder.to(device).eval()
decoder.to(device).eval()
reconstruct.to(device).eval()

# Define image transformation (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((1024,1024)),  # Resize giống với dataset đã train
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
utils = SteganographyUtils(max_msg_len=256)
# Embedding function: Encodes secret data into cover image using the loaded model.
def embed_Data(cover_input, secret_data):
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

    secret_data,password = aes_encrypt(secret_data)
    test_msg_tensor = utils.text_to_tensor(secret_data, H, W).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Dùng encoder để mã hóa
    with torch.no_grad():
        stego_image = encoder(test_img, test_msg_tensor)
    vutils.save_image(stego_image[0], STEGO_PATH)
    stego_pil = Image.open(STEGO_PATH).convert("RGB")
    print(f"✅ Đã mã hóa chuỗi '{secret_data[:30]}...' vào ảnh")
    ssim = F.mse_loss(stego_image,test_img)
    bpp = calculate_actual_bpp(test_msg_tensor,test_img)
    print(f"SSIM: {ssim} ")
    print(f"bpp: {bpp}")
    return stego_pil,ssim,bpp,password #Hàm return trả về hệ thống để hiển thị lên màn hình

# Decoding function: Extracts secret data from the stego image using the loaded model.
def extract_Data(stego_input,password):
    print("--------This is DECODE function-------")
    
    if isinstance(stego_input, str):
        stego_image = Image.open(stego_input).convert("RGB")
    elif isinstance(stego_input, Image.Image):
        stego_image = stego_input.convert("RGB")
    else:
        # Trường hợp còn lại là buffer (ví dụ BytesIO)
        stego_image = Image.open(stego_input).convert("RGB")
    recovered_text = 'ERROR - cannot recovery'
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)  # Convert to tensor
    # Gọi hàm model để chạy và lấy đặc trưng ra.
    with torch.no_grad():
       recovered_secret = decoder(stego_tensor)
       recovered_img = reconstruct(stego_tensor,recovered_secret)
    vutils.save_image(recovered_img[0],RECOVER_PATH)
    recovered_img = Image.open(RECOVER_PATH).convert("RGB")
    recovered_text = utils.tensor_to_text(recovered_secret[0])
    recovered_text = aes_decrypt(recovered_text,password)
    print(f"✅ Giải mã thành công chuỗi: ",recovered_text)
    return recovered_text,recovered_img

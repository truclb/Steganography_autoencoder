import base64
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from reedsolo import RSCodec
import zlib

# Load the trained SteganoModel
from autoencoder_class import SteganoModel # Adjust the import path as needed

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model file (assumes the entire model was saved)
MODEL_PATH = "./StegoAE_epoch_40.pth"  # Update to your correct path

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model file '{MODEL_PATH}' not found!")

# Phải import tất cả các class trong Model_class nếu muốn load full.
# model = torch.load(MODEL_PATH, map_location=device)
# model.eval()

#Load State_dict
model = SteganoModel(input_channels=3, secret_size=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# Define image transformation (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Utility functions remain the same
class SteganographyUtils:
    def __init__(self, rs_block_size=250):
        self.rs = RSCodec(rs_block_size)
    
    def text_to_bits(self, text):
        return self.bytearray_to_bits(self.text_to_bytearray(text))
    
    def bits_to_text(self, bits):
        return self.bytearray_to_text(self.bits_to_bytearray(bits))
    
    def bytearray_to_bits(self, x):
        result = []
        for i in x:
            bits = bin(i)[2:]
            bits = "00000000"[len(bits):] + bits
            result.extend([int(b) for b in bits])
        return result
    
    def bits_to_bytearray(self, bits):
        ints = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            ints.append(int("".join([str(bit) for bit in byte]), 2))
        return bytearray(ints)
    
    def text_to_bytearray(self, text):
        x = self.rs.encode(bytearray(text.encode("utf-8")))
        return x
    
    def bytearray_to_text(self, x):
        try:
            text = self.rs.decode(x)
            return text.decode("utf-8")
        except Exception as e:
            print(f"Error during decoding: {e}")
            return False

    # ... (Other utility methods remain unchanged)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def base64_to_image(base64_string, output_path):
    image_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(image_data)

def encode_secret_string(secret_string, secret_size=64):
    """
    Mã hóa chuỗi số như "123456789012" thành tensor nhị phân (float32).
    - Mỗi chữ số 0-9 được mã hóa thành 4-bit nhị phân.
    - Padding 0 nếu chưa đủ secret_size.
    """
    bin_string = ''.join(f"{int(ch):04b}" for ch in secret_string if ch.isdigit())  # '0001' cho '1', v.v.
    bin_array = [int(bit) for bit in bin_string]

    # Padding or truncation
    if len(bin_array) < secret_size:
        bin_array += [0] * (secret_size - len(bin_array))  # pad với 0
    else:
        bin_array = bin_array[:secret_size]  # cắt nếu dài quá

    return torch.tensor(bin_array, dtype=torch.float32)

def decoded_tensor_to_string(tensor):
    """
    Giải mã tensor nhị phân float32 về chuỗi số.
    - tensor có shape [secret_size] hoặc [1, secret_size]
    - mỗi 4 bit thành 1 chữ số thập phân (0–9)
    - dừng nếu gặp padding (tức 4 bit toàn 0)
    """
    if tensor.dim() == 2:
        tensor = tensor.squeeze(0)

    # Làm tròn về 0 hoặc 1 (nếu chưa là int)
    binary_array = (tensor > 0.5).int().tolist()

    digits = []
    for i in range(0, len(binary_array), 4):
        chunk = binary_array[i:i+4]
        if len(chunk) < 4:
            break
        value = int("".join(str(b) for b in chunk), 2)
        if value > 9:  # chỉ cho phép chữ số 0–9 như encode
            continue
        digits.append(str(value))

    return ''.join(digits)


# Embedding function: Encodes secret data into cover image using the loaded model.
def encode(cover_path, stego_path, secret_string:str):
    cover_image = Image.open(cover_path).convert("RGB")
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)  # Convert to tensor
    #Format chắc chắn là text

    # Convert chuỗi thành tensor nhị phân
    #secret_tensor = encode_secret_string(secret_string).unsqueeze(0).to(device)  # [1, secret_size]
    secret_tensor = encode_secret_string(secret_string)  # [1, secret_size]

    print(f"Secret_tensor: {secret_tensor}")
    print(f"decoded_secret: {decoded_tensor_to_string(secret_tensor)}")
    #H, W = cover_tensor.shape[2], cover_tensor.shape[3]
    # Nhúng
    with torch.no_grad():
        stego_image, recovered_secret, _, _ = model(cover_tensor, secret_tensor)
    binary_tensor = (recovered_secret < 0.5).float()
    decoded_digits = decoded_tensor_to_string(binary_tensor)
   
    # Convert tensor to PIL image
    stego_image = stego_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stego_image = (stego_image * 255).clip(0, 255).astype("uint8")
    Image.fromarray(stego_image).save(stego_path)
    
    print(f"Encoded secret from '{secret_string}' into {stego_path}")
    print(f"🔐 Extracted Secret Number: {decoded_digits}")

# Decoding function: Extracts secret data from the stego image using the loaded model.
def decode(stego_path, output_path):
    # Load và chuyển đổi ảnh
    stego_image = Image.open(stego_path).convert("RGB")
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        # Trích xuất secret tensor từ ảnh
        recovered_secret = model.secret_extractor(stego_tensor)  # [1, secret_size]
        decoded_digits = decode_secret_tensor(recovered_secret)  # [['1', '2', ..., '0']]

    # Kiểm tra và ghép thành chuỗi
    if isinstance(decoded_digits, list) and len(decoded_digits) > 0:
        if isinstance(decoded_digits[0], (list, tuple)):
            secret_number = ''.join(decoded_digits[0])
        else:
            secret_number = ''.join(decoded_digits)  # Fallback nếu đầu ra là list flat
    else:
        secret_number = "[Decode Failed]"
    # Ghi ra file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(secret_number)

    print(f"✅ Decoded secret from {stego_path} into {output_path}")
    print(f"🔐 Extracted Secret Number: {secret_number}")

# Using argparse to create options for your program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SteganoModel Data Hiding Tool - Embed and extract secret data in images using a trained SteganoModel."
    )
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Sub-commands: 'encode' to embed secret data into an image, 'decode' to extract secret data from a stego image."
    )
    
    encode_parser = subparsers.add_parser(
        "encode", 
        help="Embed secret data into a cover image to create a stego image."
    )
    encode_parser.add_argument("cover", help="Path to the cover image file (e.g., 'cover.jpg').")
    encode_parser.add_argument("stego", help="Path to save the resulting stego image (e.g., 'stego.jpg').")
    encode_parser.add_argument("secret", help="Path to the secret file (text file or image) to be embedded.")
    encode_parser.add_argument("--format", choices=["text", "image"], default="text",
                               help="Specify the format of the secret data: 'text' for textual secrets, 'image' for image-based secrets. Default is 'text'.")
    
    decode_parser = subparsers.add_parser(
        "decode", 
        help="Extract secret data from a stego image."
    )
    decode_parser.add_argument("stego", help="Path to the stego image file (e.g., 'stego.jpg').")
    decode_parser.add_argument("output", help="Path to save the extracted secret data (text file or image, depending on the format).")
    decode_parser.add_argument("--format", choices=["text", "image"], default="text",
                               help="Specify the expected format of the secret data to be extracted: 'text' or 'image'. Default is 'text'.")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        encode(args.cover, args.stego, args.secret)
    elif args.command == "decode":
        decode(args.stego, args.output)
    else:
        parser.print_help()
#python main.py encode path/to/cover.jpg path/to/stego.jpg <String>
#python main.py decode path/to/stego.jpg path/to/output.txt 

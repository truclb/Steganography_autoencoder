import base64
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from reedsolo import RSCodec
import zlib

# Load the trained SteganoModel
from Deploy.Model.Model_class import SteganoModel  # Adjust the import path as needed

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model file (assumes the entire model was saved)
MODEL_PATH = ".\Deploy\Model\stego_model_weights.pth"  # Update to your correct path

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model file '{MODEL_PATH}' not found!")

# Phải import tất cả các class trong Model_class nếu muốn load full.
# model = torch.load(MODEL_PATH, map_location=device)
# model.eval()

#Load State_dict
model = SteganoModel() 
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

# Embedding function: Encodes secret data into cover image using the loaded model.
def encode(cover_path, stego_path, secret_path, secret_format):
    cover_image = Image.open(cover_path).convert("RGB")
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)  # Convert to tensor

    if secret_format == "text":
        with open(secret_path, "r", encoding="utf-8") as f:
            secret_data = f.read().strip()  # Read and clean the text
        
        # Convert text to binary representation (string of 0s and 1s)
        secret_bin_str = ''.join(format(ord(c), '08b') for c in secret_data)
        secret_bin = [int(bit) for bit in secret_bin_str]
        required_length = 8 * 256 * 256  # 524288 elements
        
        if len(secret_bin) < required_length:
            # Pad with zeros if the secret is too short
            secret_bin = secret_bin + [0] * (required_length - len(secret_bin))
        else:
            # Truncate if too long
            secret_bin = secret_bin[:required_length]
        
        secret_tensor = torch.Tensor(secret_bin).view(1, 8, 256, 256).to(device)
        
    elif secret_format == "image":
        secret_img = Image.open(secret_path).convert("RGB")
        secret_tensor = transform(secret_img).unsqueeze(0).to(device)
    else:
        raise ValueError("Invalid secret format. Choose 'text' or 'image'.")
    
    with torch.no_grad():
        stego_image, _, _, _ = model(cover_tensor, secret_tensor)
    
    # Convert tensor to PIL image
    stego_image = stego_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stego_image = (stego_image * 255).clip(0, 255).astype("uint8")
    Image.fromarray(stego_image).save(stego_path)
    
    print(f"Encoded secret from {secret_path} into {stego_path}")

# Decoding function: Extracts secret data from the stego image using the loaded model.
def decode(stego_path, output_path, secret_format):
    stego_image = Image.open(stego_path).convert("RGB")
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)  # Convert to tensor
    
    with torch.no_grad():
        dummy_secret = torch.zeros((1, 8, 256, 256), device=device)
        _, recovered_secret, _, _ = model(stego_tensor, dummy_secret)
    
    if secret_format == "text":
        recovered_bits = recovered_secret.cpu().numpy().flatten().round().astype(int)
        byte_list = [int("".join(map(str, recovered_bits[i:i+8])), 2) for i in range(0, len(recovered_bits), 8)]
        recovered_text = "".join([chr(b) for b in byte_list if 32 <= b <= 126])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(recovered_text)
    elif secret_format == "image":
        recovered_img = recovered_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
        recovered_img = (recovered_img * 255).clip(0, 255).astype("uint8")
        Image.fromarray(recovered_img).save(output_path)
    else:
        raise ValueError("Invalid secret format. Choose 'text' or 'image'.")
    
    print(f"Decoded secret from {stego_path} into {output_path}")

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
        encode(args.cover, args.stego, args.secret, args.format)
    elif args.command == "decode":
        decode(args.stego, args.output, args.format)
    else:
        parser.print_help()
#python main.py encode path/to/cover.jpg path/to/stego.jpg path/to/secret.txt --format text
#python main.py decode path/to/stego.jpg path/to/output.txt --format text

import base64
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from reedsolo import RSCodec
import zlib
from collections import Counter
import numpy as np
import torch
from torch import nn
# Load the trained SteganoModel
from SteganoGANclass import SteganoModel  # Adjust the import path as needed
from SteganoGANclass import SteganographyUtils
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model file (assumes the entire model was saved)
MODEL_PATH = ".\stego_model_weights.pth"  # Update to your correct path

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
utils = SteganographyUtils()
def string_to_embedding_tensor(text, width, height, depth=8):
    """
    Encode a string into a FloatTensor with shape (1, depth, height, width).
    The message is repeated and padded to fill the whole tensor.
    """
    # Convert text to list of bits
    bits = utils.text_to_bits(text) + [0] * 32  # Add ending marker (32 zeros)
    
    # Ensure the number of bits is a multiple of the required total size
    total_bits = width * height * depth
    if len(bits) > total_bits:
        raise ValueError(f"Message is too large to fit into the specified tensor size {total_bits}.")
    
    # Repeat the message to fill the full payload space, padding if necessary
    repeated_bits = (bits * ((total_bits // len(bits)) + 1))[:total_bits]

    # Convert to tensor and reshape
    embedding_tensor = torch.Tensor(repeated_bits).view(1, depth, height, width)
    
    return embedding_tensor
def _make_payload(width, height, depth, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        """
        message = utils.text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)
def embedding_tensor_to_string(tensor):
    """
    Convert a decoded tensor of shape (1, depth, height, width) into a string.
    The tensor values are thresholded to get bits, then decoded to text.
    """
    # Step 1: Convert tensor to 1D bit vector
    bits = (tensor.detach().cpu().numpy().flatten() > 0).astype(int).tolist()

    # Step 2: Find end of message marker (32 consecutive zeros)
    END_MARKER = [0] * 32
    for i in range(len(bits) - 32):
        if bits[i:i+32] == END_MARKER:
            bits = bits[:i]
            break

    # Step 3: Convert bits to string
    return utils.bits_to_text(bits)

# def _encode(self, cover, output, text):
#         """Encode an image.
#         Args:
#             cover (str): Path to the image to be used as cover.
#             output (str): Path where the generated image will be saved.
#             text (str): Message to hide inside the image.
#         """
#         cover = Image.open(cover, pilmode='RGB')
#         cover_tensor = transform(cover).unsqueeze(0).to(device) #cover này chính là Tensor

#         cover_size = cover_tensor.size()
#         # _, _, height, width = cover.size()
#         payload = string_to_embedding_tensor(cover_size[3], cover_size[2], self.data_depth, text)
#         payload = payload.to(self.device)

#         generated = model.encoder(cover_tensor, payload)[0].clamp(-1.0, 1.0)
#         generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
#         imwrite(output, generated.astype('uint8'))

#         if self.verbose:
#             print('Encoding completed.')

def _decode(self, image): #image_path
        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        # extract a bit vector
        image = Image.OPEN(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in utils.bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = utils.bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')
        candidate, count = candidates.most_common(1)[0]
        return candidate

# Embedding function: Encodes secret data into cover image using the loaded model.
def encode(cover_path, stego_path, secret_text:str):
    cover_image = Image.open(cover_path).convert("RGB")
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)
    B, C, H, W = cover_tensor.shape  # Convert to tensor
    secret_tensor = _make_payload(width=W,height=H,depth=8,text=secret_text)        
    secret_tensor = secret_tensor.to(device)
    with torch.no_grad():
        stego_image, decoded_data, _ = model(cover_tensor, secret_tensor)
        # split and decode messages
    # np.set_printoptions(threshold=np.inf)
    # with open('./tensor.txt', 'w') as f:
    #     f.write(str((decoded_data.view(-1) > 0).int().numpy()))
    #     f.write("\n")
    #     f.write(str((secret_tensor.view(-1) > 0).int().numpy()))
    bceloss = nn.BCELoss()
    bceloss(input=(decoded_data.view(-1) > 0).float(), target=(secret_tensor.view(-1) > 0).float())
    # tensor(7.2500)
    bceloss
    # BCELoss()
    print(bceloss.weight)
    print((decoded_data.view(-1) > 0).int().numpy())
    print((secret_tensor.view(-1) > 0).int().numpy())
    # candidates = Counter()
    # bits = (decoded_data.detach().cpu().numpy().flatten() > 0).astype(int).tolist()
    #     # split and decode messages
    # for candidate in utils.bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
    #         candidate = utils.bytearray_to_text(bytearray(candidate))
    #         if candidate:
    #             candidates[candidate] += 1

    #     # choose most common message
    # if len(candidates) == 0:
    #         raise ValueError('Failed to find message.')

    # candidate, count = candidates.most_common(1)[0]
    # Convert tensor to PIL image
    stego_image = stego_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stego_image = (stego_image * 255).clip(0, 255).astype("uint8")
    Image.fromarray(stego_image).save(stego_path)
    print(f"Encoded secret from '{secret_text}' into {stego_path}")
    #print(f"Decoded Data: ")

# Decoding function: Extracts secret data from the stego image using the loaded model.
def decode(stego_path):
    stego_image = Image.open(stego_path).convert("RGB")
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)  # Convert to tensor
    
    with torch.no_grad():
        dummy_secret = torch.zeros((1, 8, 256, 256), device=device)
        _, recovered_secret, _ = model(stego_tensor, dummy_secret)
    print(recovered_secret)
    recovered_bits = recovered_secret.cpu().numpy().flatten().round().astype(int)
    byte_list = [int("".join(map(str, recovered_bits[i:i+8])), 2) for i in range(0, len(recovered_bits), 8)]
    recovered_text = "".join([chr(b) for b in byte_list if 32 <= b <= 126])
    print(f'recovertext: {recovered_text}')
    #print(f"Decoded secret from {stego_path} into {output_path}")

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
    
    decode_parser = subparsers.add_parser(
        "decode", 
        help="Extract secret data from a stego image."
    )
    decode_parser.add_argument("stego", help="Path to the stego image file (e.g., 'stego.jpg').")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        encode(args.cover, args.stego, args.secret)
    elif args.command == "decode":
        decode(args.stego)
    else:
        parser.print_help()
#python main.py encode path/to/cover.jpg path/to/stego.jpg path/to/secret.txt --format text
#python main.py decode path/to/stego.jpg path/to/output.txt --format text

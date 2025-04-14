import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import random

# Config
IMAGE_SIZE = 256
NUM_DIGITS = 12
BASE_MESSAGE_SIZE = NUM_DIGITS * 4  # 12 số * 4 bits = 48 bits
TERMINATION_BITS = 32
MESSAGE_SIZE = BASE_MESSAGE_SIZE + TERMINATION_BITS  # Tổng 80 bits
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset class của bạn
class CoverImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image

# Hàm tạo message ngẫu nhiên
def generate_random_message():
    return ''.join(str(random.randint(0, 9)) for _ in range(NUM_DIGITS))

# Chuyển đổi message sang nhị phân
def message_to_binary(message_str):
    binary_list = []
    for char in message_str:
        binary = format(int(char), '04b')
        binary_list.extend([int(bit) for bit in binary])
    binary_list += [0] * TERMINATION_BITS
    return torch.tensor(binary_list, dtype=torch.float32)

# Stego Dataset kết hợp
class StegoDataset(Dataset):
    def __init__(self, cover_dataset):
        self.cover_dataset = cover_dataset
        
    def __len__(self):
        return len(self.cover_dataset)
    
    def __getitem__(self, idx):
        cover_image = self.cover_dataset[idx]
        message_str = generate_random_message()
        message = message_to_binary(message_str)
        return cover_image, message

# Autoencoder Architecture
class SteganoNet(nn.Module):
    def __init__(self):
        super(SteganoNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Decoder (đã sửa)
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),  # Thêm lớp này
            nn.AdaptiveAvgPool2d((8, 10)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 10, MESSAGE_SIZE),  # Đã sửa 32*8*10
            nn.Sigmoid()
        )

    def forward(self, cover, message):
        # Xử lý message
        message = message.view(-1, 1, 8, 10)
        message = nn.functional.interpolate(message, size=IMAGE_SIZE)
        
        # Kết hợp với ảnh
        combined = torch.cat([cover, message], dim=1)
        encoded = self.encoder(combined)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Transformations (đã bỏ normalization)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Khởi tạo dataset
cover_dataset = CoverImageDataset(
    image_dir='/home/truclb/AnThongTin/cocostuff-10k-v1.1/images',
    transform=transform
)
stego_dataset = StegoDataset(cover_dataset)

# Hàm huấn luyện
def train():
    dataloader = DataLoader(stego_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SteganoNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        for cover, message in dataloader:
            cover = cover.to(DEVICE)
            message = message.to(DEVICE)
            
            encoded, decoded = model(cover, message)
            loss = criterion(decoded, message)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')
        if ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(),f"stegano_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), 'stegano_model.pth')
    return model

# Hàm giải mã (giữ nguyên)
def decode_message(decoded_tensor):
    binary = (decoded_tensor > 0.5).int().cpu().numpy()
    decoded_strs = []
    
    for batch in binary:
        end_pos = len(batch)
        for i in range(len(batch)-TERMINATION_BITS, len(batch)):
            if batch[i] != 0:
                end_pos = len(batch)
                break
        
        binary_str = ''.join(map(str, batch[:end_pos]))
        message_str = ''
        
        for i in range(0, len(binary_str), 4):
            if i+4 > len(binary_str):
                break
            nibble = binary_str[i:i+4]
            message_str += str(int(nibble, 2))
        
        decoded_strs.append(message_str[:NUM_DIGITS])
    
    return decoded_strs

def encode_message(model, cover_image, message_str, device=DEVICE):
    # Chuyển đổi message sang tensor nhị phân
    binary_message = message_to_binary(message_str).to(device)
    
    # Thêm batch dimension và chuẩn bị ảnh
    cover_tensor = cover_image.unsqueeze(0).to(device)
    binary_message = binary_message.unsqueeze(0)
    
    # Mã hóa
    with torch.no_grad():
        model.eval()
        encoded_image, _ = model(cover_tensor, binary_message)
    
    # Chuyển về PIL Image để lưu/sử dụng
    encoded_image = encoded_image.squeeze(0).cpu()
    return transforms.ToPILImage()(encoded_image)

def decode_message(model, encoded_image, device=DEVICE):
    # Chuẩn bị ảnh đầu vào
    img_tensor = transforms.ToTensor()(encoded_image).unsqueeze(0).to(device)
    
    # Giải mã
    with torch.no_grad():
        model.eval()
        _, decoded_tensor = model(img_tensor, torch.zeros(1, MESSAGE_SIZE).to(device))  # Dummy message
    
    # Chuyển đổi kết quả
    decoded_binary = (decoded_tensor > 0.5).int().cpu().numpy()[0]
    
    # Tìm vị trí kết thúc thực tế
    end_pos = len(decoded_binary)
    for i in range(len(decoded_binary)-TERMINATION_BITS, len(decoded_binary)):
        if decoded_binary[i] != 0:
            end_pos = len(decoded_binary)
            break
    
    # Chuyển đổi binary -> string
    message_str = ''
    binary_str = ''.join(map(str, decoded_binary[:end_pos]))
    
    try:
        for i in range(0, len(binary_str), 4):
            if i+4 > len(binary_str):
                break
            nibble = binary_str[i:i+4]
            message_str += str(int(nibble, 2))
    except:
        pass
    
    return message_str[:NUM_DIGITS]
# Test
if __name__ == '__main__':
    model = train()

    # # Load model đã trained
    # model = SteganoNet().to(DEVICE)
    # model.load_state_dict(torch.load('stegano_model.pth'))
    
    # # Test encode/decode
    # original_message = "680392574123"
    # test_image = cover_dataset[0]  # Lấy ảnh từ dataset
    
    # # Mã hóa
    # encoded_pil = encode_message(model, test_image, original_message)
    # encoded_pil.save('encoded_image.jpg')
    
    # # Giải mã
    # decoded_message = decode_message(model, encoded_pil)
    
    # print("Original Message:", original_message)
    # print("Decoded Message:", decoded_message)
    # print("Match:", original_message == decoded_message)
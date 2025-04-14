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
from tqdm import tqdm
from PIL import Image

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=8; #--> Tăng batch size cho dữ liệu train nhanh.

# Utility class remains unchanged
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
            bits = '00000000'[len(bits):] + bits
            result.extend([int(b) for b in bits])
        return result

    def bits_to_bytearray(self, bits):
        ints = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            ints.append(int(''.join([str(bit) for bit in byte]), 2))
        return bytearray(ints)

    def text_to_bytearray(self, text):
        assert isinstance(text, str), "Expected a string."
        x = zlib.compress(text.encode("utf-8"))
        x = self.rs.encode(bytearray(x))
        return x

    def bytearray_to_text(self, x):
        try:
            text = self.rs.decode(x)
            text = zlib.decompress(text)
            return text.decode("utf-8")
        except Exception as e:
            print(f"Error during decoding: {e}")
            return False

    def gaussian(self, window_size, sigma):
        _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
        gauss = torch.Tensor(_exp)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        padding_size = window_size // 2

        mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
        mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
        sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def first_element(self, storage, loc):
        return storage

# Encoder, Decoder, and Evaluate networks
class DenseEncoder(nn.Module):
    def __init__(self, data_depth=8):
        super(DenseEncoder, self).__init__()
        self.data_depth = data_depth
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + data_depth, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, image, data):
        x1 = self.conv1(image)
        x2_input = torch.cat([x1, data], dim=1)
        x2 = self.conv2(x2_input)
        x3_input = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3_input)
        x4_input = torch.cat([x1, x2, x3], dim=1)
        encoded_image = self.conv4(x4_input)
        return encoded_image

class DenseDecoder(nn.Module):
    def __init__(self, data_depth=8):
        super(DenseDecoder, self).__init__()
        self.data_depth = data_depth
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, self.data_depth, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, encoded_image):
        x1 = self.conv1(encoded_image)
        x2_input = x1
        x2 = self.conv2(x2_input)
        x3_input = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3_input)
        x4_input = torch.cat([x1, x2, x3], dim=1)
        decoded_data = self.conv4(x4_input)
        return decoded_data

class Evaluate(nn.Module):
    def __init__(self):
        super(Evaluate, self).__init__()
        
        # Layers for processing encoded images (3 channels)
        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

        # Layers for processing decoded data (data_depth channels)
        self.data_layers = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, encoded_image, decoded_data):
        # Process the encoded image with image_layers
        encoded_image_score = self.pool(self.image_layers(encoded_image)).view(encoded_image.size(0), -1)
        
        # Process the decoded data with data_layers
        decoded_data_score = self.pool(self.data_layers(decoded_data)).view(decoded_data.size(0), -1)
        
        # Combine the scores from both evaluations
        combined_score = torch.cat([encoded_image_score, decoded_data_score], dim=1)
        final_score = self.fc(combined_score)
        
        return final_score

class SteganoModel(nn.Module):
    def __init__(self, data_depth=8, block_size=64):
        super(SteganoModel, self).__init__()
        self.encoder = DenseEncoder(data_depth=data_depth)
        self.decoder = DenseDecoder(data_depth=data_depth)
        self.critic = Evaluate()
        self.block_size = block_size

    def forward(self, image, data):
        B, C, H, W = image.size()
        encoded_image = torch.zeros_like(image)
        decoded_data = torch.zeros_like(data)
        
        for i in range(0, H, self.block_size):
            for j in range(0, W, self.block_size):
                image_block = image[:, :, i:i+self.block_size, j:j+self.block_size]
                data_block = data[:, :, i:i+self.block_size, j:j+self.block_size]

                encoded_block = self.encoder(image_block, data_block)
                decoded_block = self.decoder(encoded_block)
                
                encoded_image[:, :, i:i+self.block_size, j:j+self.block_size] = encoded_block
                decoded_data[:, :, i:i+self.block_size, j:j+self.block_size] = decoded_block

        evaluation_score = self.critic(encoded_image, decoded_data)
        return encoded_image, decoded_data, evaluation_score

# Gradient Penalty Function
def gradient_penalty(critic, real_images, fake_images):
    batch_size, c, h, w = real_images.size()
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    alpha = alpha.expand_as(real_images)

    # Interpolate between real and fake images
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    interpolated_images.requires_grad_(True)

    # Forward pass through the image layers of the critic (the input has 3 channels)
    interpolated_scores = critic.image_layers(interpolated_images)

    # Compute gradients with respect to the interpolated images
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

class CoverImageDataset(Dataset):
    def __init__(self, image_dir, block_size=64, transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.block_size = block_size
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image

# Step 2: Define the Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Create an Instance of the Dataset
cover_image_dataset = CoverImageDataset(image_dir='/home/truclb/AnThongTin/cocostuff-10k-v1.1/images', transform=transform)

# Main execution block
if __name__ == '__main__':
    dataloader = DataLoader(cover_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SteganoModel(data_depth=8, block_size=64)
    model = model.to(device)
    optimizer_enc_dec = optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.0001)
    optimizer_critic = optim.AdamW(model.critic.parameters(), lr=0.0001)

    criterion_d = nn.CrossEntropyLoss()
    criterion_s = nn.MSELoss()
    criterion_r = lambda x: -torch.mean(x)

    lambda_r = 1.0
    lambda_gp = 10.0
    num_epochs = 20

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        total_critic_loss = 0.0
        total_enc_dec_loss = 0.0

        for images in progress_bar:
            images = images.to(device)
            B, C, H, W = images.size()
            embedding_data = torch.bernoulli(torch.full((B, 8, H, W), 0.5)).to(device)

            optimizer_critic.zero_grad()
            real_image_score = model.critic(images, embedding_data)
            encoded_image, _, _ = model(images, embedding_data)
            encoded_image_score = model.critic(encoded_image, embedding_data)

            loss_c = -(torch.mean(real_image_score) - torch.mean(encoded_image_score))
            gp = gradient_penalty(model.critic, images, encoded_image)
            loss_c += lambda_gp * gp
            loss_c.backward()
            optimizer_critic.step()
            total_critic_loss += loss_c.item()

            optimizer_enc_dec.zero_grad()
            encoded_image, decoded_data, critic_score = model(images, embedding_data)
            loss_d = criterion_d(decoded_data, embedding_data)
            loss_s = criterion_s(encoded_image, images)
            loss_r = criterion_r(critic_score)
            total_loss = loss_d + loss_s + lambda_r * loss_r
            total_loss.backward()
            optimizer_enc_dec.step()
            total_enc_dec_loss += total_loss.item()
            progress_bar.set_postfix(critic_loss=loss_c.item(), enc_dec_loss=total_loss.item())

        avg_critic_loss = total_critic_loss / len(dataloader)
        avg_enc_dec_loss = total_enc_dec_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Critic Loss: {avg_critic_loss:.4f}, Avg Enc-Dec Loss: {avg_enc_dec_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),f"./src/model/stegoGAN_epoch_{epoch+1}.pth")
    print("Training complete!")
    torch.save(model.state_dict(), './src/model/stegoGAN_weights.pth')
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

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
# ---------------------------------------------------------------------
# Utility Class (unchanged)
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

# ---------------------------------------------------------------------
# New Modules Based on the Revised Strategy

# 1. Feature Extractor: Extracts CNN features from the cover image.
class FeatureExtractor(nn.Module):
    def __init__(self, num_channels=64):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [B, 3, H, W]
        return self.conv(x)

# 2. Secret Embedder: Fuses cover image features and secret data (extra channels).
class SecretEmbedder(nn.Module):
    def __init__(self, num_image_channels=64, num_secret_channels=8):
        super(SecretEmbedder, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(num_image_channels + num_secret_channels, num_image_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_image_channels, num_image_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, image_features, secret_data):
        # secret_data: [B, num_secret_channels, H, W]
        combined = torch.cat([image_features, secret_data], dim=1)
        return self.fuse(combined)

# 3. Stego Reconstructor: Reconstructs the stego image from the fused features.
class StegoReconstructor(nn.Module):
    def __init__(self, num_channels=64):
        super(StegoReconstructor, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # Adding two more convolutional layers in between
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)  # Output 3-channel image
    
    def forward(self, embedded_features):
        # First conv layer
        x1 = self.conv1(embedded_features)
        x1 = self.relu1(x1)

        # Second conv layer
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)

        # Third conv layer
        x3 = self.conv3(x2)
        x3 = self.relu3(x3)

        # Skip connection: Add the output from conv3 and the original input (embedded_features)
        x = x3 + embedded_features  # Skip connection
        
        # Final conv layer
        output = self.conv4(x)
        return output

# 4. Secret Extractor: Recovers the hidden secret data from the stego image.
class SecretExtractor(nn.Module):
    def __init__(self, num_secret_channels=8):
        super(SecretExtractor, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_secret_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normalizes output to [0,1]
        )
    
    def forward(self, stego_image):
        return self.extract(stego_image)

# 5. Cover Reconstructor: Recovers the original cover image from extracted features.
class CoverReconstructor(nn.Module):
    def __init__(self, num_channels=64):
        super(CoverReconstructor, self).__init__()
        self.reconstruct = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  # Output 3-channel cover image
        )
    
    def forward(self, cover_features):
        return self.reconstruct(cover_features)

# 6. Evaluate Module: Evaluates the quality of the stego image and recovered secret.
class Evaluate(nn.Module):
    def __init__(self):
        super(Evaluate, self).__init__()
        # Process stego image (3 channels)
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
        # Process recovered secret (num_secret_channels)
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
    
    def forward(self, stego_image, recovered_secret):
        image_score = self.pool(self.image_layers(stego_image)).view(stego_image.size(0), -1)
        secret_score = self.pool(self.data_layers(recovered_secret)).view(recovered_secret.size(0), -1)
        combined = torch.cat([image_score, secret_score], dim=1)
        return self.fc(combined)

# 7. Updated SteganoModel: Integrates all components following our strategy.
class SteganoModel(nn.Module):
    def __init__(self, num_channels=64, num_secret_channels=8):
        super(SteganoModel, self).__init__()
        self.feature_extractor = FeatureExtractor(num_channels=num_channels)
        self.secret_embedder = SecretEmbedder(num_image_channels=num_channels, num_secret_channels=num_secret_channels)
        self.stego_reconstructor = StegoReconstructor(num_channels=num_channels)
        self.secret_extractor = SecretExtractor(num_secret_channels=num_secret_channels)
        self.cover_reconstructor = CoverReconstructor(num_channels=num_channels)
        self.evaluate = Evaluate()  # Evaluate module to compute auxiliary quality score
    
    def forward(self, cover_image, secret_data):
        """
        cover_image: Tensor [B, 3, H, W]
        secret_data: Tensor [B, num_secret_channels, H, W]
        """
        #--> Step 1,2,3 --> Encode - embedding data into Imgate
        # Step 1: Extract cover image features.
        cover_features = self.feature_extractor(cover_image)
        # Step 2: Fuse secret data (as extra channels) into the cover features.
        fused_features = self.secret_embedder(cover_features, secret_data)
        # Step 3: Reconstruct the stego image from the fused features.
        stego_image = self.stego_reconstructor(fused_features)
        
        #--> Step 4,5,6 --> Decode - Extracted_data.
        # Step 4: Extract features from the stego image.
        extracted_features = self.feature_extractor(stego_image)
        # Step 5: Recover secret data from the stego image.
        recovered_secret = self.secret_extractor(stego_image)
        # Step 6: Recover the original cover image from the extracted cover features.
        reconstructed_cover = self.cover_reconstructor(extracted_features)
        
        # Auxiliary evaluation score from Evaluate module.
        eval_score = self.evaluate(stego_image, recovered_secret)
        
        return stego_image, recovered_secret, reconstructed_cover, eval_score

# ---------------------------------------------------------------------
# Gradient Penalty Function (unchanged)
def gradient_penalty(critic, real_images, fake_images):
    batch_size, c, h, w = real_images.size()
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    alpha = alpha.expand_as(real_images)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    interpolated_images.requires_grad_(True)
    interpolated_scores = critic.image_layers(interpolated_images)
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

# ---------------------------------------------------------------------
# Dataset Class remains unchanged.
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

# ---------------------------------------------------------------------
# Transformation and Dataset Instance
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cover_image_dataset = CoverImageDataset(image_dir='/kaggle/input/cocostuff-10k-v1-1/images', transform=transform)

# ---------------------------------------------------------------------
# Loss Function for End-to-End Training
def hybrid_loss(stego_image, cover_image, recovered_secret, secret_data, reconstructed_cover, alpha=0.5):
    loss_visual = nn.functional.mse_loss(stego_image, cover_image)
    loss_secret = nn.functional.mse_loss(recovered_secret, secret_data)
    loss_cover = nn.functional.mse_loss(reconstructed_cover, cover_image)
    return alpha * (loss_visual + loss_cover) + (1 - alpha) * loss_secret


if __name__ == '__main__':
    dataloader = DataLoader(cover_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = SteganoModel(num_channels=64, num_secret_channels=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    num_epochs = 50

    for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            total_loss_epoch = 0.0

            for images in progress_bar:
                images = images.to(device)
                B, C, H, W = images.size()
                secret_data = torch.bernoulli(torch.full((B, 8, H, W), 0.5)).to(device)

                optimizer.zero_grad()
                stego_image, recovered_secret, reconstructed_cover, eval_score = model(images, secret_data)
                loss = hybrid_loss(stego_image, images, recovered_secret, secret_data, reconstructed_cover, alpha=0.5)
                loss.backward()
                optimizer.step()
                total_loss_epoch += loss.item()
                progress_bar.set_postfix(loss=loss.item(), eval_score=eval_score.mean().item())

            avg_loss = total_loss_epoch / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")
            if ((epoch + 1) % 5) == 0:
                torch.save(model.state_dict(),f"./src/model/stegoAE_epoch_{epoch+1}.pth")
    print("Training complete!")
    torch.save(model.state_dict(), './src/model/stegoAE_weights.pth')

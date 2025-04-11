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
# 1. Encoder components
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # B/2, 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # B/4, 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # B/8, 128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),            # B/16, 256
            nn.ReLU()
        )

    def forward(self, cover_image):
        return self.conv(cover_image) #tạo ra img_feat
    
# 1.2 Embedding Network
class SecretEmbedder(nn.Module):
    def __init__(self, secret_size=64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.ConvTranspose2d(256 + secret_size, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, image_features, secret_data):
        # Expand secret vector to feature map
        B, _, H, W = image_features.shape
        secret_map = secret_data.view(B, self.secret_size, 1, 1).expand(-1, -1, H, W)
        combined = torch.cat([image_features, secret_map], dim=1)
        return self.fuse(combined)

# 2. Decoder components
class StegoReconstructor(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.refine = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1)

    def forward(self, embedded_features, cover_image):
        decoded = self.decoder(embedded_features)
        concat = torch.cat([decoded, cover_image], dim=1)
        return self.refine(concat)


# 4. Secret Extractor: Recovers the hidden secret data from the stego image.
class SecretExtractor(nn.Module):
    def __init__(self, secret_size=64, input_channels=3):
        super(SecretExtractor, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, secret_size),
            nn.Sigmoid()
        )

    def forward(self, stego_image):
        features = self.extract(stego_image)
        flattened = self.flatten(features)
        extracted_secret = self.fc(flattened)
        return extracted_secret

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
    def __init__(self,secret_size=100):
        super(Evaluate, self).__init__()
        # Process stego image (3 channels)
        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),    # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # output [B, 256, 1, 1]
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 + secret_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: score
        )
    
    def forward(self, stego_image, recovered_secret):
        B = stego_image.size(0)
        # Extract image feature
        img_feat = self.image_layers(stego_image).view(B, -1)  # [B, 256]
        # Combine with recovered secret [B, secret_size]
        combined = torch.cat([img_feat, recovered_secret], dim=1)
        # Final score
        return self.fc(combined)  # [B, 1]

# 7. Updated SteganoModel: Integrates all components following our strategy.
class SteganoModel(nn.Module):
    def __init__(self, input_channels=3, secret_size=100):
        super(SteganoModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels=input_channels) #Encoder
        self.secret_embedder = SecretEmbedder(secret_size=secret_size) #embedding network
        self.stego_reconstructor = StegoReconstructor(input_channels=input_channels) #Decoder

        self.secret_extractor = SecretExtractor(secret_size=secret_size,input_channels=input_channels) #extractor M'
        
        #self.cover_reconstructor = CoverReconstructor(num_channels=num_channels) 
        self.evaluate = Evaluate()  # Evaluate module to compute auxiliary quality score
    
    def forward(self, cover_image, secret_data):
        """
        cover_image: Tensor [B, 3, H, W]
        secret_data: Tensor [B, num_secret_channels, H, W]
        """
        #--> Step 1,2,3 --> Encode - embedding data into Image
        # Step 1: Extract cover image features.
        cover_features = self.feature_extractor(cover_image)
        # Step 2: Fuse secret data (as extra channels) into the cover features.
        fused_features = self.secret_embedder(cover_features, secret_data)
        # Step 3: Reconstruct the stego image from the fused features.
        stego_image = self.stego_reconstructor(fused_features,cover_image)
        
        #--> Step 4,5,6 --> Extracted_data.
        # Step 4: Extract features from the stego image.
        #extracted_features = self.feature_extractor(stego_image)
        # Step 5: Recover secret data from the stego image.
        recovered_secret = self.secret_extractor(stego_image)

        # Step 6: Recover the original cover image from the extracted cover features.
        #reconstructed_cover = self.cover_reconstructor(extracted_features)
        reconstructed_cover = 'reconstructed_cover'
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
#cover_image_dataset = CoverImageDataset(image_dir='/kaggle/input/cocostuff-10k-v1-1/images', transform=transform)

# ---------------------------------------------------------------------
# Loss Function for End-to-End Training
def hybrid_loss(stego_image, cover_image, recovered_secret, secret_data, reconstructed_cover=0, alpha=0.5):
    loss_visual = nn.functional.mse_loss(stego_image, cover_image)
    loss_secret = nn.functional.mse_loss(recovered_secret, secret_data)
    #loss_cover = nn.functional.mse_loss(reconstructed_cover, cover_image)
    loss_cover = 0
    return alpha * (loss_visual + loss_cover) + (1 - alpha) * loss_secret
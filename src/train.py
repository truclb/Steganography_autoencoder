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

# For save and trace results
from autoencoder_class import CoverImageDataset, SteganoModel,hybrid_loss

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
INPUT_CHANNELS = 3
SECRET_SIZE = 64
# Transformation and Dataset Instance
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cover_image_dataset = CoverImageDataset(image_dir='/home/truclb/AnThongTin/dataset/', transform=transform)

# ---------------------------------------------------------------------
# Set the experiment name (this can be customized)
#mlflow.set_experiment("SteganoModel_Experiment")

# Main Execution Block for Training with MLflow logging
if __name__ == '__main__':
    dataloader = DataLoader(cover_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    model = SteganoModel(input_channels=INPUT_CHANNELS, secret_size=SECRET_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("---------Training START--------")
    for epoch in range(NUM_EPOCHS):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
            total_loss_epoch = 0.0

            for images in progress_bar:
                images = images.to(device)
                B, C, H, W = images.size()
                secret_data = torch.bernoulli(torch.full((B, SECRET_SIZE), 0.5)).to(device)

                optimizer.zero_grad()
                stego_image, recovered_secret, reconstructed_cover, eval_score = model(images, secret_data)
                loss = hybrid_loss(stego_image, images, recovered_secret, secret_data, reconstructed_cover=0, alpha=0.5) #Không đánh giá Reconstructed_cover.
                loss.backward()
                optimizer.step()
                total_loss_epoch += loss.item()
                progress_bar.set_postfix(loss=loss.item(), eval_score=eval_score.mean().item())

            avg_loss = total_loss_epoch / len(dataloader)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Avg Loss: {avg_loss:.4f}")
            if (epoch + 1) % 20 == 0:
                torch.save(model.state_dict(),f"./model/StegoAE_epoch_{epoch+1}.pth")
        # Log the final model
        # mlflow.pytorch.log_model(model, "SteganoModel")
    print("Training complete!")
    torch.save(model.state_dict(), './model/stego_AEmodel_weights.pth')
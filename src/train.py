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
import mlflow
import mlflow.pytorch
from src import autoencoder_class
from .autoencoder_class import CoverImageDataset, SteganoModel,hybrid_loss

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
num_epochs = 10
# Transformation and Dataset Instance
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cover_image_dataset = CoverImageDataset(image_dir='/kaggle/input/cocostuff-10k-v1-1/images', transform=transform)

# ---------------------------------------------------------------------
# Set the experiment name (this can be customized)
mlflow.set_experiment("SteganoModel_Experiment")

# Main Execution Block for Training with MLflow logging
if __name__ == '__main__':
    dataloader = DataLoader(cover_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = SteganoModel(num_channels=64, num_secret_channels=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    with mlflow.start_run():
        # Log initial parameters
        mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("num_channels", 64)
        mlflow.log_param("num_secret_channels", 8)
        
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
            # Log average loss and evaluation score for the epoch
            mlflow.log_metric("avg_loss", avg_loss, step=epoch+1)
            mlflow.log_metric("eval_score", eval_score.mean().item(), step=epoch+1)
            
            # Save the model after each epoch
            mlflow.pytorch.log_model(model, f"SteganoModel_epoch_{epoch+1}")
        # Log the final model
        # mlflow.pytorch.log_model(model, "SteganoModel")
        print("Training complete!")
    torch.save(model.state_dict(), '/kaggle/working/stego_model_weights.pth')
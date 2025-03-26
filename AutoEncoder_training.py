import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import AutoEncoder_class
from AutoEncoder_class import ECGAutoencoder,ECGDataset

# 1. Định nghĩa Dataset tùy chỉnh
# 2. Định nghĩa Autoencoder

# 3. Các tham số huấn luyện
root_dir = './data'
batch_size = 32
learning_rate = 0.001
epochs = 50
# 4. Tiền xử lý hình ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 5. Tạo Dataset và DataLoader
train_dataset = ECGDataset(root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 6. Khởi tạo mô hình, hàm mất mát và optimizer
model = ECGAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. Vòng lặp huấn luyện
for epoch in range(epochs):
    for images in train_loader:
        # Forward pass
        output = model(images)
        loss = criterion(output, images)

        # Backward pass và tối ưu hóa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print('Huấn luyện hoàn tất!')

# 8. Lưu mô hình đã huấn luyện (tùy chọn)
torch.save(model.state_dict(), 'ecg_autoencoder.pth')

# 9. Hiển thị kết quả (tùy chọn)
# (Bạn có thể thêm mã để hiển thị hình ảnh gốc và hình ảnh tái tạo)
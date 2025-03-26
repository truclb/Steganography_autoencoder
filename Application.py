import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from AutoEncoder_class import ECGAutoencoder,ECGDataset
from AES import giai_ma,ma_hoa

# 1. Tải mô hình đã huấn luyện
model = ECGAutoencoder()  # Tạo một thể hiện của mô hình
model.load_state_dict(torch.load('ecg_autoencoder.pth'))  # Tải trọng số đã huấn luyện
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# 2. Chuẩn bị dữ liệu để nhúng (ví dụ: chuỗi văn bản)
data_to_embed = "086099002114"
data_bytes = data_to_embed.encode('utf-8')
data_tensor = torch.tensor(list(data_bytes), dtype=torch.float32)

# 3. Tải hình ảnh ECG
image_path = './Testdata/Cover_image/HB(1).jpg'  # Thay thế bằng đường dẫn hình ảnh của bạn
image = Image.open(image_path).convert('RGB')

# 4. Tiền xử lý hình ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # Thêm chiều batch

# 5. Nhúng dữ liệu vào hình ảnh
with torch.no_grad():
    encoded_image = model.encoder(image_tensor)
    # Chèn dữ liệu vào encoded_image (cần điều chỉnh tùy thuộc vào cách bạn muốn nhúng dữ liệu)
    # Ví dụ đơn giản: Thay thế một phần của encoded_image bằng data_tensor
    encoded_image[0, :data_tensor.size(0)] = data_tensor

    embedded_image = model.decoder(encoded_image)

# 6. Lưu hình ảnh đã nhúng
embedded_image_pil = transforms.ToPILImage()(embedded_image.squeeze(0))
embedded_image_pil.save('./Testdata/Stego_image/ecg_embedded.jpg')

# 7. Trích xuất dữ liệu từ hình ảnh đã nhúng
embedded_image = Image.open('./Testdata/Stego_image/ecg_embedded.jpg').convert('RGB')
embedded_image_tensor = transform(embedded_image).unsqueeze(0)

with torch.no_grad():
    extracted_encoded_image = model.encoder(embedded_image_tensor)
    # Trích xuất dữ liệu từ extracted_encoded_image (cần điều chỉnh tùy thuộc vào cách bạn đã nhúng dữ liệu)
    extracted_data_tensor = extracted_encoded_image[0, :data_tensor.size(0)]

    extracted_data_bytes = bytes(extracted_data_tensor.tolist())
    extracted_data = extracted_data_bytes.decode('utf-8')

print("Dữ liệu đã trích xuất:", extracted_data)
# print("Giải mã dữ liệu: ")
# khoa = 'ORhc6w43H9SmNZtpqDSh17A20b+4lPP/yvhAb7/TFs4='
# thong_tin_goc_giai_ma = giai_ma(ciphertext_base64, khoa)
# ho_ten_giai_ma, ngay_sinh_giai_ma, cccd_giai_ma = thong_tin_goc_giai_ma.split("-")
# print("\nThông tin gốc sau khi giải mã:")
# print(f"Họ và tên bệnh nhân: {ho_ten_giai_ma} - Ngày sinh: {ngay_sinh_giai_ma} - CCCD: {cccd_giai_ma}")
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
from AES import giai_ma,ma_hoa

# Nhập thông tin từ người dùng
ho_ten = input("Nhập Họ và tên: ")
ngay_sinh = input("Nhập Ngày sinh (dd/mm/yyyy): ")
cccd = input("Nhập CCCD: ")

# Kết hợp thông tin thành một chuỗi
thong_tin_goc = f"{ho_ten}-{ngay_sinh}-{cccd}"

# Tạo khóa mã hóa ngẫu nhiên (phải giữ bí mật!)
khoa = get_random_bytes(32)  # Khóa AES-256 (32 byte)

# Mã hóa thông tin
ciphertext_base64 = ma_hoa(thong_tin_goc, khoa)
print("\nThông tin đã mã hóa:", ciphertext_base64)

# Hiển thị thông tin gốc sau khi giải mã
thong_tin_goc_giai_ma = giai_ma(ciphertext_base64, khoa)
ho_ten_giai_ma, ngay_sinh_giai_ma, cccd_giai_ma = thong_tin_goc_giai_ma.split("-")
print("\nThông tin gốc sau khi giải mã:")
print(f"Họ và tên bệnh nhân: {ho_ten_giai_ma} - Ngày sinh: {ngay_sinh_giai_ma} - CCCD: {cccd_giai_ma}")

# In ra khóa mã hóa(chỉ in ra để kiểm tra, trong thực tế phải giữ bí mật tuyệt đối)
print("Khóa mã hóa (base64):", base64.b64encode(khoa).decode('utf-8'))
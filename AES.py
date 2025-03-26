from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64

def ma_hoa(thong_tin, khoa):
    """Mã hóa thông tin sử dụng AES."""
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(khoa, AES.MODE_CBC, iv)
    padded_thong_tin = pad(thong_tin.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_thong_tin)
    return base64.b64encode(iv + ciphertext).decode('utf-8')

def giai_ma(ciphertext_base64, khoa):
    """Giải mã thông tin đã mã hóa bằng AES."""
    ciphertext = base64.b64decode(ciphertext_base64)
    iv = ciphertext[:AES.block_size]
    ciphertext = ciphertext[AES.block_size:]
    cipher = AES.new(khoa, AES.MODE_CBC, iv)
    plaintext_padded = cipher.decrypt(ciphertext)
    plaintext = unpad(plaintext_padded, AES.block_size)
    return plaintext.decode('utf-8')
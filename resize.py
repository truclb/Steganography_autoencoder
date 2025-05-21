from PIL import Image

# Mở ảnh PNG
img = Image.open("Deploy\Storage\Stego_images\stego_image_v2.png")

# In kích thước ban đầu
original_width, original_height = img.size
print(f"Kích thước ban đầu: {original_width}x{original_height}")

# --- Resize theo tỉ lệ ---
scales = [0.95, 0.85, 0.75, 0.65, 0.50]

for scale in scales:
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    output_filename = f"resized_{int(scale*100)}percent.png"
    resized_img.save(output_filename, format="PNG")
    print(f"Đã lưu ảnh {output_filename} với kích thước: {new_width}x{new_height}")

# --- Xoay ảnh ---
angles = [90, 180, 270]

for angle in angles:
    rotated_img = img.rotate(angle, expand=True)
    output_filename = f"rotated_{angle}.png"
    rotated_img.save(output_filename, format="PNG")
    print(f"Đã lưu ảnh xoay {angle} độ: {output_filename} với kích thước: {rotated_img.size[0]}x{rotated_img.size[1]}")

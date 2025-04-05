from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()
templates = Jinja2Templates(directory="./Deploy/templates")

#Defautl Web
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    try:
        # Trả về template chính với request
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Error in GET: {e}")  # In lỗi ra console để dễ dàng debug
        return HTMLResponse(f"Internal Server Error: {e}", status_code=500)

# Endpoint trích xuất dữ liệu từ ảnh
@app.post("/extract-data/", response_class=HTMLResponse)
async def extract_data(request: Request, imageUpload: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file upload
        img_bytes = await imageUpload.read()
        img = Image.open(BytesIO(img_bytes))

        # Trích xuất dữ liệu từ ảnh (Ví dụ: trả về văn bản)
        extracted_data = "Dữ liệu trích xuất từ ảnh (Ví dụ)"
        
        # Phục hồi ảnh (Giả sử ảnh không thay đổi)
        restored_image = img

        # Lưu ảnh phục hồi vào bộ đệm
        buffered = BytesIO()
        restored_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return templates.TemplateResponse("extract-data.html", {
            "request": request,
            "extractedData": extracted_data,
            "restoredImageURL": f"data:image/png;base64,{img_str}",
            "imageWithDataURL": None  # Không có ảnh với dữ liệu nhúng
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})

# Endpoint nhúng dữ liệu vào ảnh
@app.post("/embed-data/", response_class=HTMLResponse)
async def embed_data(request: Request, imageUploadEmbed: UploadFile = File(...), dataInput: str = Form(...)):
    try:
        # Đọc ảnh từ file upload
        img_bytes = await imageUploadEmbed.read()
        img = Image.open(BytesIO(img_bytes))

        # Nhúng dữ liệu vào ảnh (Ví dụ: thêm văn bản vào ảnh)
        img_with_data = img.copy()  # Đây chỉ là ví dụ, bạn có thể thực hiện nhúng dữ liệu thật sự vào ảnh

        # Lưu ảnh với dữ liệu đã nhúng
        img_with_data.save("image_with_data.png")

        # Chuyển ảnh với dữ liệu đã nhúng thành base64
        buffered_with_data = BytesIO()
        img_with_data.save(buffered_with_data, format="PNG")
        img_with_data_str = base64.b64encode(buffered_with_data.getvalue()).decode("utf-8")

        return templates.TemplateResponse("embed-data.html", {
            "request": request,
            "extractedData": None,  # Không có dữ liệu trích xuất
            "restoredImageURL": None,  # Không có ảnh phục hồi
            "imageWithDataURL": f"data:image/png;base64,{img_with_data_str}"
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})
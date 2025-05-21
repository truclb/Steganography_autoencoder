from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
from PIL import Image
import base64
import model_start

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
async def extract_data(request: Request, imageUpload: UploadFile = File(...), password: str=Form(...)):
    try:
        # Đọc ảnh từ file upload
        img_bytes = await imageUpload.read()
        img = Image.open(BytesIO(img_bytes))

        # Gọi decode không cần output_path nữa
        extracted_data, restored_image = model_start.extract_Data(img,password)

         # Chuyển ảnh với dữ liệu đã nhúng thành base64 --> để hiển thị trên web
        buffer_display = BytesIO()
        restored_image.save(buffer_display, format="PNG")
        img_str = base64.b64encode(buffer_display.getvalue()).decode("utf-8")

        return templates.TemplateResponse("extract-data.html", {
            "request": request,
            "extractedData": extracted_data,
            "restoredImageURL": f"data:image/png;base64,{img_str}",
        })
    except Exception as e:
        print(e)
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})

# Endpoint nhúng dữ liệu vào ảnh
@app.post("/embed-data/", response_class=HTMLResponse)
async def embed_data(request: Request, imageUploadEmbed: UploadFile = File(...), dataInput: str = Form(...)):
    try:
        # Đọc ảnh từ file upload
        img_bytes = await imageUploadEmbed.read()
        img = Image.open(BytesIO(img_bytes))
        secret_data = dataInput
        print("Secret data la: ",secret_data)
        # Nhúng dữ liệu vào ảnh (Ví dụ: thêm văn bản vào ảnh)
        img_with_data,ssim_value,bpp_value,password = model_start.embed_Data(img,secret_data)

        # Chuyển ảnh với dữ liệu đã nhúng thành base64 --> để hiển thị trên web
        buffered_with_data = BytesIO()
        img_with_data.save(buffered_with_data, format="PNG")
        img_with_data_str = base64.b64encode(buffered_with_data.getvalue()).decode("utf-8")

        return templates.TemplateResponse("embed-data.html", {
            "request": request,
            "imageWithDataURL": f"data:image/png;base64,{img_with_data_str}",
            "ssim_value": f"{ssim_value}",
            "bpp_value": f"{bpp_value}",
            "hidden_string":f"{password}"
        })
    except Exception as e:
        print(e)
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})
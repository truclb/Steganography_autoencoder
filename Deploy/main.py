# hàm FastAPI sẽ được tạo tại đây
from fastapi import FastAPI #import class FastAPI() từ thư viện fastapi
from PIL import Image
app = FastAPI() # gọi constructor và gán vào biến app


@app.get("/") #khai bao phuong thuc get
def root():
    return {"message": "Hello World"}

@app.post("/embed_data")
def embeding_data(text: str, image: Image):
    return {"answer": "File ảnh stego_image"}

@app.post("/extract_data")
def extract_data(stego_img: Image):
    return{"Cover_image":"File ảnh sau khi recover"}
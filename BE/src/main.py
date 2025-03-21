import article_crawl as ac
import url_extract as ue
from piplines import OcrPipeLine, ImageEncoderPipeLine
from fastapi import FastAPI, File, UploadFile, Path, HTTPException, Form, Request
import uvicorn
from contextlib import asynccontextmanager
from models.vgg16 import VGG16
from models.vintern_1b_v2 import vintern_1b_v2, load_image
from models.gemini_OCR import gemini_ocr
from PIL import Image
import torch
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import  preprocess_input
from fastapi.middleware.cors import CORSMiddleware
from db import initialize_collection
import base64
import numpy as np
import io
import requests
from pydantic import BaseModel
import re
import gc
import tensorflow.keras.backend as K
from dotenv import load_dotenv
import os


imgEncoder, ocrScanner ,vector_db= None, None, None
@asynccontextmanager
async def lifespan(app:FastAPI):
    load_dotenv()
    global imgEncoder, ocrScanner, vector_db
    model = VGG16(include_top=True, weights='imagenet')
    layer_name = 'fc2'
    imgEncoder = ImageEncoderPipeLine(Model(inputs=model.input, outputs=model.get_layer(layer_name).output), preprocess_input)
    # ocrScanner = OcrPipeLine(*vintern_1b_v2())
    ocrScanner = OcrPipeLine(model=gemini_ocr(os.getenv("GEMINI_API_KEY")), model_type="gemini")
    vector_db = initialize_collection()
    print("Model and db loaded successfully.")
    yield
    del imgEncoder, ocrScanner ,vector_db
    torch.cuda.empty_cache()  
    imgEncoder, ocrScanner ,vector_db= None, None, None
    print("Model and db unloaded successfully.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def insertVector(data):
    vector_db.load()
    try:
        vector_db.insert(data)
        return True
    except Exception as e:
        print(f"Error inserting data: {e}")
        return False
    
def contains_keywords(text, keywords):
    pattern = r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    return bool(re.search(pattern, text, re.IGNORECASE))
def parse_keywords(user_input):
    return [kw.strip() for kw in user_input.split(",") if kw.strip()]
def decode_base64_image(file: str):
    try:
        if file.startswith("data:image/"):
            header, file_content = file.split(",", 1) 
            padding = len(file_content) % 4
            if padding != 0:
                file_content += '=' * (4 - padding)
            return base64.b64decode(file_content)
        else:
            raise HTTPException(status_code=400, detail="Invalid base64 string.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding base64: {str(e)}") 
def searchVector(embedding, company_name):
    vector_db.load()
    res = vector_db.search(
        [embedding],
        "embedding",
        filter=f"company_name == '{company_name}'",
        limit=1,
        output_fields=["text"],
        param={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )
    return res



# @app.middleware("http")
# async def cleanup_memory(request: Request, call_next):
#     response = await call_next(request)

#     # Giải phóng bộ nhớ GPU và dọn dẹp TensorFlow sau mỗi request
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     K.clear_session()
#     gc.collect()

#     return response
def caculateMem(tensor):
    size_in_bytes = tensor.numel() * tensor.element_size()
    size_in_MB = size_in_bytes / (1024 * 1024)  # Chuyển sang MB
    print(f"Tensor size: {size_in_MB:.2f} MB")


@app.post("/search_img")
async def searchImg(
    image: bytes = File(...),
    company_name: str = Form(...) , 
    keywords: str = Form(...)   
):
   
    try:
        pixel_values = None
        en_vec = imgEncoder.encodeImage(io.BytesIO(image))
        if en_vec is not None:
            res = searchVector(en_vec.squeeze().astype(np.float32).tolist(), company_name)
            # print(res[0][0].distance)
            # print("ent: "+str(res[0][0].entity.text))
            if res[0][0].distance < 0.95:
                return {"is_valid": False}
            # prompt = '''<image>\nOCR hết từ trên xuống nhét hết nội dung vào:
            # [
            # {
            #     "Content": "Nội dung",
            # },
            # ]x
            # '''
            prompt = """
            Hãy nhận diện chữ trong ảnh này (OCR) và trích xuất tất cả nội dung ra file JSON sau:
            {
                "Content": "...",
            }
            Nếu không có thông tin, hãy trả về giá trị null.
            """
            pixel_values = Image.open(io.BytesIO(image))
            # pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda(non_blocking=True)
            # caculateMem(pixel_values)
            # with torch.no_grad():
            text_test = ocrScanner.scan(pixel_values, prompt)
            del pixel_values
            # torch.cuda.empty_cache()  
            # torch.cuda.synchronize()
            gc.collect()
            if text_test == res[0][0].entity.text and not contains_keywords(text_test, parse_keywords(keywords)):
                return {"is_valid": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding image: {str(e)}")
    finally:
        if 'pixel_values' in locals() and pixel_values is not None:
            del pixel_values
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

    return {"is_valid": False}

@app.post("/add_img")
async def addImg(    
    image: bytes = File(...),   
    company_name: str = Form(...)   
    ):


    # encoded_vectors = []
    # for file in files:
    try:
        # Reading the binary content of the file
        # file_content = decode_base64_image(file)
        # Convert the binary content to a PIL Image
        en_vec = imgEncoder.encodeImage(io.BytesIO(image))
        print(en_vec.squeeze().shape)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")
    # if len(encoded_vectors) == 3:
    # Insert the user data, assuming encoded_vectors are processed correctly
    pixel_values = Image.open(io.BytesIO(image))
    # pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()
    # prompt = '''<image>\nOCR hết từ trên xuống nhét hết nội dung vào:
    # [
    # {
    #     "Content": "Nội dung",
    # },
    # ]
    # '''
    prompt = """
    Hãy nhận diện chữ trong ảnh này (OCR) và trích xuất tất cả nội dung ra file JSON sau:
    {
        "Content": "...",
    }
    Nếu không có thông tin, hãy trả về giá trị null.
    """
    # with torch.inference_mode():
    text = ocrScanner.scan(pixel_values, prompt)
    del pixel_values
    # torch.cuda.empty_cache() 
    # torch.cuda.synchronize()
    print(len(text["Content"]))
    data = [{"company_name": company_name, "embedding": en_vec.squeeze().astype(np.float32).tolist(), "text": text["Content"]}]
    print("inserting")
    success = insertVector(data)
    print("inserted")
    if success:
        return {"name": company_name, "status": "registered successfully"}
    else:
        return {"error": "Error inserting user data."}
    
    
    # else:
    #     return {"error": "Failed to encode all images."}
    
@app.post("/get_invalid_keyword/url")
async def get_invalid_keyword(url: str):
    DEFAULT_KEYWORDS = ["mới nhất", "hiện đại nhất", "độc quyền", "duy nhất", "hoàn toàn", "nhất", "hoàn toàn 100%"]
    if not url:
        return {"error": "Invalid URL."}
    else:
        article = ac.crawl_and_clean_article(url=url)

        filtered_keywords = ue.extract_keywords(article["content"], DEFAULT_KEYWORDS)
        return {"data": filtered_keywords}






# path ="D:/333.jpg"
# model, tokenizer, generation_config  = vintern_1b_v2()
# o_p = OcrPipeLine(model, tokenizer, generation_config)
# pixel_values = load_image(path, max_num=6).to(torch.bfloat16).cuda()
# print(o_p.scan(pixel_values))

# path ="D:/OneDrive - Trường ĐH CNTT - University of Information Technology/Ảnh/434558165_1089114642207289_1270081782924023452_n.jpg"
# model = VGG16(include_top=True, weights='imagenet')
# im_e = ImageEncoderPipeLine(model, preprocess_input)

# print(im_e.encodeImage(path))
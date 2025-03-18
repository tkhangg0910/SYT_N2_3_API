from tensorflow.keras.preprocessing import image
import numpy as np
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vintern_1b_v2 import load_image

class ImageEncoderPipeLine(): 
    def __init__(self,model, preprocesesor):
        self.model = model
        self.preprocesesor = preprocesesor
    def encodeImage(self, img):
        img = image.load_img(img, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocesesor(x)
        preds = self.model.predict(x, verbose = 0)
        return preds
    
class OcrPipeLine:
    def __init__(self, model, tokenizer=None, generation_config=None, model_type="vintern"):
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.prompt = """Hãy nhận diện chữ trong ảnh quảng cáo y tế  này (OCR) và trích xuất thông tin sản phẩm theo định dạng JSON sau:
        {
            "Tên công ty": "...",
            "Tên sản phẩm": "...",
            "Nội dung quảng cáo": "..."
        }
        Nếu không có thông tin, hãy trả về giá trị null.
        """

    
    def scan(self, pixel_values, prompt=None):
        prompt = prompt if prompt is not None else self.prompt

        if self.model_type == "vintern":
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt, generation_config=self.generation_config
            )
        elif self.model_type == "gemini":
            response = self._query_gemini(pixel_values, prompt)
            response = response.strip("```json").strip("```").strip()
        else:
            raise ValueError("Unsupported model type: Choose 'vintern' or 'gemini'.")

        print(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error decoding JSON response:", response)
            return response  # Return raw response if JSON parsing fails
    
    def _query_gemini(self, pixel_values, prompt):
        """Gửi dữ liệu ảnh và prompt tới Gemini API"""
       
        gemini_response = self.model.generate_content([pixel_values, prompt])

        return gemini_response.text


        
    
    
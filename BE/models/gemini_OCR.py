import google.generativeai as genai
from google.genai import types
def gemini_ocr(API_KEY):
    genai.configure(api_key=API_KEY)
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    return model
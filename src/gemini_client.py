# src/gemini-client.py

import google.generativeai as genai

class GeminiClient:
    def __init__(self, api_key: str, model_name: str = 'gemini-pro'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str) -> str:
        """
        Gửi prompt đến mô hình Gemini và trả về câu trả lời.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Lỗi khi gọi Gemini API: {e}")
            return None
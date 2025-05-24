import time
import logging
from typing import List, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import GeminiConfig

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.setup_client()
        
    def setup_client(self):
        """Initialize Gemini client with configuration"""
        try:
            genai.configure(api_key=self.config.api_key)
            
            # Configure model with safety settings
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Initialized Gemini client with model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """Generate answer using Gemini API with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        raise Exception("Failed to get response from Gemini after all retries")
    
    def generate_paraphrased_questions(self, original_question: str) -> List[str]:
        """Generate paraphrased versions of the original question"""
        prompt = f"""
    [NHIỆM VỤ]: Viết lại câu hỏi sau theo 3 cách diễn đạt khác nhau nhưng vẫn giữ nguyên ý nghĩa.
    [CÂU HỎI GỐC]: {original_question.strip()}

    Vui lòng chỉ cung cấp 3 câu hỏi đã được viết lại, mỗi câu kết thúc bằng dấu chấm hỏi.
    """
   
        
        try:
            response = self.generate_answer(prompt)
            
            # Parse the response to extract individual questions
            questions = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and line.endswith('?'):
                    # Remove numbering or bullet points
                    clean_line = line
                    for prefix in ['1.', '2.', '3.', '-', '*', '•']:
                        if clean_line.startswith(prefix):
                            clean_line = clean_line[len(prefix):].strip()
                    questions.append(clean_line)
            
            # Fallback: split by question marks if parsing fails
            if not questions:
                questions = [q.strip() + '?' for q in response.split('?') if q.strip()]
                questions = questions[:-1] if len(questions) > 3 else questions
            
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            logger.error(f"Failed to generate paraphrased questions: {e}")
            return []  # Return empty list on failure
    
    def health_check(self) -> bool:
        """Check if Gemini API is accessible"""
        try:
            test_prompt = "Xin chào, đây là kiểm tra nếu ổn hãy trả lời 'OK'."
            response = self.generate_answer(test_prompt)
            return "OK" in response or len(response) > 0
        except:
            return False
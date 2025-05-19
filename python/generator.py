from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import Config

class AnswerGenerator:
    def __init__(self):
        self.config = Config()
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Load a Vietnamese-optimized causal LM"""
        try:
            print("🔄 Loading Vietnamese GPT model...")
            
            model_name = "vinai/PhoGPT-7B5"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32
            )
            
            print(f"✅ Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            print("🔄 Falling back to smaller model...")
            try:
                model_name = "VietAI/gpt-j-6B-vietnamese-news"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                return model, tokenizer
            except:
                print("⚠️ Using echo response fallback")
                return None, None
    
    def generate(self, question: str, context: str) -> str:
        if not self.model or not self.tokenizer:
            return "Xin lỗi, hệ thống đang gặp sự cố"
        
        # Vietnamese-optimized prompt template
        prompt = f"""
        ### Câu hỏi:
        {question}

        ### Ngữ cảnh:
        {context}

        ### Trả lời:
        """.strip()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                  truncation=True, max_length=1024)
            
            if self.config.DEVICE == "cuda":
                inputs = inputs.to("cuda")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                early_stopping=True
            )
            
            # Decode and clean output
            answer = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            answer = answer.split("### Trả lời:")[-1].strip()
            
            # Post-process Vietnamese output
            answer = answer.replace(" .", ".").replace(" ,", ",")
            return answer if answer else "Không thể tạo câu trả lời"
            
        except Exception as e:
            return f"Lỗi khi tạo câu trả lời: {str(e)}"
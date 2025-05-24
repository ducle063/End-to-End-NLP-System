import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    EMBEDDING_MODEL_NAME = "all-minilm-l6-v2"
    if not API_KEY:
        raise ValueError("Vui lòng đặt biến môi trường GEMINI_API_KEY.")
    LLM_MODEL_NAME = "gemini-1.5-flash-latest"
    TOP_K = 3
    DATA_DIR = "../data/QnA"
    TRAIN_QUESTIONS_FILE = os.path.join(DATA_DIR, "train", "train_questions.txt")
    TRAIN_ANSWERS_FILE = os.path.join(DATA_DIR, "train", "train_answers.txt")
    TEST_QUESTIONS_FILE = os.path.join(DATA_DIR, "demo", "test_questions.txt")
    TEST_ANSWERS_FILE = os.path.join(DATA_DIR, "demo", "test_references.txt")

#Usage example:
if __name__ == "__main__":
    config = Config()
    print(f"API Key: {config.API_KEY}")
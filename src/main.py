# src/main.py

import os
import json
import datetime
import time
from evaluate import evaluate_answer
from embedder import Embedder
from retriever import Retriever
from prompt_builder import build_prompt
from gemini_client import GeminiClient
from evaluate import evaluate_batch
from config import Config



def load_lines(file_path):
    """Đọc các dòng từ file text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return None

def main():
    # Khởi tạo các thành phần
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_results_{timestamp}.json"
    results = []

    config = Config()
    if not config.API_KEY:
        raise ValueError("Vui lòng đặt biến môi trường GEMINI_API_KEY.")
    embedder = Embedder(embedding_model_name=config.EMBEDDING_MODEL_NAME)
    retriever = Retriever(embedding_model_name=config.EMBEDDING_MODEL_NAME, collection_name="reference_docs") 
    gemini_client = GeminiClient(api_key=config.API_KEY, model_name=config.LLM_MODEL_NAME)

    # Load dữ liệu train
    train_questions = load_lines(config.TRAIN_QUESTIONS_FILE)
    train_answers = load_lines(config.TRAIN_ANSWERS_FILE)

    train_data = []
    if train_questions and train_answers and len(train_questions) == len(train_answers):
        for q, a in zip(train_questions, train_answers):
            train_data.append({"question": q, "answer": a})
        print(f"Đã tải {len(train_data)} cặp hỏi-đáp từ dữ liệu train.")
    else:
        print("Không thể tải hoặc dữ liệu train không khớp.")
        train_data = None

    # Sử dụng câu trả lời làm documents để tạo embedding (hoặc bạn có thể có documents riêng)
    # train_documents = train_answers

    # Tạo embeddings cho training documents và thêm vào retriever
    # document_embeddings = embedder.embed_documents(train_documents)
    # Giả sử retriever có hàm add_embeddings(embeddings, documents)
    # retriever.add_embeddings(document_embeddings, train_documents)

    # Load dữ liệu test
    test_questions = load_lines(config.TEST_QUESTIONS_FILE)
    test_answers = load_lines(config.TEST_ANSWERS_FILE)

    if not test_questions or not test_answers or len(test_questions) != len(test_answers):
        print("Lỗi: Không thể tải hoặc số lượng câu hỏi và câu trả lời trong test không khớp.")
        return

    print("--- Bắt đầu đánh giá trên tập test ---")
    predicted_answers = []
    ground_truth_answers = test_answers

    for i, question in enumerate(test_questions):
        print(f"\nCâu hỏi: {question}")

        # Truy xuất các tài liệu liên quan
        relevant_documents = retriever.retrieve(question, top_k=config.TOP_K)
        print(f"Ngữ cảnh truy xuất được: {relevant_documents}")

        # Chọn một vài ví dụ train ngẫu nhiên để làm few-shot
        few_shot_examples = []
        if train_data:
            import random
            num_few_shot = min(3, len(train_data)) # Ví dụ: lấy tối đa 3 ví dụ
            few_shot_examples = random.sample(train_data, num_few_shot)

        # Xây dựng prompt
        prompt = build_prompt(question, relevant_documents, few_shot_examples=few_shot_examples)

        # Nhận câu trả lời từ Gemini
        predicted_answer = gemini_client.generate_response(prompt)
        print(f"Câu trả lời dự đoán: {predicted_answer}")
        predicted_answers.append(predicted_answer)

        # Đánh giá từng câu trả lời (kiểm tra None trước)
        evaluation = {}
        if predicted_answer is not None:
            evaluation = evaluate_answer(predicted_answer, ground_truth_answers[i])
            print("Đánh giá:", evaluation)
        else:
            print("Không nhận được câu trả lời từ Gemini, không thể đánh giá.")
            evaluation = {"exact_match": 0, "f1": 0} # Gán giá trị mặc định

        results.append({
            "question": question,
            "relevant_documents": relevant_documents,
            "predicted_answer": predicted_answer,
            "ground_truth_answer": ground_truth_answers[i],
            "evaluation": evaluation
        })

    time.sleep(30)

    # Đánh giá trên toàn bộ tập test
    overall_evaluation = evaluate_batch(predicted_answers, ground_truth_answers)
    print("\n--- Kết quả đánh giá trên tập test ---")
    print(f"Exact Match (Average): {overall_evaluation['exact_match']:.4f}")
    print(f"F1 Score (Average): {overall_evaluation['f1']:.4f}")

    # Lưu kết quả vào file JSON trong folder "system_outputs"
    output_folder = "system_outputs"
    os.makedirs(output_folder, exist_ok=True)  # Tạo folder nếu chưa tồn tại
    output_path = os.path.join(output_folder, output_filename)

    output_data = {
        "timestamp": timestamp,
        "config": {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "llm_model": config.LLM_MODEL_NAME,
            "top_k": config.TOP_K
        },
        "individual_results": results,
        "overall_evaluation": overall_evaluation
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nKết quả đánh giá đã được lưu vào file: {output_path}")

if __name__ == "__main__":
    main()
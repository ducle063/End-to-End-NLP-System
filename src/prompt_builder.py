def build_prompt(question: str, context: list[str], few_shot_examples: list[dict] = None) -> str:
    """
    Xây dựng prompt bằng cách kết hợp câu hỏi, ngữ cảnh và (tùy chọn) few-shot examples.
    """
    prompt = ""
    if few_shot_examples:
        for example in few_shot_examples:
            prompt += f"Câu hỏi: {example['question']}\n"
            prompt += f"Trả lời: {example['answer']}\n\n"

    prompt += f"Dựa vào thông tin sau đây, hãy trả lời câu hỏi:\n"
    prompt += f"Context:\n"
    for doc in context:
        prompt += f"{doc}\n"
    prompt += f"\nCâu hỏi: {question}\n"
    prompt += "Trả lời:"
    return prompt

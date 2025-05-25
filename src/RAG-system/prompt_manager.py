from typing import List
from langchain.schema import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document

from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.chains.combine_documents.stuff import format_document
from typing import List

class PromptManager:
    def __init__(self):
        self.document_prompt = PromptTemplate.from_template(template="{page_content}")

    def get_basic_prompt(self) -> str:
        """Basic RAG prompt template in Vietnamese"""
        return """
Bạn là một trợ lý cho các tác vụ hỏi đáp và các câu hỏi liên quan đến Trường đại học Quốc Gia Hà Nội và các trường thành viên. Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. Không vượt quá một câu trả lời. Trả lời trực tiếp ngay cả khi câu trả lời không mạch lạc.
"""

    def get_few_shot_prompt(self) -> str:
        """Few-shot prompt template with examples about UET in Vietnamese"""
        return """
Bạn là một trợ lý cho các tác vụ hỏi đáp liên quan đến Trường Đại học Công nghệ (UET), Đại học Quốc gia Hà Nội. Sử dụng các đoạn ngữ cảnh được truy xuất để trả lời câu hỏi. Trả lời ngắn gọn và trực tiếp nhất đáp án, không cần chủ ngữ, vị ngữ.
Nếu có câu hỏi về số lượng, năm, tên, địa chỉ, chức vụ, ngành học, điểm chuẩn, khoa, chương trình đào tạo, chỉ cần trả lời chính xác con số, địa chỉ, tên ngành-khoa theo thông tin đã cho. Nếu không có thông tin trong ngữ cảnh, hãy trả lời "Không có thông tin".

Dưới đây là một vài ví dụ về câu hỏi và câu trả lời liên quan đến UET:

Câu hỏi: UET được thành lập vào năm nào?
Trả lời: 2004

Câu hỏi: Hiệu trưởng hiện tại của UET là ai?
Trả lời: GS. TS Chử Đức Trình

Câu hỏi: Trường Đại học Công nghệ trực thuộc đơn vị nào?
Trả lời: Đại học Quốc gia Hà Nội

Câu hỏi: Số lượng ngành đào tạo trình độ đại học của UET là bao nhiêu?
Trả lời: 12

Câu hỏi: UET có chương trình đào tạo nào bằng tiếng Anh không?
Trả lời: Có

Câu hỏi: Điểm chuẩn ngành Công nghệ thông tin năm 2023 của UET là bao nhiêu?
Trả lời: 28.25

Câu hỏi: Ngành Trí tuệ nhân tạo tại UET thuộc khoa nào?
Trả lời: Khoa Công nghệ thông tin

Câu hỏi: UET có bao nhiêu khoa chuyên môn?
Trả lời: 7

Câu hỏi: Tên tiếng Anh chính thức của UET là gì?
Trả lời: University of Engineering and Technology

Câu hỏi: Địa chỉ trụ sở chính của UET là ở đâu?
Trả lời: 144 Xuân Thủy, Cầu Giấy, Hà Nội
"""

    def get_paraphrase_prompt(self, question: str) -> str:
        """Prompt for generating paraphrased questions in Vietnamese"""
        return f"""
                [NHIỆM VỤ]: Viết lại câu hỏi sau theo ba cách diễn đạt khác nhau.
                [CÂU HỎI GỐC]: {question}
                """

    def combine_documents(self, docs: List[Document], document_separator: str = "\n\n") -> str:
        """Combine documents into a single context string"""
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    def build_qa_prompt(self, question: str, context: str, use_few_shot: bool = False) -> str:
        """Build complete QA prompt with question and context in Vietnamese"""
        if use_few_shot:
            prompt_start = self.get_few_shot_prompt()
        else:
            prompt_start = self.get_basic_prompt()

        return f"{prompt_start}Câu hỏi: {question}Ngữ cảnh: {context}Trả lời: "

    def build_multi_query_prompt(self, question: str) -> str:
        """Build prompt for multi-query generation in Vietnamese"""
        return f"""
[NHIỆM VỤ]: Viết câu hỏi dưới đây theo 3 cách khác nhau.
[CÂU HỎI]: {question}
"""
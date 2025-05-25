import os
import re
import logging
import string
import unicodedata
import glob
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys

# Try to import pyvi instead of underthesea
try:
    from pyvi import ViTokenizer, ViPosTagger, ViUtils
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thêm danh sách từ dừng tiếng Việt
vi_stopwords = {
    "và", "của", "cho", "là", "không", "được", "có", "những", "các", "một", "này", "đã", 
    "với", "trong", "về", "như", "từ", "trên", "theo", "để", "đến", "tới", "thì", "mà", 
    "tại", "vì", "nên", "khi", "nếu", "bởi", "qua", "từng", "cùng", "vào", "hoặc", "hay",
    "tất", "cả", "rất", "thế", "còn", "bị", "bởi", "qua", "lại", "đang", "sẽ", "nhưng",
    "mới", "sau", "trước", "đó", "này", "kia", "thế", "do", "vẫn", "đều", "đi", "chỉ",
    "được", "phải", "những", "cũng", "ra", "thì", "lên", "xuống", "làm", "nói", "biết",
    "thấy", "nhận", "đang", "nên", "muốn", "thật", "hai", "ba", "bốn", "năm", "mình", "tôi",
    "chúng", "họ", "bạn", "anh", "chị", "tất", "cả", "ông", "bà", "em", "thôi", "vậy", "nhé",
    "à", "ừ", "ờ", "ý", "ạ", "nhỉ", "thế", "thì", "đi", "nào", "nha", "vào", "bằng", "hơn", "quá"
}

# Danh sách từ dừng tiếng Anh cơ bản
en_stopwords = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while", "of", "at", 
    "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", 
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", 
    "shouldn", "wasn", "weren", "won", "wouldn"
}

# Danh sách các ký tự đặc trưng tiếng Việt để phát hiện ngôn ngữ
VI_CHARS = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"

# Hàm phát hiện ngôn ngữ
def detect_language(text, min_words=10):
    """
    Phát hiện ngôn ngữ của văn bản dựa trên tỷ lệ ký tự đặc trưng và từ đặc biệt
    """
    if not text or len(text.split()) < min_words:
        return "unknown"
    
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    if not words:
        return "unknown"
    
    # Đếm số ký tự tiếng Việt trong văn bản
    vi_char_count = sum(1 for c in text if c.lower() in VI_CHARS)
    text_len = len(text)
    
    # Nếu tỷ lệ ký tự tiếng Việt cao, đây có khả năng là văn bản tiếng Việt
    if text_len > 0 and vi_char_count / text_len > 0.05:
        return "vietnamese"
    
    # Tính tỷ lệ từ khớp với từ dừng
    en_match = sum(1 for word in words if word in en_stopwords) / len(words) if words else 0
    vi_match = sum(1 for word in words if word in vi_stopwords) / len(words) if words else 0
    
    # Heuristic để phát hiện ngôn ngữ
    if en_match > 0.1 and en_match > vi_match:
        return "english"
    elif vi_match > 0.1 and vi_match > en_match:
        return "vietnamese"
    else:
        # Kiểm tra văn bản có chứa ký tự đặc trưng tiếng Việt không
        if any(c in text for c in VI_CHARS):
            return "vietnamese"
        else:
            return "english"

# Tách câu tiếng Việt cơ bản khi pyvi không khả dụng
def basic_sent_tokenize(text):
    """Phương pháp tách câu cơ bản khi không có pyvi"""
    # Chuẩn hóa dấu câu
    text = re.sub(r'\.{2,}', '.', text)  # Loại bỏ dấu chấm liên tiếp
    text = re.sub(r'([.!?;:])([^\s])', r'\1 \2', text)  # Thêm khoảng trắng sau dấu câu
    
    # Tách câu bằng regex cho các dấu câu phổ biến
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Lọc các câu rỗng
    return [sent.strip() for sent in sentences if sent.strip()]

# Lớp TextPreprocessor chính
class TextPreprocessor:
    def __init__(self, input_dir, output_dir, min_words=None, max_files=None):
        """
        Khởi tạo bộ tiền xử lý văn bản
        
        Args:
            input_dir (str): Đường dẫn thư mục chứa các file txt cần tiền xử lý
            output_dir (str): Đường dẫn thư mục lưu kết quả
            min_words (int): Số từ tối thiểu trong một câu để giữ lại
            max_files (int, optional): Số lượng file tối đa cần xử lý, None nếu xử lý tất cả
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_words = min_words
        self.max_files = max_files
        
        # Kiểm tra pyvi
        if not PYVI_AVAILABLE:
            logger.warning("Thư viện pyvi không được cài đặt. Vui lòng cài đặt bằng lệnh: pip install pyvi")
            logger.warning("Chương trình sẽ sử dụng phương pháp xử lý đơn giản thay thế.")
        else:
            logger.info("Đã phát hiện thư viện pyvi. Sẽ sử dụng cho xử lý tiếng Việt.")
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    
    def clean_text(self, text, is_vietnamese=False):
        """
        Làm sạch văn bản: loại bỏ HTML, chuẩn hóa khoảng trắng,
        loại bỏ ký tự đặc biệt, v.v.
        """
        
        
        # Loại bỏ HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Chuẩn hóa unicode (quan trọng cho tiếng Việt)
        if is_vietnamese:
            text = unicodedata.normalize('NFC', text)
        
        # Loại bỏ ký tự đặc biệt và số, nhưng giữ lại dấu câu cần thiết và ký tự tiếng Việt
        if is_vietnamese:
            # Giữ lại các ký tự tiếng Việt khi lọc
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\"\'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+', ' ', text)
        else:
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\"\']+', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Chuẩn hóa dấu câu (khoảng trắng sau dấu câu)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text
    
    def normalize_vietnamese(self, text):
        """
        Chuẩn hóa văn bản tiếng Việt, xử lý các vấn đề cụ thể với dấu
        """
        # Chuẩn hóa dấu theo kiểu Việt Nam
        text = unicodedata.normalize('NFC', text)
        
        # Sửa lỗi các chữ 'đ/Đ' bị mã hóa sai
        text = text.replace('đ', 'đ')
        text = text.replace('Đ', 'Đ')
        
        # Xử lý các lỗi cụ thể trong tiếng Việt
        # Ghép các từ bị tách không đúng
        text = re.sub(r'(\w+) - (\w+)', r'\1-\2', text)
        
        # Chuẩn hóa dấu gạch ngang trong các từ ghép
        text = re.sub(r' - ', ' – ', text)
        
        return text
    
    def segment_sentences(self, text, language="english"):
        """
        Tách văn bản thành các câu dựa trên ngôn ngữ.
        """
        # Sửa lỗi phổ biến trước khi tách câu
        text = re.sub(r'\.\.+', '.', text)  # Loại bỏ dấu chấm liên tiếp
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Sửa lỗi thiếu khoảng trắng sau dấu chấm

        if language == "vietnamese":
            if PYVI_AVAILABLE:
                try:
                    # Sử dụng regex cơ bản kết hợp với những đặc điểm của tiếng Việt
                    text = re.sub(r'([.!?])\s*', r'\1\n', text)
                    sentences = [s.strip() for s in text.split('\n') if s.strip()]
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng phương pháp tách câu với pyvi: {e}")
                    sentences = basic_sent_tokenize(text)
            else:
                sentences = basic_sent_tokenize(text)
        else:  # language == "english" hoặc các trường hợp khác
            sentences = basic_sent_tokenize(text)

        # Lọc các câu quá ngắn
        filtered_sentences = []
        for sentence in sentences:
            # Làm sạch câu
            clean_sentence = sentence.strip()

            # Kiểm tra số từ trong câu
            if len(clean_sentence.split()) >= self.min_words:
                filtered_sentences.append(clean_sentence)

        return filtered_sentences
    
    def tokenize(self, text, language="english"):
        """
        Tách từ trong văn bản, phục vụ cho việc phân tích sau này
        """
        if language == "vietnamese" and PYVI_AVAILABLE:
            try:
                # Sử dụng pyvi để tách từ tiếng Việt
                text_lower = text.lower()
                tokenized_text = ViTokenizer.tokenize(text_lower)
                # Trong pyvi, kết quả trả về là chuỗi với từ đã tách được nối bằng dấu gạch dưới
                # Ví dụ: "Tôi_là sinh_viên" -> sẽ đổi thành list ["Tôi_là", "sinh_viên"]
                tokens = tokenized_text.split()
                # Loại bỏ từ dừng - cần kiểm tra cả từ có dấu gạch dưới và không có
                tokens = [token for token in tokens if token not in vi_stopwords and 
                         not any(token.startswith(sw + "_") or token.endswith("_" + sw) for sw in vi_stopwords)]
                # Loại bỏ dấu câu đơn lẻ
                tokens = [token for token in tokens if token not in string.punctuation]
                return tokens
            except Exception as e:
                logger.warning(f"Lỗi khi sử dụng pyvi để tách từ: {e}")
        
        # Phương pháp đơn giản cho cả tiếng Anh và tiếng Việt (nếu pyvi không khả dụng)
        # Tiền xử lý
        text = text.lower()
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tách từ cơ bản
        words = text.split()
        
        # Loại bỏ từ dừng
        if language == "vietnamese":
            words = [word for word in words if word not in vi_stopwords]
        else:
            words = [word for word in words if word not in en_stopwords]
        
        # Loại bỏ dấu câu đơn lẻ
        words = [word for word in words if word not in string.punctuation]
        
        return words
    
    def process_file(self, file_path):
        """
        Xử lý một file văn bản
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Tách phần header (URL, title) và nội dung chính
            content_parts = content.split('-'*80 + "\n", 1)
            header = content_parts[0]
            
            if len(content_parts) > 1:
                main_content = content_parts[1]
            else:
                main_content = ""
            
            ## Phát hiện ngôn ngữ
            language = detect_language(main_content)
            is_vietnamese = (language == "vietnamese")
            cleaned_content = self.clean_text(main_content, is_vietnamese)

            
            # Tách câu
            sentences = self.segment_sentences(cleaned_content, language = language)
            
            # Nếu không có câu nào sau khi lọc, trả về None
            if not sentences:
                logger.warning(f"Không có đủ nội dung hợp lệ trong file {os.path.basename(file_path)}")
                return None, None
            
            # Tạo nội dung đã xử lý
            processed_content = header + "\n\n" + "\n".join(sentences)
            
            # Thống kê cơ bản
            stats = {
                "file": os.path.basename(file_path),
                "language": language,
                "original_chars": len(main_content),
                "processed_chars": len(cleaned_content),
                "original_words": len(main_content.split()),
                "processed_words": len(cleaned_content.split()),
                "sentences": len(sentences)
            }
            
            return processed_content, stats
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file {file_path}: {e}")
            return None, None
    
    def get_file_output_path(self, original_path, output_base_dir):
        """
        Xác định đường dẫn file đầu ra, lưu vào folder tương ứng với folder đầu vào.
        """
        relative_path = os.path.relpath(os.path.dirname(original_path), self.input_dir)
        output_sub_dir = os.path.join(self.output_dir, relative_path)
        os.makedirs(output_sub_dir, exist_ok=True)
        base_name = os.path.basename(original_path)
        return os.path.join(output_sub_dir, base_name)

    
    def run(self, parallel=True):
        """
        Chạy tiền xử lý trên tất cả các file trong thư mục đầu vào
        
        Args:
            parallel (bool): Sử dụng xử lý song song nếu True
        """
        # Tìm tất cả các file txt trong thư mục đầu vào và các thư mục con
        all_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.txt'):
                    all_files.append(os.path.join(root, file))
        
        logger.info(f"Tìm thấy {len(all_files)} file txt trong thư mục {self.input_dir}")
        
        # Giới hạn số lượng file cần xử lý nếu có
        if self.max_files and len(all_files) > self.max_files:
            all_files = all_files[:self.max_files]
            logger.info(f"Giới hạn xử lý {self.max_files} file")
        
        # Thống kê tổng hợp
        total_stats = {
            "total_files": len(all_files),
            "processed_files": 0,
            "english_files": 0,
            "vietnamese_files": 0,
            "other_files": 0,
            "total_original_chars": 0,
            "total_processed_chars": 0,
            "total_original_words": 0,
            "total_processed_words": 0,
            "total_sentences": 0
        }
        
        # Xử lý song song
        if parallel and len(all_files) > 1:
            logger.info("Bắt đầu xử lý song song")
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(self.process_file, all_files))
        else:
            # Xử lý tuần tự
            logger.info("Bắt đầu xử lý tuần tự")
            results = [self.process_file(file) for file in all_files]
        
        # Ghi kết quả và cập nhật thống kê
        for idx, (processed_content, stats) in enumerate(results):
            if processed_content is None or stats is None:
                continue
            
            # Cập nhật thống kê
            total_stats["processed_files"] += 1
            
            if stats["language"] == "english":
                total_stats["english_files"] += 1
            elif stats["language"] == "vietnamese":
                total_stats["vietnamese_files"] += 1
            else:
                total_stats["other_files"] += 1
            
            total_stats["total_original_chars"] += stats["original_chars"]
            total_stats["total_processed_chars"] += stats["processed_chars"]
            total_stats["total_original_words"] += stats["original_words"]
            total_stats["total_processed_words"] += stats["processed_words"]
            total_stats["total_sentences"] += stats["sentences"]
            
            # Ghi file đã xử lý
            original_file = all_files[idx]
            output_path = self.get_file_output_path(original_file, stats["language"])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
        
        # Hiển thị thống kê tổng hợp
        logger.info("\n--- Thống kê tổng hợp ---")
        logger.info(f"Tổng số file: {total_stats['total_files']}")
        logger.info(f"Số file đã xử lý: {total_stats['processed_files']}")
        logger.info(f"- File tiếng Anh: {total_stats['english_files']}")
        logger.info(f"- File tiếng Việt: {total_stats['vietnamese_files']}")
        logger.info(f"- File ngôn ngữ khác: {total_stats['other_files']}")
        
        if total_stats['processed_files'] > 0:
            avg_reduction = 100 * (1 - total_stats["total_processed_chars"] / total_stats["total_original_chars"])
            logger.info(f"Tổng số ký tự ban đầu: {total_stats['total_original_chars']}")
            logger.info(f"Tổng số ký tự sau xử lý: {total_stats['total_processed_chars']}")
            logger.info(f"Tỷ lệ giảm: {avg_reduction:.2f}%")
            logger.info(f"Tổng số từ ban đầu: {total_stats['total_original_words']}")
            logger.info(f"Tổng số từ sau xử lý: {total_stats['total_processed_words']}")
            logger.info(f"Tổng số câu: {total_stats['total_sentences']}")
        
        # Tạo file tóm tắt
        with open(os.path.join(self.output_dir, "preprocessing_stats.txt"), 'w', encoding='utf-8') as f:
            f.write("--- THỐNG KÊ TIỀN XỬ LÝ VĂN BẢN ---\n\n")
            for key, value in total_stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Đã hoàn thành tiền xử lý và lưu kết quả vào {self.output_dir}")
        return total_stats


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu văn bản từ các file txt')
    parser.add_argument('--input', '-i', type=str, default='/workspaces/End-to-End-NLP-System/data/extra-info/', 
                        help='Thư mục chứa dữ liệu đầu vào')
    parser.add_argument('--output', '-o', type=str, default='/workspaces/End-to-End-NLP-System/data/clean/extra-info/', 
                        help='Thư mục lưu kết quả')
    parser.add_argument('--min-words', type=int, default=3, 
                        help='Số từ tối thiểu trong một câu để giữ lại')
    parser.add_argument('--max-files', type=int, default=None, 
                        help='Số lượng file tối đa cần xử lý')
    parser.add_argument('--sequential', action='store_true', 
                        help='Chạy xử lý tuần tự thay vì song song')
    
    args = parser.parse_args()
    
    # Thông báo về pyvi
    if not PYVI_AVAILABLE:
        logger.warning("Thư viện pyvi không được cài đặt.")
        logger.warning("Để cài đặt pyvi: pip install pyvi")
        logger.warning("Chương trình sẽ tiếp tục với phương pháp xử lý đơn giản.")
    
    # Khởi tạo và chạy bộ tiền xử lý
    preprocessor = TextPreprocessor(
        input_dir=args.input,
        output_dir=args.output,
        min_words=args.min_words,
        max_files=args.max_files
    )
    
    preprocessor.run(parallel=not args.sequential)


if __name__ == "__main__":
    main()
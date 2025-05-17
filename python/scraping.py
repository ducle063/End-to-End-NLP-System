import requests
from bs4 import BeautifulSoup
import re
import csv
import os
import time
import logging
import concurrent.futures
from urllib.parse import urljoin, urlparse
from collections import defaultdict

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HTMLScraper:
    def __init__(self, urls, output_dir="../data/raw/VNU", max_depth=2, delay=1, max_workers=5):
        """
        Khởi tạo scraper
        
        Args:
            urls (list): Danh sách các URL cần thu thập dữ liệu
            output_dir (str): Thư mục lưu kết quả
            max_depth (int): Độ sâu tối đa khi thu thập các trang con
            delay (int): Thời gian chờ giữa các request (giây)
            max_workers (int): Số lượng worker tối đa cho xử lý đồng thời
        """
        self.urls = urls
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.delay = delay
        self.max_workers = max_workers
        self.visited_urls = set()
        self.data = defaultdict(list)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_domain(self, url):
        """Lấy tên miền từ URL"""
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return domain
    
    def fetch_url(self, url):
        """Tải nội dung từ URL và trả về BeautifulSoup object"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Tìm encoding phù hợp
            if response.encoding == 'ISO-8859-1':
                encodings = ['utf-8', 'cp1252']
                for enc in encodings:
                    try:
                        response.encoding = enc
                        response.text.encode(enc)
                        break
                    except UnicodeEncodeError:
                        continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi truy cập {url}: {e}")
            return None
    
    def extract_links(self, soup, base_url):
        """Trích xuất các liên kết từ trang web"""
        links = []
        domain = self.get_domain(base_url)
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            # Bỏ qua các liên kết không hợp lệ
            if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
            
            # Tạo URL tuyệt đối
            full_url = urljoin(base_url, href)
            
            # Chỉ lấy các URL cùng domain
            if full_url.startswith(domain):
                links.append(full_url)
        
        return links
    
    def extract_data(self, soup, url):
        """
        Trích xuất dữ liệu từ trang web
        Có thể tùy chỉnh hàm này tùy theo yêu cầu cụ thể
        """
        data = {
            'url': url,
            'title': soup.title.text.strip() if soup.title else 'No Title',
            'meta_description': '',
            'headers': [],
            'text_content': '',
            'links': []
        }
        
        # Lấy meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            data['meta_description'] = meta_desc['content'].strip()
        
        # Lấy các thẻ tiêu đề h1, h2
        headers = []
        for h_tag in soup.find_all(['h1', 'h2']):
            header_text = h_tag.text.strip()
            if header_text:
                headers.append(f"{h_tag.name}: {header_text}")
        data['headers'] = headers
        
        # Lấy nội dung văn bản (loại bỏ các script, style)
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        data['text_content'] = text
        
        # Lấy danh sách các liên kết
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            text = a_tag.text.strip()
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                links.append({'href': href, 'text': text})
        data['links'] = links
        
        return data
    
    def crawl_url(self, url, depth=0):
        """Thu thập dữ liệu từ một URL và các trang con"""
        if depth > self.max_depth or url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        logger.info(f"Đang thu thập từ URL: {url} (độ sâu: {depth})")
        
        soup = self.fetch_url(url)
        if not soup:
            return []
        
        # Trích xuất dữ liệu từ trang hiện tại
        data = self.extract_data(soup, url)
        domain = urlparse(url).netloc
        self.data[domain].append(data)
        
        # Lấy các liên kết và thu thập dữ liệu từ các trang con
        if depth < self.max_depth:
            links = self.extract_links(soup, url)
            child_links = [link for link in links if link not in self.visited_urls]
            
            # Thêm độ trễ để tránh tải quá nhiều request
            time.sleep(self.delay)
            
            return child_links
        return []
    
    def crawl_with_threading(self):
        """Thu thập dữ liệu song song với ThreadPoolExecutor"""
        links_to_crawl = self.urls.copy()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for depth in range(self.max_depth + 1):
                if not links_to_crawl:
                    break
                
                logger.info(f"--- Đang thu thập dữ liệu ở độ sâu {depth} ---")
                logger.info(f"Số lượng URL cần thu thập: {len(links_to_crawl)}")
                
                # Thu thập dữ liệu từ tất cả URL ở độ sâu hiện tại
                future_to_url = {
                    executor.submit(self.crawl_url, url, depth): url 
                    for url in links_to_crawl
                }
                
                next_links = []
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        child_links = future.result()
                        next_links.extend(child_links)
                    except Exception as e:
                        logger.error(f"Lỗi khi thu thập từ {url}: {e}")
                
                links_to_crawl = list(set(next_links))
    
    def crawl(self):
        """Thu thập dữ liệu từ tất cả URL"""
        logger.info(f"Bắt đầu thu thập dữ liệu từ {len(self.urls)} URL")
        
        # Sử dụng crawl_with_threading để xử lý song song
        self.crawl_with_threading()
        
        logger.info(f"Hoàn thành thu thập dữ liệu. Đã thu thập {len(self.visited_urls)} trang.")
        return self.data
    
    def save_to_csv(self):
        """Lưu dữ liệu đã thu thập thành file CSV"""
        for domain, items in self.data.items():
            safe_domain = re.sub(r'[^\w\-_]', '_', domain)
            filename = os.path.join(self.output_dir, f"{safe_domain}.csv")
            
            logger.info(f"Đang lưu dữ liệu cho domain {domain} vào {filename}")
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                # Xác định các trường sẽ được lưu
                fieldnames = ['url', 'title', 'meta_description', 'headers']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in items:
                    # Chuẩn bị dữ liệu để lưu vào CSV
                    row = {
                        'url': item['url'],
                        'title': item['title'],
                        'meta_description': item['meta_description'],
                        'headers': '|'.join(item['headers'])
                    }
                    writer.writerow(row)
            
            # Lưu nội dung văn bản vào file riêng
            text_dir = os.path.join(self.output_dir, f"{safe_domain}_texts")
            if not os.path.exists(text_dir):
                os.makedirs(text_dir)
                
            for idx, item in enumerate(items):
                page_id = re.sub(r'[^\w\-_]', '_', item['url'])[-50:]  # Lấy phần cuối URL làm ID
                text_file = os.path.join(text_dir, f"{idx}_{page_id}.txt")
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"URL: {item['url']}\n")
                    f.write(f"Title: {item['title']}\n\n")
                    f.write(item['text_content'])
    
    def save_to_json(self):
        """Lưu dữ liệu đã thu thập thành file JSON"""
        import json
        
        for domain, items in self.data.items():
            safe_domain = re.sub(r'[^\w\-_]', '_', domain)
            filename = os.path.join(self.output_dir, f"{safe_domain}.json")
            
            logger.info(f"Đang lưu dữ liệu cho domain {domain} vào {filename}")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    # Danh sách URL cần thu thập dữ liệu
    url_list = [
        "https://uet.vnu.edu.vn",
    "https://ussh.vnu.edu.vn",
    "https://ulis.vnu.edu.vn",
        
    ]
    
    # Tạo và cấu hình scraper
    scraper = HTMLScraper(
        urls=url_list,
        output_dir="../data/raw/VNU",  # Thư mục lưu kết quả
        max_depth=2,  # Thu thập 3 cấp độ các trang con
        delay=1,      # Đợi 1 giây giữa các request
        max_workers=5 # Số luồng xử lý đồng thời
    )
    
    # Thu thập dữ liệu
    scraper.crawl()
    
    # Lưu kết quả
    scraper.save_to_csv()
    scraper.save_to_json()
    
    logger.info("Quá trình thu thập dữ liệu đã hoàn tất!")


if __name__ == "__main__":
    main()
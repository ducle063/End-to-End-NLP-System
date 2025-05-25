import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
import re
import os
import time
import logging
import json
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automated_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedMenuScraper:
    def __init__(self, universities, output_dir="../data/raw/universities", use_selenium=True, max_workers=2):
        """
        Scraper tự động hóa thu thập dữ liệu từ menu navigation
        
        Args:
            universities (dict): Dictionary với key là tên viết tắt, value là URL
            output_dir (str): Thư mục lưu kết quả
            use_selenium (bool): Sử dụng Selenium cho các trang dynamic
            max_workers (int): Số lượng worker tối đa
        """
        self.universities = universities
        self.output_dir = output_dir
        self.use_selenium = use_selenium
        self.max_workers = max_workers
        self.visited_urls = set()
        self.collected_data = defaultdict(list)
        self.menu_structure = defaultdict(dict)
        
        # Tạo thư mục đầu ra
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Cấu hình Selenium
        if use_selenium:
            self.setup_selenium()
    
    def setup_selenium(self):
        """Thiết lập Selenium WebDriver"""
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    def get_selenium_driver(self):
        """Tạo instance của Selenium WebDriver"""
        return webdriver.Chrome(options=self.chrome_options)
    
    def extract_menu_navigation(self, url, university_code):
        """
        Trích xuất cấu trúc menu navigation từ trang web
        """
        logger.info(f"Đang phân tích menu navigation của {university_code}: {url}")
        
        menu_links = []
        
        try:
            if self.use_selenium:
                menu_links = self.extract_menu_with_selenium(url)
            else:
                menu_links = self.extract_menu_with_requests(url)
            
            # Lưu cấu trúc menu
            self.menu_structure[university_code] = {
                'base_url': url,
                'menu_links': menu_links,
                'total_links': len(menu_links)
            }
            
            logger.info(f"Tìm thấy {len(menu_links)} liên kết menu cho {university_code}")
            return menu_links
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất menu từ {url}: {e}")
            return []
    
    def extract_menu_with_selenium(self, url):
        """Sử dụng Selenium để trích xuất menu (cho các trang dynamic)"""
        driver = self.get_selenium_driver()
        menu_links = []
        
        try:
            driver.get(url)
            time.sleep(3)  # Đợi trang load
            
            # Tìm các selector menu phổ biến
            menu_selectors = [
                'nav a',
                '.navbar a',
                '.menu a',
                '.navigation a',
                '.main-menu a',
                '.header-menu a',
                '.top-menu a',
                'ul.menu a',
                '#menu a',
                '.nav-item a'
            ]
            
            for selector in menu_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        for element in elements:
                            href = element.get_attribute('href')
                            text = element.text.strip()
                            if href and text and self.is_valid_menu_link(href, url):
                                menu_links.append({
                                    'url': href,
                                    'text': text,
                                    'selector': selector
                                })
                        break  # Nếu tìm thấy menu, dừng tìm kiếm
                except Exception as e:
                    continue
            
            # Xử lý dropdown menu
            try:
                dropdown_triggers = driver.find_elements(By.CSS_SELECTOR, '.dropdown, .has-submenu, .menu-item-has-children')
                for trigger in dropdown_triggers:
                    try:
                        # Hover hoặc click để mở dropdown
                        webdriver.ActionChains(driver).move_to_element(trigger).perform()
                        time.sleep(1)
                        
                        submenu_links = trigger.find_elements(By.CSS_SELECTOR, 'a')
                        for link in submenu_links:
                            href = link.get_attribute('href')
                            text = link.text.strip()
                            if href and text and self.is_valid_menu_link(href, url):
                                menu_links.append({
                                    'url': href,
                                    'text': text,
                                    'selector': 'dropdown'
                                })
                    except Exception as e:
                        continue
            except Exception as e:
                pass
            
        finally:
            driver.quit()
        
        return self.deduplicate_menu_links(menu_links)
    
    def extract_menu_with_requests(self, url):
        """Sử dụng requests để trích xuất menu (cho các trang static)"""
        menu_links = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
            
            response = requests.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tìm các selector menu
            menu_selectors = [
                'nav a',
                '.navbar a',
                '.menu a',
                '.navigation a',
                '.main-menu a',
                'ul.menu a'
            ]
            
            for selector in menu_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        href = element.get('href')
                        text = element.get_text(strip=True)
                        if href and text and self.is_valid_menu_link(href, url):
                            full_url = urljoin(url, href)
                            menu_links.append({
                                'url': full_url,
                                'text': text,
                                'selector': selector
                            })
                    break
                    
        except Exception as e:
            logger.error(f"Lỗi khi extract menu với requests: {e}")
        
        return self.deduplicate_menu_links(menu_links)
    
    def is_valid_menu_link(self, href, base_url):
        """Kiểm tra xem link có hợp lệ để crawl không"""
        if not href:
            return False
        
        # Bỏ qua các link không cần thiết
        invalid_patterns = [
            r'#', r'javascript:', r'mailto:', r'tel:', r'ftp:',
            r'\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|jpg|jpeg|png|gif|mp4|mp3)$',
            r'facebook\.com', r'twitter\.com', r'youtube\.com', r'linkedin\.com'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, href, re.IGNORECASE):
                return False
        
        # Chỉ lấy link cùng domain
        if href.startswith('http'):
            return urlparse(href).netloc == urlparse(base_url).netloc
        
        return True
    
    def deduplicate_menu_links(self, menu_links):
        """Loại bỏ các link trùng lặp"""
        seen_urls = set()
        unique_links = []
        
        for link in menu_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)
        
        return unique_links
    
    def categorize_menu_links(self, menu_links):
        """Phân loại các link menu theo danh mục"""
        categories = {
            'gioi_thieu': ['giới thiệu', 'about', 'overview', 'tổng quan'],
            'dao_tao': ['đào tạo', 'education', 'academic', 'chương trình', 'program'],
            'sinh_vien': ['sinh viên', 'student', 'học sinh', 'ctsv'],
            'khoa_hoc': ['khoa học', 'research', 'nghiên cứu', 'công nghệ', 'technology'],
            'tuyen_sinh': ['tuyển sinh', 'admission', 'tuyển dụng', 'recruitment'],
            'tin_tuc': ['tin tức', 'news', 'thông báo', 'announcement', 'sự kiện', 'event'],
            'lien_he': ['liên hệ', 'contact', 'địa chỉ', 'address'] 
        }
        
        categorized = defaultdict(list)
        uncategorized = []
        
        for link in menu_links:
            text_lower = link['text'].lower()
            url_lower = link['url'].lower()
            
            categorized_flag = False
            for category, keywords in categories.items():
                if any(keyword in text_lower or keyword in url_lower for keyword in keywords):
                    categorized[category].append(link)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                uncategorized.append(link)
        
        return dict(categorized), uncategorized
    
    def crawl_menu_link(self, link_info, university_code):
        """Crawl dữ liệu từ một link menu"""
        url = link_info['url']
        
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        logger.info(f"Crawling {university_code}: {link_info['text']} - {url}")
        
        try:
            # Sử dụng requests để crawl nội dung
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
            
            response = requests.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Trích xuất nội dung
            content_data = self.extract_page_content(soup, url, link_info['text'])
            if content_data:
                content_data['menu_category'] = link_info.get('category', 'uncategorized')
                content_data['menu_text'] = link_info['text']
                self.collected_data[university_code].append(content_data)
            
            time.sleep(1)  # Delay
            
        except Exception as e:
            logger.error(f"Lỗi khi crawl {url}: {e}")
    
    def extract_page_content(self, soup, url, page_title):
        """Trích xuất nội dung từ trang web"""
        # Loại bỏ các thẻ không cần thiết
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Lấy tiêu đề
        title = soup.title.text.strip() if soup.title else page_title
        
        # Lấy nội dung chính
        content_selectors = [
            'main', 'article', '.content', '#content', '.main-content',
            '.post-content', '.entry-content', '.page-content', '.container'
        ]
        
        main_content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                break
        
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)
        
        # Làm sạch nội dung
        main_content = re.sub(r'\s+', ' ', main_content.strip())
        
        if len(main_content) < 100:
            return None
        
        return {
            'url': url,
            'title': title,
            'content': main_content
        }
    
    def crawl_university_by_menu(self, university_code, base_url):
        """Crawl dữ liệu từ một trường dựa trên menu navigation"""
        logger.info(f"Bắt đầu crawl {university_code} theo menu navigation")
        
        # Trích xuất menu navigation
        menu_links = self.extract_menu_navigation(base_url, university_code)
        
        if not menu_links:
            logger.warning(f"Không tìm thấy menu navigation cho {university_code}")
            return
        
        # Phân loại menu links
        categorized_links, uncategorized = self.categorize_menu_links(menu_links)
        
        # Thêm category vào link info
        all_links = []
        for category, links in categorized_links.items():
            for link in links:
                link['category'] = category
                all_links.append(link)
        
        for link in uncategorized:
            link['category'] = 'uncategorized'
            all_links.append(link)
        
        # Crawl từng link
        for link_info in all_links:
            self.crawl_menu_link(link_info, university_code)
        
        logger.info(f"Hoàn thành crawl {university_code} - Thu thập {len(self.collected_data[university_code])} trang")
    
    def crawl_all_universities(self):
        """Crawl tất cả các trường đại học"""
        logger.info(f"Bắt đầu crawl {len(self.universities)} trường đại học")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for university_code, url in self.universities.items():
                future = executor.submit(self.crawl_university_by_menu, university_code, url)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Lỗi khi crawl university: {e}")
    
    def save_results(self):
        """Lưu kết quả crawl"""
        logger.info("Đang lưu kết quả...")
        
        for university_code, pages in self.collected_data.items():
            if not pages:
                continue
            
            # Tạo thư mục cho trường
            university_dir = os.path.join(self.output_dir, university_code)
            if not os.path.exists(university_dir):
                os.makedirs(university_dir)
            
            # Lưu từng trang theo category
            for idx, page_data in enumerate(pages):
                category = page_data.get('menu_category', 'uncategorized')
                
                # Tạo thư mục category
                category_dir = os.path.join(university_dir, category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                
                # Tạo tên file
                safe_title = re.sub(r'[^\w\-_]', '_', page_data.get('menu_text', 'page'))[:30]
                filename = f"{university_code}_{category}_{idx:04d}_{safe_title}.txt"
                filepath = os.path.join(category_dir, filename)
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"UNIVERSITY: {university_code.upper()}\n")
                        f.write(f"CATEGORY: {category.upper()}\n")
                        f.write(f"MENU_TEXT: {page_data.get('menu_text', '')}\n")
                        f.write(f"URL: {page_data['url']}\n")
                        f.write(f"TITLE: {page_data['title']}\n")
                        f.write("-" * 80 + "\n")
                        f.write(page_data['content'])
                except Exception as e:
                    logger.error(f"Lỗi khi lưu {filepath}: {e}")
        
        # Lưu cấu trúc menu
        menu_file = os.path.join(self.output_dir, "menu_structure.json")
        try:
            with open(menu_file, 'w', encoding='utf-8') as f:
                json.dump(self.menu_structure, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu menu structure: {e}")
        
        # Thống kê
        total_pages = sum(len(pages) for pages in self.collected_data.values())
        logger.info(f"Đã lưu {total_pages} trang từ {len(self.collected_data)} trường")


def main():
    """Hàm chính"""
    # Cấu hình các trường cần crawl
    universities = {
        "uet": "https://uet.vnu.edu.vn",
        "vnu": "https://vnu.edu.vn",
        "uslis": "https://uslis.vnu.edu.vn",
        "ussh": "https://ussh.vnu.edu.vn",
        "hnue": "https://hnue.edu.vn",
        "ueb": "https://ueb.edu.vn",
        "is": "https://is.vnu.edu.vn",
    }
    
    # Tạo scraper
    scraper = AutomatedMenuScraper(
        universities=universities,
        output_dir="../data/raw/universities_menu",
        use_selenium=True,  # Sử dụng Selenium cho trang dynamic
        max_workers=1  # Giảm xuống 1 để tránh quá tải
    )
    
    # Chạy crawl
    scraper.crawl_all_universities()
    
    # Lưu kết quả
    scraper.save_results()
    
    logger.info("Hoàn thành crawl tự động!")


if __name__ == "__main__":
    main()
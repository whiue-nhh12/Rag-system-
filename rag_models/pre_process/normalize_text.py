import re
from typing import List


class BasePreProcess:
    """Lớp cơ sở cho các bước tiền xử lý văn bản."""

    def __init__(self):
        self._is_garbage_zone = False
        self._is_middle_garbage_zone = False

    @staticmethod
    def replace_slash_n_with_space(text: str) -> str:
        """Nối các dòng thường thành một đoạn và giữ nguyên ngắt dòng cho heading/bullet."""
        lines = text.splitlines()
        if not lines:
            return ""

        structure_pattern = r'^\s*(?:CHƯƠNG|Chương|CHAPTER|Chapter|PHẦN|Phần|\d+(?:\.\d+)*\b|[-*+•])'
        current_buffer: List[str] = []
        end_sentence_pattern = r'.*[.?!]\s*$'
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if re.match(structure_pattern, line):
                current_buffer.append("\n" + line)
            else:
                if current_buffer:
                    prev_line = current_buffer[-1]

                # Nếu dòng trước kết thúc bằng dấu câu (. ? !) 
                # Hoặc dòng trước là Heading/Bullet -> TÁCH DÒNG MỚI
                    if re.search(end_sentence_pattern, prev_line):
                        current_buffer.append("\n" + line)
                    else:
                        # Dòng trước chưa hết câu -> NỐI DÒNG kèm khoảng trắng
                        current_buffer[-1] += " " + line
                else:
                    current_buffer.append(line)

        return "".join(current_buffer).strip()

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """Xóa số trang xuất hiện ở đầu mỗi trang hoặc dưới dạng số đơn độc."""
        # Xóa các dòng chỉ chứa một số trang
        text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
        # Xóa số trang tách biệt bởi nhiều khoảng trắng trong văn bản nối liền
        text = re.sub(r'(?<=\s{2})\d{1,3}(?=\s{2})', '', text)
        return text

    @staticmethod
    def normalize_bullets(text: str) -> str:
        """Thống nhất các dấu đầu dòng '-', '+', '*' thành '-'."""
        # Chuyển các ký tự đầu dòng bullet khác nhau về một dấu '-' duy nhất
        text = re.sub(r'(?m)^\s*[+*\u2022\u2023\u2013\u2014-]+\s*', '- ', text)
        return text

    @staticmethod
    def _is_toc_header(line: str) -> bool:
        trimmed = line.strip()
    
    # 1. Nếu dòng dài hơn 50 ký tự thì CHẮC CHẮN là văn bản thường, không phải Header
        if len(trimmed) > 50:
            return False

    # 2. Bắt buộc từ khóa Header phải chiếm trọn vẹn cả dòng (dùng ^ và $)
        pattern = r'(?i)^\s*(?:MỤC\s*LỤC|TABLE\s+OF\s+CONTENTS|NỘI\s+DUNG|MỤC\s*LỤC:)\s*$'
        return bool(re.match(pattern, trimmed))

    @staticmethod
    def _is_chapter_header(line: str) -> bool:
        trimmed = line.strip()
    # Phủ định nếu cuối dòng có chứa dải dấu chấm/khoảng trắng kéo dài đến số trang
        chapter = r'(?i)^\s*(?:CHƯƠNG|Chapter)(?:\s*\d+)?\b(?!.*[\.\s…]{2,}\d+$).*$'
        return bool(re.match(chapter, trimmed))

    @staticmethod
    def _is_toc_entry(line: str) -> bool:
        trimmed = line.strip()
        if re.search(r'\.{5,}\s*\d{1,4}\s*$', trimmed):
            return True
        if re.search(r'^(?:\s*(?:CHƯƠNG|CHƯƠNG\s*\d+|Chương\s*\d+|Chapter\s*\d+))\b.*\.{5,}\s*\d{1,4}\s*$', trimmed):
            return True
        return bool(re.match(r'^\s*\d{1,2}(?:\.\d+)*\s+.*\.{5,}\s*\d{1,4}\s*$', trimmed))

    def remove_content_table(self, text: str,) -> str:
        """Xóa mục lục khi xử lý theo từng trang."""
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            """
            print(f"Processing line: {line}")
            print(f"line is header {self._is_chapter_header(line)}") 
            print(f"Line is TOC entry: {self._is_toc_entry(line)}")
            print(f"Line is TOC header: {self._is_toc_header(line)}")
            """  # Debug: In ra dòng hiện tại đang xử lý
            if self._is_garbage_zone:
                if self._is_chapter_header(line):
                    self._is_garbage_zone = False
                    cleaned_lines.append(line)
                else:
                    continue

            elif self._is_toc_header(line) or self._is_toc_entry(line):
                self._is_garbage_zone = True
                continue

            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def remove_middle_sections(self, text: str) -> str:
        """Xóa các khối rác giữa các chương khi gặp phần CÂU HỎI ÔN TẬP."""
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if self._is_middle_garbage_zone:
                if self._is_chapter_header(line):
                    self._is_middle_garbage_zone = False
                    if self._is_chapter_header(line):
                        cleaned_lines.append(line)
                    continue
                continue

            if re.search(r'(?i)\bCÂU\s*HỎI\s*ÔN\s*TẬP\b', stripped):
                self._is_middle_garbage_zone = True
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Xóa khoảng trắng thừa và chuẩn hóa về một khoảng trắng giữa các từ."""
        # Thay nhiều khoảng trắng hoặc tab thành một khoảng trắng
        text = re.sub(r'[^\S\r\n]+', ' ', text)
        # Loại bỏ khoảng trắng đầu/cuối
        return text.strip()

    def clean(self, text: str) -> str:
        """Thực hiện toàn bộ bước tiền xử lý chuẩn hóa văn bản."""
        text = self.remove_content_table(text)
        text = self.remove_middle_sections(text)
        text = self.remove_page_numbers(text)
        text = self.replace_slash_n_with_space(text)
        text = self.normalize_bullets(text)
        text = self.normalize_whitespace(text)
        return text

    @staticmethod
    def clean_text(text: str,) -> str:
        return BasePreProcess().clean(text)


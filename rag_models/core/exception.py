class URLError(Exception):
    """Lỗi khi tải nội dung từ URL."""


class FileNotFound(Exception):
    """Lỗi khi file không tồn tại trên hệ thống."""


class LoadImageError(Exception):
    """Lỗi khi tải file nhưng file không phải ảnh hợp lệ."""


class OCRProcessingError(Exception):
    """Lỗi khi xử lý OCR cho ảnh."""


class ContentTooLargeError(Exception):
    """Lỗi khi nội dung tải về quá lớn để xử lý."""

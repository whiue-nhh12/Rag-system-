from rag_models.pre_process.normalize_text import BasePreProcess


def test_clean_text_removes_toc_on_first_page_without_needing_full_text():
    sample_text = """MỤC LỤC
1. Giới thiệu
2. Nội dung chính

CHƯƠNG 1
Đây là nội dung chương đầu tiên."""

    cleaned = BasePreProcess.clean_text(sample_text, page_number=1)

    assert "MỤC LỤC" not in cleaned
    assert "CHƯƠNG 1" in cleaned
    assert "Đây là nội dung chương đầu tiên" in cleaned


def test_clean_text_keeps_removing_toc_entries_with_chapter_headings():
    sample_text = """MỤC LỤC
CHƯƠNG 1 : NHỮNG VẤN ĐỀ CƠ BẢN VỀ NHÀ NƯỚC VÀ PHÁP LUẬT .................... 3
CHƯƠNG 2 : QUY PHẠM PHÁP LUẬT, VĂN BẢN QUY PHẠM PHÁP LUẬT, QUAN .................... 47

CHƯƠNG 1
Đây là nội dung chương đầu tiên."""

    cleaned = BasePreProcess.clean_text(sample_text, page_number=1)

    assert "MỤC LỤC" not in cleaned
    assert "CHƯƠNG 1 : NHỮNG VẤN ĐỀ CƠ BẢN" not in cleaned
    assert "CHƯƠNG 2 : QUY PHẠM PHÁP LUẬT" not in cleaned
    assert "CHƯƠNG 1" in cleaned
    assert "Đây là nội dung chương đầu tiên" in cleaned

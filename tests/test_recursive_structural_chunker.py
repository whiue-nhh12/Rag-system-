from langchain_core.documents import Document

from rag_models.chunking import RecursiveStructuralChunker


def test_recursive_structural_chunker_preserves_hierarchy_and_splits_by_heading():
    documents = [
        Document(
            page_content=(
                "CHƯƠNG 1 Giới thiệu\n\n"
                "1.1 Tổng quan\n\n"
                "Đây là một đoạn văn bản dài được thiết kế để vượt quá giới hạn chunk size. "
                "Đoạn văn này lặp lại nhiều từ để chắc chắn việc chia nhỏ sẽ diễn ra. "
                "Đây là nội dung đầu tiên của chương để kiểm tra thuật toán chia theo đề mục."
            ),
            metadata={"source": "sample.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "1.2 Cài đặt\n\n"
                "- Bước 1: Mở ứng dụng.\n"
                "- Bước 2: Chọn cấu hình.\n"
                "- Bước 3: Nhấn nút lưu.\n\n"
                "Đây là một đoạn văn bản khác cũng đủ dài để kích hoạt quá trình phân chia."
                "Nó tiếp tục mở rộng để kiểm tra rằng chunk metadata được giữ nguyên khi "
                "tách thành nhiều phần nhỏ hơn."
            ),
            metadata={"source": "sample.pdf", "page": 2},
        ),
    ]

    chunker = RecursiveStructuralChunker(max_chunk_size=160, chunk_overlap=20)
    chunks = chunker.chunk(documents)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "sample.pdf"
        assert chunk.metadata["chapter"] == "CHƯƠNG 1 Giới thiệu"
        assert chunk.metadata["hierarchy_path"].startswith("CHƯƠNG 1 Giới thiệu")
        assert chunk.metadata["chunk_index"] is not None
        assert chunk.metadata["start_page"] in {1, 2}
        assert chunk.metadata["end_page"] in {1, 2}


def test_recursive_structural_chunker_splits_by_smaller_subsection_heading():
    documents = [
        Document(
            page_content=(
                "CHƯƠNG 1 Giới thiệu\n\n"
                "1.1 Mở đầu\n\n"
                "Đoạn văn bản đầu tiên rất dài để buộc chunker phải tách theo heading nhỏ. "
                "Nội dung này lặp lại nhiều lần để vượt quá kích thước chunk mặc định và tạo điều kiện "
                "cho việc kiểm tra việc phân tách theo smaller subsection.\n\n"
                "1.1.1.1 Mục nhỏ thứ nhất\n\n"
                "Nội dung mục nhỏ thứ nhất. Đây là phần cần được tách thành một chunk riêng vì nó là heading cấp 4.\n\n"
                "1.1.1.2 Mục nhỏ thứ hai\n\n"
                "Nội dung mục nhỏ thứ hai. Đây là phần tiếp theo sau khi tách bằng smaller_subsection."
            ),
            metadata={"source": "sample.pdf", "page": 1},
        )
    ]

    chunker = RecursiveStructuralChunker(max_chunk_size=220, chunk_overlap=20)
    chunks = chunker.chunk(documents)

    assert len(chunks) >= 2
    heading_texts = [chunk.page_content for chunk in chunks]
    assert any("1.1.1.1 Mục nhỏ thứ nhất" in text for text in heading_texts)
    assert any("1.1.1.2 Mục nhỏ thứ hai" in text for text in heading_texts)

    smaller_chunk = next(
        chunk for chunk in chunks if "1.1.1.1 Mục nhỏ thứ nhất" in chunk.page_content
    )
    assert smaller_chunk.metadata["section"] == "1.1 Mở đầu"
    assert smaller_chunk.metadata["subsection"] == "1.1.1.1 Mục nhỏ thứ nhất"

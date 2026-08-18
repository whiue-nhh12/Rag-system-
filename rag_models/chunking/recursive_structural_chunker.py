import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple,Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_models.chunking.base import BaseChunking

title_dictionary = {
    "pháp luật" :"Pháp luật",
    "nhà nước" :"Nhà nước",
    "vi phạm pháp luật" :"Vi phạm",
    "trách nhiệm pháp lí" :"Trách nhiệm",
    "hiến pháp" :"Hiến pháp",
    "hành chính" :"Hành chính",
    "dân sự" :"Dân sự",
    "hình sự" :"Hình sự",
    "lao động" :"Lao động",
    "kinh doanh" :"Kinh doanh",
}

@dataclass
class StructuralBlock:
    text: str
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    smaller_subsection : str = ""
    title : Optional[List[str]] = None
    source: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    source_doc_index: Optional[int] = None
    hierarchy_path: str = field(init=False, default="")

    @classmethod
    def from_document(
        cls,
        document: Document,
        source_doc_index: Optional[int] = None,
    ) -> "StructuralBlock":
        metadata = document.metadata or {}
        return cls(
            text=(document.page_content or "").strip(),
            chapter=str(metadata.get("chapter", "")) if metadata.get("chapter") is not None else "",
            section=str(metadata.get("section", "")) if metadata.get("section") is not None else "",
            subsection=str(metadata.get("subsection", "")) if metadata.get("subsection") is not None else "",
            source=metadata.get("source"),
            start_page=metadata.get("start_page") or metadata.get("page") or metadata.get("page_number") or metadata.get("page_num"),
            end_page=metadata.get("end_page") or metadata.get("page") or metadata.get("page_number") or metadata.get("page_num"),
            source_doc_index=source_doc_index if source_doc_index is not None else metadata.get("source_doc_index"),
        )

    def to_document(self, chunk_index: Optional[int] = None) -> Document:
        """Chuyển đổi StructuralBlock thành Document chuẩn của LangChain."""
        metadata: Dict[str, Any] = {
            "source": self.source,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "title": self.title,
            "hierarchy_path": self.hierarchy_path,
            "source_doc_index": self.source_doc_index,
        }

        if chunk_index is not None:
            metadata["chunk_index"] = chunk_index

        doc_id = (
            f"chunk_{chunk_index}_p{self.start_page or 0}"
            if chunk_index is not None
            else None
        )

        return Document(
            page_content=self.text,
            metadata=metadata,
            id=doc_id,
        )

    def __post_init__(self) -> None:
        parts = [p for p in (self.chapter, self.section, self.subsection) if p]
        self.hierarchy_path = " > ".join(parts) if parts else "root"


logger = logging.getLogger(__name__)


class RecursiveStructuralChunker(BaseChunking):
    """Chunk structured documents by headings in a recursive, hierarchy-aware way."""

    def __init__(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        super().__init__(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chapter_pattern = r"^CHƯƠNG\s+\d+.*"
        # Separators for recursive splitting - ORDERED BY PRIORITY (highest first)
        # Headings have highest priority to maintain document structure
        self.split_separators = [       # Highest priority
            ("heading_section", r"^\d+\.\d+\s+.*"),         # Mid-high priority
            ("heading_subsection", r"^\d+\.\d+\.\d+\.?\s+.*"), # Mid priority
            ("heading_smaller_subsection", r"^\d+\.\d+\.\d+\.\d+\.?\s+.*")
                               # Lower prior                                  # Last resort
        ]

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", " ", ""],
        )
    
    def _calculate_text_length(self, text: str) -> int:
        """Calculate length of text for chunk size checks."""
        return len(text)

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split a list of page-like Documents into hierarchy-aware chunks."""
        logger.info(
            "Chunking %s document(s) with RecursiveStructuralChunker",
            len(documents),
        )
        all_chunks = self.build_flow(documents)

        logger.info(
            "Completed chunking: %s document(s) -> %s chunk(s)",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    def build_flow(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Assemble blocks and split them into chunks, exposing the processing flow."""
        logger.info(
            "Building structural chunking flow for %s document(s)",
            len(documents),
        )
        structural_blocks = self._assemble_structural_blocks(documents)

        all_chunks: List[Document] = []
        global_chunk_counter = 0

        for block in structural_blocks:
            block_chunks = self._split_structural_block(block, global_chunk_counter)
            all_chunks.extend(block_chunks)
            global_chunk_counter += len(block_chunks)

        return all_chunks

    def _assemble_structural_blocks(self, documents: List[Document]) -> List[StructuralBlock]:
        blocks: List[StructuralBlock] = []
        current_buffer: List[str] = []
        current_metadata = {
            "title": [],
            "chapter": "",
            "section": "",
            "subsection": "",
        }
        current_source: Optional[str] = None
        current_start_page: Optional[int] = None
        current_end_page: Optional[int] = None
        current_start_doc_idx: Optional[int] = None

        for doc_idx, doc in enumerate(documents):
            content = doc.page_content or ""
            if not content.strip():
                continue

            source = doc.metadata.get("source") or current_source
            if source is not None:
                current_source = source

            page_num = self._get_page_number(doc, doc_idx)
            if current_start_page is None:
                current_start_page = page_num
            current_end_page = page_num
            if current_start_doc_idx is None:
                current_start_doc_idx = doc_idx

            for raw_line in content.splitlines():
                line = raw_line.strip()
                if not line:
                    if current_buffer:
                        current_buffer.append("")
                    continue
                # Only detect structural headings here (chapter, section, subsection).
                pattern =  self.chapter_pattern
                if re.match(pattern,line) :
                    if current_buffer:
                        blocks.append(
                            self._build_chapter_block(
                                current_buffer,
                                current_metadata,
                                current_source,
                                current_start_page,
                                current_end_page,
                                current_start_doc_idx,
                            )
                        )
                        current_buffer = []

                    chapter_title = line
                    lower_line = chapter_title.lower()
                    topics = []
                    for key, value in title_dictionary.items():
                        if key in lower_line:
                            topics.append(value)

                    current_metadata = {
                        "chapter": chapter_title,
                        "section": "",
                        "subsection": "",
                        "smaller_subsection":"",
                        "title": topics,
                    }
                    current_buffer = [chapter_title]
                    current_start_page = page_num
                    current_end_page = page_num
                    current_start_doc_idx = doc_idx
                    continue

                if not current_buffer:
                    current_buffer = []
                current_buffer.append(raw_line.rstrip())
                current_end_page = page_num

        if current_buffer:
            blocks.append(
                self._build_chapter_block(
                    current_buffer,
                    current_metadata,
                    current_source,
                    current_start_page,
                    current_end_page,
                    current_start_doc_idx,
                )
            )

        return blocks

    def _build_chapter_block(
        self,
        buffer_lines: List[str],
        metadata: Dict[str, str],
        source: Optional[str],
        start_page: Optional[int],
        end_page: Optional[int],
        start_doc_idx: Optional[int],
    ) -> StructuralBlock:
        text = "\n".join(line for line in buffer_lines if line is not None).strip()
        if not text:
            text = ""
        return StructuralBlock(
            text=text,
            chapter=metadata.get("chapter", ""),
            section=metadata.get("section", ""),
            subsection=metadata.get("subsection", ""),
            title=metadata.get("title", []),
            source=source,
            start_page=start_page,
            end_page=end_page,
            source_doc_index=start_doc_idx,
        )

    def _split_structural_block(
        self, block: StructuralBlock, global_chunk_counter: int
    ) -> List[Document]:
        """Split a structural block using recursive separator-based approach with heading priority."""
        block_text = str(block.text)
        if self._calculate_text_length(block_text) <= self.max_chunk_size:
            return [
                self._create_document_chunk(
                    block,
                    global_chunk_counter,
                )
            ]

        all_chunks = self._recursive_split_text(block, self.split_separators)
        if not isinstance(all_chunks, list):
            all_chunks = [all_chunks]

        return [
            self._create_document_chunk(
                chunk,
                global_chunk_counter + offset,
            )
            for offset, chunk in enumerate(all_chunks)
        ]

    def _recursive_split_text(
        self, block: StructuralBlock, separators: list
    ) -> list:
        """
        Recursively split text using separators with heading priority.
        Format: [(separator_type_name, separator_pattern), ...]
        Tries separators in order until finding one that exists in text.
        """
        final_chunks: list[StructuralBlock] = []

        if not separators or not block.text:
            return [block]

        separator_type = None
        separator_pattern = None
        next_separators = []

        for i, (sep_type, sep_pattern) in enumerate(separators):
            if sep_pattern is None:
                separator_type = sep_type
                separator_pattern = sep_pattern
                next_separators = []
                break

            if self._separator_exists_in_text(block.text, sep_pattern):
                separator_type = sep_type
                separator_pattern = sep_pattern
                next_separators = separators[i + 1:]
                break

        if separator_pattern is None:
            return [block]

        if separator_type.startswith("heading_") or separator_type == "bullet":
            splits = self._split_by_pattern(block, separator_pattern)
        else:
            split_docs = self.fallback_splitter.split_documents([block.to_document()])
            splits = [StructuralBlock.from_document(doc) for doc in split_docs]

        temp: list[StructuralBlock] = []

        for s in splits:
            if self._calculate_text_length(s.text) <= self.max_chunk_size:
                temp.append(s)
            else:
                if temp:
                    final_chunks.extend(temp)
                    temp = []

                if not next_separators:
                    final_chunks.append(s)
                else:
                    recursive_chunks = self._recursive_split_text(s, next_separators)
                    if isinstance(recursive_chunks, StructuralBlock):
                        final_chunks.append(recursive_chunks)
                    elif isinstance(recursive_chunks, list):
                        final_chunks.extend(recursive_chunks)

        if temp:
            final_chunks.extend(temp)

        return final_chunks

    def _separator_exists_in_text(self, text: str, separator: str) -> bool:
        """Check if a separator (string or regex pattern) exists in text."""
        if not separator:
            return False
        
        # For regex patterns (headings, bullets)
        if separator.startswith(r"^"):
            return any(re.match(separator, line) for line in text.splitlines())
        
        # For simple string separators
        return separator in text

    def _build_block(
    self,
    texts: List[str],
    parent_block: StructuralBlock,
    header_content: str,
    hierarchy: Dict[str, str],
) -> StructuralBlock:
        """
        Khởi tạo StructuralBlock con, tự động kế thừa source, pages, 
        và làm giàu metadata phân cấp từ khối cha.
        """
        full_text = "\n".join(texts).strip()

        topics_list = []
        matched_titles = list(parent_block.title or [])
        normalized_header = header_content.lower()
        for key, value in title_dictionary.items():
            if key in normalized_header:
                topics_list.append(value)
        if topics_list:
            matched_titles = list(set(matched_titles) & set(topics_list))

        section_value = hierarchy.get("section", parent_block.section)
        subsection_value = hierarchy.get("subsection", parent_block.subsection)
        smaller_subsection_value = hierarchy.get("smaller_subsection", parent_block.subsection)

        return StructuralBlock(
            text=full_text,
            chapter=hierarchy.get("chapter", parent_block.chapter),
            section=section_value,
            subsection=subsection_value,
            smaller_subsection= smaller_subsection_value,
            title=matched_titles,
            source=parent_block.source,
            start_page=parent_block.start_page,
            end_page=parent_block.end_page,
            source_doc_index=parent_block.source_doc_index,
        )


    def _split_by_pattern(
    self, block: StructuralBlock, pattern: str
) -> List[StructuralBlock]:
        """
        Phân tách một StructuralBlock cha thành danh sách các StructuralBlock con
        dựa trên regex pattern, kế thừa và cập nhật ngữ cảnh phân cấp (hierarchy memory).
        """
        all_blocks: List[StructuralBlock] = []
        current_part: List[str] = []

        # 1. Kế thừa trạng thái phân cấp ban đầu từ khối cha
        hierarchy_state = {
            "chapter": block.chapter,
            "section": block.section,
            "subsection": block.subsection,
            "smaller_subsection": block.smaller_subsection,
        }
        current_header_content = ""

        for line in (block.text or "").splitlines(keepends=False):
            stripped_line = line.strip()
            if not stripped_line:
                if current_part:
                    current_part.append("")
                continue

            # 2. Kiểm tra dòng có khớp với pattern phân cấp đang xét hay không
            if re.match(pattern, stripped_line):
                # Đóng gói khối CŨ trước khi chuyển sang header mới
                if current_part:
                    content = "\n".join(current_part).strip()
                    if content:
                        child_block = self._build_block(
                            texts=current_part,
                            parent_block=block,
                            header_content=current_header_content,
                            hierarchy=dict(hierarchy_state),
                        )
                        all_blocks.append(child_block)

                    current_part = []

                # Cập nhật phân cấp ngữ cảnh cho khối MỚI
                current_header_content = stripped_line
                header_level = self._detect_heading_level(stripped_line)

                if header_level == "chapter":
                    hierarchy_state["chapter"] = stripped_line
                    hierarchy_state["section"] = ""
                    hierarchy_state["subsection"] = ""
                    hierarchy_state["smaller_subsection"] = ""
                elif header_level == "section":
                    hierarchy_state["section"] = stripped_line
                    hierarchy_state["subsection"] = ""
                    hierarchy_state["smaller_subsection"] = ""
                elif header_level == "subsection":
                    hierarchy_state["subsection"] = stripped_line
                    hierarchy_state["smaller_subsection"] = ""
                elif header_level == "smaller_subsection":
                    hierarchy_state["subsection"] = stripped_line
                    hierarchy_state["smaller_subsection"] = stripped_line

                current_part.append(line.rstrip())
            else:
                current_part.append(line.rstrip())

        # 3. Đóng gói khối cuối cùng còn lại trong buffer
        if current_part:
            content = "\n".join(current_part).strip()
            if content:
                child_block = self._build_block(
                    texts=current_part,
                    parent_block=block,
                    header_content=current_header_content,
                    hierarchy=dict(hierarchy_state),
                )
                all_blocks.append(child_block)

        return all_blocks


    def _detect_heading_level(self, line: str) -> Optional[str]:
        candidate = line.strip()
        if re.match(self.chapter_pattern, candidate):
            return "chapter"
        return None

    def _create_document_chunk(
        self,
        parent_block: StructuralBlock,
        chunk_index: int,
    ) -> Document:
        text = parent_block.text
        metadata = {
            "source": parent_block.source,
            "start_page": parent_block.start_page,
            "end_page": parent_block.end_page,
            "chapter": parent_block.chapter or "",
            "section": parent_block.section or "",
            "subsection": parent_block.subsection or "",
            "hierarchy_path": parent_block.hierarchy_path,
            "chunk_index": chunk_index,
            "source_doc_index": parent_block.source_doc_index,
        }
        return Document(page_content=text, metadata=metadata)

    def _build_hierarchy_path(self, chapter: str, section: str, subsection: str) -> str:
        parts = [part for part in [chapter, section, subsection] if part]
        return " > ".join(parts) if parts else "root"

    def _get_page_number(self, doc: Document, fallback_index: int) -> int:
        for key in ("page", "page_number", "page_num"):
            value = doc.metadata.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return fallback_index + 1

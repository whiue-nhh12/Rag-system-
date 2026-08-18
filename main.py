import os
from dotenv import load_dotenv
import argparse
import sys
from pathlib import Path
from typing import Optional,List, Tuple
from rag_models.chunking.text_chunker import TextChunker
from rag_models.chunking.recursive_structural_chunker import RecursiveStructuralChunker
from rag_models.embedding.sentence_transformer import SentenceTransformerEmbedder
from rag_models.ingestion.pdf_loader import PDFLoader
from rag_models.llm.geminillm import GeminiLLM
from rag_models.vectordb.chormavectordb import ChromaVectorDB
from rag_models.pre_process.normalize_text import BasePreProcess
from rag_models.retrival.retriever import HybridRetriever,Retriever
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_core.documents import Document
import pickle
import time
from rag_models.retrival.reranker import RerankerSystem
from rag_models.prompt.phapluat_prompt import LawPrompt

load_dotenv()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class BM25Service:
    """Class quản lý việc Lưu / Tải / Search BM25 bền vững dưới ổ cứng."""
    def __init__(self, store_path: Path = Path("rag_models/data/bm25_index.pkl")):
        self.store_path = store_path
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[Document] = []
        self._load_if_exists()

    def build_and_save(self, chunks: List[Document]) -> None:
        """Tạo BM25 Index từ danh sách chunks và lưu xuống file pkl."""
        self.chunks = chunks
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Đảm bảo thư mục tồn tại trước khi save
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "wb") as f:
            pickle.dump({"index": self.bm25_index, "chunks": self.chunks}, f)
        print(f"-> Đã lưu BM25 Index ({len(chunks)} chunks) tại: {self.store_path}")

    def _load_if_exists(self) -> None:
        """Tự động load index từ đĩa nếu file pkl đã tồn tại."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_index = data["index"]
                    self.chunks = data["chunks"]
                print("-> Đã tải BM25 Index thành công từ đĩa.")
            except Exception as e:
                print(f"Warning: Không thể load BM25 store: {e}")

    def keyword_search(self, query: str, k: int = 5, filter: dict = None) -> List[Tuple[Document, float]]:
        """Hàm Adapter cho HybridRetriever."""
        if not self.bm25_index or not self.chunks:
            raise ValueError("BM25 Index chưa được khởi tạo! Hãy chạy ingestion (run_rag) trước.")

        tokenized_query = query.lower().split()
        raw_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Chuẩn hóa Min-Max về [0, 1]
        max_score = np.max(raw_scores) if np.max(raw_scores) > 0 else 1.0
        normalized_scores = raw_scores / max_score
        
        # Lấy Top K indices có điểm cao nhất
        top_indices = np.argsort(normalized_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if normalized_scores[idx] > 0:
                doc = self.chunks[idx]
                score = float(normalized_scores[idx])
                results.append((doc, score))
                
        return results

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
def run_rag(file_path: Optional[str] = None, query: Optional[str] = None) -> None:
    target = Path(file_path).expanduser() if file_path else None
    if target is None or not target.exists():
        raise FileNotFoundError(
            "Vui lòng truyền đường dẫn file hợp lệ. Ví dụ: python -m rag_models.main test.pdf"
        )

    print(f"Đang đọc file: {target}")
    pdf_loader = PDFLoader(str(target))
    documents = pdf_loader.load()
    base_preprocessor = BasePreProcess()
    # Cập nhật trực tiếp nội dung văn bản, giữ nguyên Metadata và cấu trúc Document
    for idx, doc in enumerate(documents):
        if idx == 244:
            print(f"\n--- Nội dung trang {idx + 1} trước khi tiền xử lý ---\n{doc.page_content}...\n")
        doc.page_content = base_preprocessor.clean(doc.page_content)


    print(f"Đã load {len(documents)} tài liệu")
    chunker = RecursiveStructuralChunker(max_chunk_size=1200, chunk_overlap=100)
    chunks = chunker.chunk(documents)
    bm25_service = BM25Service()
    bm25_service.build_and_save(chunks)
    print(f"Đã chia thành {len(chunks)} chunks")
    embedder = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embedded_docs = embedder.add_embeddings_to_documents(chunks, embedder.embed_documents(chunks))
    persist_dir = str(ROOT / "rag_models" / "data" / "chroma_db")
    vector_db = ChromaVectorDB(
        collection_name="rag_demo1",
        persist_directory=persist_dir,
        embedding=None,
    )
    vector_db.add_documents(chunks, embedder.embed_documents(chunks))

    print("Đã thêm vào vector DB")

    if not query:
        query = "Tóm tắt nội dung tài liệu"

    results = vector_db.similarity_search(query=query, k=3)
    print(f"\nCâu hỏi: {query}")
    print("Kết quả tìm kiếm:")
    for idx, result in enumerate(results, start=1):
        print(f"\n[{idx}] {result.page_content[:500]}...")
        print("Metadata:", result.metadata)


def ask(query: str,llm: Optional[object] = None, k: int = 10) -> str:
    """Nhận câu hỏi, tìm document liên quan và gửi cho LLM để trả lời."""
    if not query or not query.strip():
        raise ValueError("Query không được để trống")

    persist_dir = str(ROOT / "rag_models" / "data" / "chroma_db")
    vector_db = ChromaVectorDB(
        collection_name="rag_demo1",
        persist_directory=persist_dir,
        embedding=None,
    )
    store = vector_db.get_store()
    retriever = Retriever(vector_store=store)
    #results = retriver.retrieve_with_scores(query=query, k=k)
    lowerquery = query.lower()
    filterlist = []
    for key,value in title_dictionary.items():
        if key in lowerquery:
            filterlist.append(value)
    print(f"\nFilter list: {filterlist if filterlist else 'Không có filter'}")
    hybrid_retriever = HybridRetriever(semantic_search_fn=retriever.retrieve_with_scores, keyword_search_fn=BM25Service().keyword_search)
    if llm is None:
                llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"),model_name="gemini-flash-lite-latest")
                print(llm.model_name)
    queryhype = llm.hyde_generate(query=query)
    #print(f"\nQuery hyde: {queryhype}")
    results = hybrid_retriever.retrieve_with_scores(query=queryhype, k=20)
    content = [result[0].page_content for result in results]
    reranker = RerankerSystem.get_instance()
    results =  reranker.rerank(query,content,top_k=10)
    print(f"\nSố document tìm được: {len(results)}")
    if not results:
        raise ValueError("Không tìm thấy document phù hợp")
    context = "\n\n".join(
        f"[Document {idx}] {result[0]} and {result[1]:.4f}"
        for idx, result in enumerate(results, start=1)
    )
    prompt = LawPrompt().build_prompt(user_query=query,context=context)
    print(f"\nPrompt gửi cho LLM:\n{prompt}\n")
    #api_key = os.getenv("GEMINI_API_KEY")
    #print(f"Using Gemini API Key: {api_key}")

    start_time = time.time()
    llm_response = llm.ask(prompt)
    end_time = time.time()
    print(f"\nThời gian LLM trả lời: {end_time - start_time:.2f} giây")
    
    return llm_response


def ask_to_test(query: str,llm: Optional[object] = None, k: int = 10) -> str:
    """Nhận câu hỏi, tìm document liên quan và gửi cho LLM để trả lời."""
    if not query or not query.strip():
        raise ValueError("Query không được để trống")

    persist_dir = str(ROOT / "rag_models" / "data" / "chroma_db")
    vector_db = ChromaVectorDB(
        collection_name="rag_demo1",
        persist_directory=persist_dir,
        embedding=None,
    )
    store = vector_db.get_store()
    retriever = Retriever(vector_store=store)
    lowerquery = query.lower()
    filterlist = []
    for key,value in title_dictionary.items():
        if key in lowerquery:
            filterlist.append(value)
    print(f"\nFilter list: {filterlist if filterlist else 'Không có filter'}")
    hybrid_retriever = HybridRetriever(semantic_search_fn=retriever.retrieve_with_scores, keyword_search_fn=BM25Service().keyword_search)
    if llm is None:
                llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"),model_name="gemini-flash-lite-latest")
                print(llm.model_name)
    queryhype = llm.hyde_generate(query=query)
    #print(f"\nQuery hyde: {queryhype}")
    results = hybrid_retriever.retrieve_with_scores(query=queryhype, k=20)
    content = [result[0].page_content for result in results]
    reranker = RerankerSystem.get_instance()
    results =  reranker.rerank(query,content,top_k=10)
    if not results:
        raise ValueError("Không tìm thấy document phù hợp")
    context_to_test = [result[0] for result in results]
    context = "\n\n".join(
        f"[Document {idx}] {result[0]} and {result[1]:.4f}"
        for idx, result in enumerate(results, start=1)
    )
    prompt = LawPrompt().build_prompt(user_query=query,context=context)
    #api_key = os.getenv("GEMINI_API_KEY")
    #print(f"Using Gemini API Key: {api_key}")

    return {"output":llm.ask(prompt),"context_to_test":context_to_test}


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Chạy demo RAG với load -> chunk -> embedding -> vector search")
    parser.add_argument("file", nargs="?", help="Đường dẫn tới file PDF/TXT/MD")
    parser.add_argument("--query", help="Câu hỏi truy vấn")
    args = parser.parse_args()

    run_rag(file_path= ROOT/"rag_models"/"data"/"gtpldc.pdf", query=args.query)
    """
    query ="Pháp luật có những chức năng cơ bản nào?"
    answer = ask(query=query)
    print(f"\nCâu hỏi: {query}")
    print(f"Trả lời: {answer}")
    
    
    
    
    
    

    

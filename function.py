"""
from langchain_community.document_loaders import PyMuPDFLoader
from rag_models.pre_process.normalize_text import BasePreProcess
doc = PyMuPDFLoader("rag_models/data/gtpldc.pdf").load()  # Load toàn bộ tài liệu PDF
pages = doc[38:40]  # Lấy trang thứ 18 (index bắt đầu từ 0) # In ra nội dung văn bản thô của từng trang
base_preprocessor = BasePreProcess()
raw_text = "                              ".join([p.page_content for p in pages]) 
print(repr(raw_text))
print('\n\n') # Dùng repr() để hiện rõ mọi ký tự ẩn như \n, \t # Lấy nội dung văn bản thô của các trang
for page in pages:
    page.page_content = base_preprocessor.clean(page.page_content)  # Tiền xử lý văn bản thô của từng trang

raw_text = "                              ".join([p.page_content for p in pages]) 
#print(repr(raw_text)) # Dùng repr() để hiện rõ mọi ký tự ẩn như \n, \t # Lấy nội dung văn bản thô của các trang
#full_text = "\n\n".join([p.page_content for p in doc])  # Lấy toàn bộ nội dung văn bản thô của tài liệu
#full_text = BasePreProcess.clean_text(full_text)  # Tiền xử lý văn bản thô
print(repr(raw_text)) # Dùng repr() để hiện rõ mọi ký tự ẩn như \n, \t
  # In ra metadata của trang 50
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from time import time

ollama_url = "https://baseball-losing-recovered-stewart.trycloudflare.com"
evaluator_llm = ChatOllama(
    base_url=ollama_url,
    model = "qwen2.5:7b-instruct",
    temperature=0.0,
)

prompt = "Bạn có đóng vai trò làm giám khảo trong LLM as judge được không?"

message = HumanMessage(content=prompt)

first_time =  time()

evaluator_llm_response = evaluator_llm.invoke(prompt)

last_time = time()

need_time = last_time - first_time

print("Prompt:", prompt)
print("Response:", evaluator_llm_response.content)
print("Time taken:", need_time, "seconds")
"""

import os

# Tắt TorchDynamo/Inductor để tránh tìm kiếm compiler C++ (cl.exe)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TextItem, TableItem

# Cấu hình pipeline an toàn cho môi trường Windows
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
source = "rag_models/data/gtpldc.pdf"
result = converter.convert(source)
doc = result.document

print(doc.export_to_markdown()[:10000])

# 3. Duyệt toàn bộ các node trong cây dữ liệu
print(f"Tổng số phần tử nhận diện: {len(list(doc.iterate_items()))}\n")
"""
for item, level in doc.iterate_items():
    # Lấy thông tin bounding box và số trang
    bbox_info = ""
    if item.prov:
        prov = item.prov[0]
        bbox_info = f"[Trang {prov.page_no} | BBox: ({prov.bbox.l:.1f}, {prov.bbox.t:.1f}, {prov.bbox.r:.1f}, {prov.bbox.b:.1f})]"

    # Phân nhánh xử lý theo kiểu Node
    if isinstance(item, TextItem):
        # TextItem: lấy nhãn (paragraph, section_header, list_item...)
        label = item.label
        text_preview = item.text.strip().replace("\n", " ")
        if len(text_preview) > 60:
            text_preview = text_preview[:60] + "..."
        print(f"[TEXT - {label.upper()}] {bbox_info}")
        print(f"  -> Content: {text_preview}")

    elif isinstance(item, TableItem):
        # TableItem: xuất dữ liệu sang Markdown hoặc Pandas DataFrame
        df = item.export_to_dataframe()
        print(f"\n[TABLE DETECTED] {bbox_info}")
        print(f"  -> Kích thước: {df.shape[0]} hàng x {df.shape[1]} cột")
        print("  -> Dữ liệu bảng (Markdown representation):")
        print(item.export_to_markdown())
        print("-" * 50)
"""
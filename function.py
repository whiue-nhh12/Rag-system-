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
  """
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
import os
from datasets import Dataset
from dotenv import load_dotenv
from main import ask_to_test
from rag_models.llm.geminillm import GeminiLLM
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

# ==========================================
# 1. Khởi tạo Evaluator LLM & Embeddings
# ==========================================
# Khuyên dùng mô hình mạnh làm Giám khảo (né anti-pattern Weak Judge)
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

evaluator_embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Bọc bằng Wrapper của Ragas
ragas_llm = LangchainLLMWrapper(evaluator_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)

# ==========================================
# 2. Chuẩn bị tập dữ liệu Eval (Testset)
# ==========================================
with open("rag_eval/evals/datasets/bm25_test.json", "r", encoding="utf-8") as f:
    data_samples = eval(f.read())

llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-flash-lite-latest")
for sample in data_samples:
    query = sample["user_input"]
    try:
        result = ask_to_test(query=query, llm=llm, k=10)
        sample["model_output"] = result["output"]
        sample["context"] = result["context"]
    except Exception as e:
        print(f"Error processing query '{query}': {e} and id: {sample.get('id', 'N/A')}")
        sample["model_output"] = "Error: " + str(e)



# Chuyển đổi sang HuggingFace Dataset
eval_dataset = Dataset.from_list(data_samples)

# ==========================================
# 3. Chạy Đánh Giá
# ==========================================
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

# Gán LLM & Embedding Giám khảo vào từng metric
for metric in metrics:
    metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings

print("Đang chạy Ragas Evaluation...")
results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

# ==========================================
# 4. Xuất Kết Quả
# ==========================================
print("\n=== KẾT QUẢ TỔNG QUAN ===")
print(results)

# Chuyển sang Pandas DataFrame để xem chi tiết từng câu
df_results = results.to_pandas()
print("\n=== CHI TIẾT TỪNG TEST CASE ===")
print(df_results[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

# Lưu ra file CSV phục vụ theo dõi (Tracking / Baseline)
df_results.to_csv("rag_eval/evals/experiments/ragas_bm25_eval_report.csv", index=False)
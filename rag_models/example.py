"""
Example usage of LangChain RAG patterns
Shows how to integrate RAG chains into FastAPI endpoints
"""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag_models.test import (
    DocumentLoader,
    RAGChainBuilder,
    get_vector_store,
    get_embeddings,
)
from app.rag_models.chains import AdvancedChainPatterns
from app.rag_models.utils import (
    ConversationMemory,
    DocumentUtils,
    RAGLogger,
    EvaluationUtils,
)
from app.rag_models.config import RAGConfig


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str
    use_history: bool = True


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    question: str
    answer: str
    sources: List[str] = []
    confidence: float = 0.0


class DocumentUploadRequest(BaseModel):
    """Request for uploading documents"""
    file_path: str
    collection_name: str = "company_docs"


# ============================================================================
# EXAMPLE: FASTAPI ENDPOINTS FOR RAG
# ============================================================================

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Initialize components
config = RAGConfig.for_production()
conversation_memory = ConversationMemory(max_history=20)
logger = RAGLogger(log_file="rag_operations.log")
vector_store = None  # Will be initialized on first use


def initialize_vector_store():
    """Initialize vector store from existing documents"""
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()
    return vector_store


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Example:
    ```python
    curl -X POST "http://localhost:8000/api/rag/query" \\
      -H "Content-Type: application/json" \\
      -d '{"question": "What is the company vacation policy?"}'
    ```
    """
    try:
        vector_store = initialize_vector_store()
        
        # Build RAG chain
        rag_chain = RAGChainBuilder.build_qa_chain(
            vector_store,
            system_prompt="You are a helpful company assistant. Answer questions based on company documents."
        )
        
        # Prepare input
        query_input = {
            "question": request.question,
        }
        
        if request.use_history:
            history = conversation_memory.get_history(limit=3)
            # Could enhance prompt with history here
        
        # Generate answer
        answer = rag_chain.invoke(query_input)
        
        # Log the query
        conversation_memory.add_turn(request.question, answer)
        logger.log_query(request.question, answer)
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=[],
            confidence=0.85
        )
    
    except Exception as e:
        logger.log_error(str(e), {"question": request.question})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-documents")
async def upload_documents(request: DocumentUploadRequest):
    """
    Upload documents to the RAG system
    
    Example:
    ```python
    curl -X POST "http://localhost:8000/api/rag/upload-documents" \\
      -H "Content-Type: application/json" \\
      -d '{"file_path": "/path/to/documents", "collection_name": "company_docs"}'
    ```
    """
    try:
        # Load documents
        documents = DocumentLoader.load_document_from_path(request.file_path)
        
        # Split documents
        split_docs = DocumentUtils.split_documents(documents)
        
        # Add metadata
        split_docs = DocumentUtils.add_metadata_to_docs(
            split_docs,
            collection=request.collection_name,
            upload_date=str(__import__('datetime').datetime.now().isoformat())
        )
        
        # Add to vector store
        embeddings = get_embeddings()
        from langchain_community.vectorstores import Chroma
        
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=request.collection_name,
            persist_directory=str(config.vector_store.db_path)
        )
        
        logger.log_query(f"Uploaded {len(documents)} documents", "Success")
        
        return {
            "status": "success",
            "documents_uploaded": len(documents),
            "collection": request.collection_name
        }
    
    except Exception as e:
        logger.log_error(str(e), {"file_path": request.file_path})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fact-check")
async def fact_check(request: QueryRequest):
    """
    Fact-check a statement against documents
    
    Example:
    ```python
    curl -X POST "http://localhost:8000/api/rag/fact-check" \\
      -H "Content-Type: application/json" \\
      -d '{"question": "Employees get 12 days vacation per year"}'
    ```
    """
    try:
        vector_store = initialize_vector_store()
        
        # Build fact-checking chain
        fact_check_chain = AdvancedChainPatterns.build_fact_checking_chain(vector_store)
        
        result = fact_check_chain.invoke({"statement": request.question})
        
        logger.log_query(f"Fact-check: {request.question}", result)
        
        return {
            "statement": request.question,
            "verification": result
        }
    
    except Exception as e:
        logger.log_error(str(e), {"statement": request.question})
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation-history")
async def get_conversation_history(limit: int = 10):
    """
    Get conversation history
    
    Example:
    ```python
    curl "http://localhost:8000/api/rag/conversation-history?limit=5"
    ```
    """
    return {
        "history": conversation_memory.get_history(limit=limit),
        "total_turns": len(conversation_memory.history)
    }


@router.post("/clear-history")
async def clear_history():
    """Clear conversation history"""
    conversation_memory.clear()
    return {"status": "cleared"}


@router.post("/self-reflect")
async def self_reflect_query(request: QueryRequest):
    """
    Query with self-reflection verification
    
    Example:
    ```python
    curl -X POST "http://localhost:8000/api/rag/self-reflect" \\
      -H "Content-Type: application/json" \\
      -d '{"question": "Tell me about vacation policies"}'
    ```
    """
    try:
        vector_store = initialize_vector_store()
        
        # Build self-reflection chain
        reflect_chain = AdvancedChainPatterns.build_self_reflection_chain(vector_store)
        
        result = reflect_chain.invoke({"question": request.question})
        
        logger.log_query(f"Self-reflect: {request.question}", str(result))
        
        return {
            "question": request.question,
            "reflection": result
        }
    
    except Exception as e:
        logger.log_error(str(e), {"question": request.question})
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# USAGE EXAMPLES (for testing without FastAPI)
# ============================================================================

if __name__ == "__main__":
    print("🔧 RAG System Examples")
    print("=" * 50)
    
    # Example 1: Initialize vector store
    print("\n1️⃣  Initializing vector store...")
    vs = get_vector_store()
    print("✅ Vector store initialized")
    
    # Example 2: Query the RAG system
    print("\n2️⃣  Querying RAG system...")
    rag_chain = RAGChainBuilder.build_qa_chain(vs)
    response = rag_chain.invoke({"question": "What is the vacation policy?"})
    print(f"Q: What is the vacation policy?\nA: {response}")
    
    # Example 3: Fact-checking
    print("\n3️⃣  Fact-checking statement...")
    fact_chain = AdvancedChainPatterns.build_fact_checking_chain(vs)
    verification = fact_chain.invoke({"statement": "Employees get 12 days vacation"})
    print(f"Verification: {verification}")
    
    # Example 4: Save conversation
    print("\n4️⃣  Saving conversation...")
    conversation_memory.add_turn("What are IT policies?", "IT policies include...")
    conversation_memory.save_to_file("conversation_log.json")
    print("✅ Conversation saved")

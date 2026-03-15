from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.nlp.rag_engine import RAGEngine

app = FastAPI(
    title="Advanced GenAI NLP Orchestrator API",
    description="A production-ready API for LLM orchestration and RAG pipelines.",
    version="1.0.0"
)

# Global singleton for RAG engine
rag_engine = RAGEngine()

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's input question.")
    use_reranker: bool = Field(default=False, description="Whether to apply semantic reranking.")

class QueryResponse(BaseModel):
    answer: str
    metadata: Optional[dict] = None

class IngestionRequest(BaseModel):
    documents: List[str] = Field(..., description="A list of raw texts to ingest.")

@app.post("/ingest", status_code=201)
async def ingest_docs(request: IngestionRequest):
    """
    Endpoint to ingest raw text documents into the RAG vector store.
    """
    try:
        rag_engine.ingest_documents(request.documents)
        return {"message": f"Successfully ingested {len(request.documents)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def perform_query(request: QueryRequest):
    """
    Endpoint to query the LLM with RAG-enhanced context.
    """
    try:
        if request.use_reranker:
            answer = rag_engine.rerank_and_query(request.question)
        else:
            answer = rag_engine.query(request.question)

        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint for observability.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

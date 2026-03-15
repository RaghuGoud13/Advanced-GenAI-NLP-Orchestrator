from fastapi import FastAPI
from src.api.v1.router import router as v1_router

app = FastAPI(
    title="Comprehensive Agentic AI Framework",
    description="Advanced NLP Orchestrator for complex multi-step reasoning and hybrid retrieval.",
    version="1.0.0"
)

# Include API v1 Router
app.include_router(v1_router, prefix="/api")

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {
        "status": "healthy",
        "service": "Agentic AI Framework",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

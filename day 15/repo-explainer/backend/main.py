import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ingest import ingest_repository
from backend.chain import query_repository

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-based GitHub Repository Explainer Backend",
    description="FastAPI backend to ingest repositories and query them using LangChain, ChromaDB, and Gemini/Groq.",
    version="1.0.0"
)

# Enable CORS for standard web environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response schemas
class IngestRequest(BaseModel):
    url: str = Field(..., description="The HTTPS URL of the public GitHub repository.")
    force_reingest: bool = Field(False, description="If True, existing vector embeddings for the repo will be deleted and recreated.")

class IngestResponse(BaseModel):
    status: str
    repo_id: str
    repo_name: str
    owner: str
    total_files: int
    total_chunks: int
    languages: List[str]
    summary: str

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role in conversation: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")

class QueryRequest(BaseModel):
    repo_id: str = Field(..., description="The repository ID (owner/repo) to query against.")
    query: str = Field(..., description="The user's query or specific question.")
    mode: str = Field(..., description="Analysis mode: 'tech_stack', 'data_flow', 'system_design', or 'chat'.")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Previous chat messages for context (optional).")

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    provider: str


@app.get("/")
def read_root():
    return {"message": "GitHub Repo Explainer API is running."}


@app.post("/api/ingest", response_model=IngestResponse)
def api_ingest(request: IngestRequest):
    """
    Ingests a public GitHub repository. Checks if it already exists, parses code files,
    chunks them, saves to ChromaDB, generates a summary, and returns repo stats.
    """
    logger.info(f"Received ingestion request for URL: {request.url} (force_reingest={request.force_reingest})")
    try:
        data = ingest_repository(request.url, force_reingest=request.force_reingest)
        return IngestResponse(
            status="success",
            repo_id=data["repo_id"],
            repo_name=data["repo_name"],
            owner=data["owner"],
            total_files=data["total_files"],
            total_chunks=data["total_chunks"],
            languages=data["languages"],
            summary=data["summary"]
        )
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
def api_query(request: QueryRequest):
    """
    Runs a search and LLM query on the vector store for the given repository.
    Supports Tech Stack, Data Flow, System Design, and Chat modes.
    """
    logger.info(f"Received query request for repo: {request.repo_id}, mode: {request.mode}")
    
    # Simple validation on mode
    valid_modes = {"tech_stack", "data_flow", "system_design", "chat"}
    if request.mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
        
    # Convert ChatMessage items to dictionaries for chain ingestion
    history_list = []
    if request.chat_history:
        history_list = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
    try:
        response_data = query_repository(
            repo_id=request.repo_id,
            query=request.query,
            mode=request.mode,
            chat_history=history_list
        )
        return QueryResponse(
            answer=response_data["answer"],
            sources=response_data["sources"],
            provider=response_data["provider"]
        )
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

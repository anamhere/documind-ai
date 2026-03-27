"""
DocuMind AI — FastAPI Backend
==============================
Main application file with REST API endpoints.

Endpoints:
  POST /api/upload     — Upload and process a document
  POST /api/chat       — Send a question, get RAG-powered answer
  GET  /api/documents  — List all uploaded documents
  DELETE /api/documents/{doc_id} — Delete a document
  GET  /api/health     — Health check
"""

import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Global RAG Engine Instance ─────────────────────────────────────────────
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes the RAG engine when the server starts.
    """
    global rag_engine

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print("\n" + "=" * 60)
        print("⚠️  GEMINI API KEY NOT SET!")
        print("=" * 60)
        print("Please set your Gemini API key:")
        print("  1. Copy backend/.env.example to backend/.env")
        print("  2. Replace 'your_gemini_api_key_here' with your key")
        print("  3. Get a free key at: https://aistudio.google.com/apikey")
        print("=" * 60 + "\n")
        raise RuntimeError("GEMINI_API_KEY not configured. See instructions above.")

    # Import here to keep startup clean
    from rag_engine import RAGEngine

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    rag_engine = RAGEngine(gemini_api_key=gemini_api_key, data_dir=data_dir)
    print("\n[READY] DocuMind AI is ready!\n")

    yield

    print("\n[SHUTDOWN] DocuMind AI shutting down.\n")


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind AI",
    description="RAG-powered Document Intelligence Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ─────────────────────────────────────────────────────────
# Allow the frontend (running on a different port) to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)

# ─── Serve Frontend Static Files ─────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ─── Request/Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    query: str                        # The user's question
    chat_history: list[dict] = []     # Previous messages for context


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""
    answer: str                       # The LLM's response
    sources: list[str]                # Source document names
    has_context: bool                 # Whether context was found
    chunks_used: int = 0             # Number of chunks used for context


class DocumentResponse(BaseModel):
    """Response body for document operations."""
    doc_id: str
    file_name: str
    chunk_count: int
    message: str


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend HTML page."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "DocuMind AI API is running. Frontend not found."}


@app.get("/api/download/{filename}")
async def download_export(filename: str):
    file_path = os.path.join(rag_engine.data_dir, "exports", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="Export not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint — useful for monitoring."""
    return {
        "status": "healthy",
        "service": "DocuMind AI",
        "documents_loaded": len(rag_engine.documents) if rag_engine else 0,
        "total_chunks": len(rag_engine.chunks) if rag_engine else 0,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.

    Flow:
    1. Validate file type
    2. Save file to disk
    3. Process through RAG pipeline (extract → chunk → embed → index)
    4. Return document info
    """
    global rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".md", ".csv", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file
    upload_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Use UUID to avoid filename collisions
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(upload_dir, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document through RAG pipeline
        doc_info = rag_engine.add_document(file_path, file.filename)

        return {
            "doc_id": doc_info["doc_id"],
            "file_name": file.filename,
            "file_size": doc_info.get("file_size_readable", "Unknown"),
            "chunk_count": doc_info["chunk_count"],
            "text_length": doc_info["text_length"],
            "message": f"Successfully processed '{file.filename}' into {doc_info['chunk_count']} searchable chunks.",
        }

    except ValueError as e:
        # Clean up file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)} | type: {type(e).__name__}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat query using the RAG engine or trigger automation tools."""
    global rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
    query = request.query.strip()
    
    # === AUTOMATION AGENT COMMANDS ===
    if query.startswith("/web "):
        url = query[5:].strip()
        if not url:
            raise HTTPException(status_code=400, detail="URL cannot be empty for /web command.")
        try:
            doc_info = rag_engine.add_website(url)
            return ChatResponse(
                answer=f"✅ Successfully scraped **{url}** and added {doc_info['chunk_count']} chunks to my knowledge base. You can now ask me questions about it!",
                sources=[],
                has_context=True,
                chunks_used=0
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape website: {e}")
            
    elif query.startswith("/export "):
        prompt = query[8:].strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty for /export command.")
        try:
            filename = rag_engine.export_report(prompt)
            return ChatResponse(
                answer=f"📊 **Insight Report Generated!**\n\nI have successfully compiled a comprehensive report based on your documents.\n\n[Download `insight_report.md`](/api/download/{filename})",
                sources=[],
                has_context=True,
                chunks_used=0
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to generate report: {e}")
            
    # === STANDARD RAG QUERY ===
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = rag_engine.generate_response(
            query=request.query,
            chat_history=request.chat_history,
        )
        return ChatResponse(**result)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            return ChatResponse(
                answer="⚠️ **AI Quota Limit Reached.** All available models (Gemini & Gemma) are currently at their trial capacity. Please try again in 60 seconds. *Note: This demonstrates system resiliency to API constraints.*",
                sources=[],
                has_context=False,
                chunks_used=0
            )
        raise HTTPException(status_code=500, detail=f"Error generating response: {error_msg}")


@app.get("/api/documents")
async def list_documents():
    """List all uploaded and processed documents."""
    global rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    return {
        "documents": rag_engine.get_documents(),
        "total_chunks": len(rag_engine.chunks),
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks from the index."""
    global rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    success = rag_engine.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"message": "Document deleted successfully", "doc_id": doc_id}


# ─── Run Server ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  DocuMind AI — Document Intelligence Chatbot")
    print("  Document Upload & Intelligent RAG Chat")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# DocuMind AI — RAG-Powered Document Assistant

DocuMind AI is a professional Document Intelligence Assistant built with **FastAPI**, **Google Gemini**, and a custom **NumPy Vector Index**. It provides a robust, resilient RAG (Retrieval Augmented Generation) pipeline for intelligent document analysis and conversational intelligence.

---

## Key Features

### Resilient Model Orchestrator
DocuMind AI features an automated **Multi-Model Fallback** system. If a Gemini model hits quota limits or is unavailable, the backend automatically shifts to the next available tier (e.g., `Gemini 1.5-Flash` → `Gemini Pro` → `Gemma 3`), ensuring high availability.

### Intelligent Content Scraping
Uses specialized extraction logic to scrape modern Single Page Applications (SPAs) and portals. It handles metadata fallback and deep tag extraction to ensure comprehensive knowledge ingestion.

### Universal Document Support
- **PDF:** High-fidelity text extraction via PyPDF2.
- **Word (DOCX):** Support for professional business documents.
- **Markdown/Text:** Seamless ingestion of codebases and documentation.
- **Web URLs:** Instant scraping into the local Knowledge Base.

### Responsive UI/UX
Optimized for both desktop and mobile devices. Features a glassmorphism sidebar, fluid chat interactions, and real-time processing status.

---

## Architecture

The system follows a standard RAG pipeline:
1. **Frontend**: Vanilla JS/CSS communicating via REST API.
2. **Backend**: FastAPI orchestrating the document pipeline.
3. **Processing**: Text extraction, overlap chunking, and embedding generation.
4. **Search**: NumPy-based cosine similarity for fast context retrieval.
5. **LLM**: Context-grounded generation with automated fallback logic.

---

## Local Development

### 1. Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your `GEMINI_API_KEY` in the `.env` file.
5. Start the server:
   ```bash
   python main.py
   ```

### 2. Accessing the Application
The FastAPI backend serves the frontend automatically. Once the server is running, open:
`http://localhost:8000`

---

## Deployment Configuration

The project is configured for unified deployment on platforms like Render or Railway. 

- **Runtime**: Python 3.10+
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn backend.main:app -k uvicorn.workers.UvicornWorker`

---

## Technical Rationale
Built as a demonstration of **Reliability Engineering** in AI applications. The focus is on handling API constraints (Rate Limits, Model Availability) gracefully through intelligent orchestrators, providing a stable experience for end-users.


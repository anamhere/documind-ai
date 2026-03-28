# 🚀 DocuMind AI — Agentic RAG Assistant

DocuMind AI is a production-grade, "Self-Healing" Agentic RAG (Retrieval Augmented Generation) Assistant designed for seamless document intelligence. Built with **FastAPI**, **Google Gemini**, and a custom-engineered **NumPy Vector Index**, it overcomes standard RAG limitations with automated resilience.

---

## 🌟 Key Features

### 🧠 Self-Healing Model Orchestrator
DocuMind AI features a sophisticated **Multi-Model Fallback** system. If a specific Gemini model hits its free-tier quota (429) or is unavailable, the backend automatically and silently shifts the conversation to the next available model (e.g., `Gemini 1.5-Flash` → `Gemini Pro` → `Gemma 3`). 

### 🌐 Portal-Aware Intelligent Scraper
Standard scrapers fail on modern Single Page Applications (SPAs). Our scraper uses **Metadata Fallback** and **Deep Tag Extraction** to successfully "read" sites like `alactic.io` and `wikipedia.org`, even when the main body content is hidden.

### 📄 Universal Document Support
-   **PDF:** High-fidelity text extraction via PyPDF2.
-   **Word (DOCX):** Full support for professional business documents.
-   **Markdown/Text:** Seamless ingestion of codebases and notes.
-   **Web URLs:** Instant agentic scraping into the Knowledge Base.

### 🛡️ Sync-Integrity Guardian
A custom-built **Vector Watchdog** that detects and repairs "ghost data" or desynchronization between the database and the vector index on startup, ensuring 100% stability.

---

## 🛠️ Architecture

```mermaid
graph TD
    A[Frontend: Vanilla CSS/JS] -->|API| B[FastAPI Backend]
    B -->|Ingest| C[Document Processor]
    C -->|Embeddings| D[NumPy Vector Search]
    B -->|Agentic Commands| E[Scraper/Exporter]
    E -->|Context| F[Resilient LLM Orchestrator]
    F -->|Fallback| G[Gemini 1.5/2.0 / Gemma 3]
    G -->|Response| B
```

---

## 🏃‍♂️ Quick Start (Local)

1. **Backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   # Add GEMINI_API_KEY to .env
   python main.py
   ```

2. **Frontend:**
   Open `frontend/index.html` in any browser or use a Live Server.

---

## ☁️ Deployment Guide

### 🛡️ Backend (Render.com)
1.  Link your GitHub repository to **Render.com**.
2.  Choose **Web Service**.
3.  Environment: `Python 3`.
4.  Build Command: `pip install -r requirements.txt`.
5.  Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
6.  Add `GEMINI_API_KEY` to the **Environment Variables** in Render.

### ⚛️ Frontend (Vercel/Netlify)
1.  Upload the `frontend` directory.
2.  Update `BASE_URL` in `app.js` to point to your new Render URL.

---

## 🏆 Selection Rationale (For Alactic Inc)
DocuMind AI focuses on **Reliability Engineering**. It was built to stay alive during high-pressure demonstrations by handling Rate Limits (429) and Model Availability (404) gracefully. This showcases proactive system design beyond a standard student application.


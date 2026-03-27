"""
RAG (Retrieval Augmented Generation) Engine
============================================
This is the CORE of DocuMind AI. It implements the complete RAG pipeline:

1. CHUNKING    — Split documents into smaller, overlapping pieces
2. EMBEDDING   — Convert text chunks into numerical vectors (embeddings)
3. INDEXING    — Store embeddings in a vector index for fast retrieval
4. RETRIEVAL   — Find the most relevant chunks for a user query
5. GENERATION  — Use retrieved context + query to generate an answer via Gemini

WHY RAG?
- LLMs have knowledge cutoffs (they don't know about YOUR documents)
- LLMs can hallucinate (make up facts)
- RAG grounds the LLM's responses in YOUR actual data
- Result: Accurate, source-cited answers from your documents
"""

import os
import json
import uuid
import numpy as np
import google.generativeai as genai
from typing import Optional, Any
from document_processor import DocumentProcessor

# Force UTF-8 encoding for Windows consoles to prevent UnicodeEncodeError with emojis
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


class RAGEngine:
    """
    The complete RAG pipeline for document question-answering.

    Architecture:
    ┌──────────┐    ┌──────────┐    ┌────────┐    ┌──────────┐
    │ Document │───▶│ Chunking │───▶│ Embed  │───▶│  Vector  │
    │  Upload  │    │          │    │        │    │  Index   │
    └──────────┘    └──────────┘    └────────┘    └──────────┘
                                                       │
    ┌──────────┐    ┌──────────┐    ┌────────┐         │
    │  Answer  │◀───│  Gemini  │◀───│Context │◀────────┘
    │          │    │   LLM    │    │  + Q   │
    └──────────┘    └──────────┘    └────────┘
    """

    def __init__(self, gemini_api_key: str, data_dir: str = "data"):
        """
        Initialize the RAG engine.

        Args:
            gemini_api_key: API key for Google Gemini
            data_dir: Directory to store documents, chunks, and vector index
        """
        # === LLM Setup with Fallback Orchestration ===
        # Configure Google Gemini API Key first!
        genai.configure(api_key=gemini_api_key)
        
        # Final Robust Model Priority for Alactic Interview 
        # (Fallbacks from Gemini to Gemma 3 to guarantee availability)
        self.model_priorities = [
            "models/gemini-1.5-flash", 
            "models/gemini-pro", 
            "models/gemini-pro-preview-12-2025",
            "models/gemini-2.0-flash",
            "models/gemma-3-4b-it"  # Final failover to Gemma (Confirmed working on site)
        ]
        self.llm = None 
        print(f"RAG Engine Initialized. LLM will be lazy-loaded on first call.")

        # === Embedding Setup ===
        # We use Gemini's own embedding model: "models/text-embedding-004"
        # It produces 768-dimensional vectors that capture semantic meaning
        # The embedding model understands the MEANING of text, not just keywords
        # Example: "car" and "automobile" would have very similar embeddings
        self.embedding_model_name = "models/gemini-embedding-001"
        self.embedding_dim = 768  # Gemini text-embedding-004 outputs 768-dim vectors
        print("Using Gemini gemini-embedding-001 for embeddings (768-dim)")

        # === Storage Setup ===
        self.data_dir = data_dir
        self.uploads_dir = os.path.join(data_dir, "uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)

        # === Vector Index (NumPy-based cosine similarity) ===
        # Instead of FAISS, we use a simple NumPy-based vector index
        # This is conceptually identical to FAISS IndexFlatL2 but with no extra dependency
        # For production with millions of vectors, you'd use FAISS or a vector DB like Pinecone
        self.vectors = np.array([], dtype="float32").reshape(0, self.embedding_dim)

        # === Document & Chunk Storage ===
        self.chunks = []         # List of chunk texts
        self.chunk_metadata = [] # Metadata for each chunk (doc_id, position, etc.)
        self.documents = {}      # doc_id -> document info

        # Load any previously saved data
        self._load_state()

    # =========================================================================
    # STEP 1: TEXT CHUNKING
    # =========================================================================
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> list[str]:
        """
        Split text into overlapping chunks.

        WHY CHUNKING?
        - LLMs have token limits (can't process entire books at once)
        - Smaller chunks = more precise retrieval
        - Overlapping ensures we don't lose context at chunk boundaries

        Args:
            text: The full document text
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get chunk end position
            end = min(start + chunk_size, text_length)

            # Try to break at a sentence boundary (period, newline)
            if end < text_length:
                for break_char in ["\n\n", "\n", ". ", "! ", "? ", ", "]:
                    break_pos = text.rfind(break_char, start + chunk_size // 2, end)
                    if break_pos != -1:
                        end = break_pos + len(break_char)
                        break

            # Add chunk and move start with overlap
            chunks.append(text[start:end].strip())
            
            # Prevent infinite loop if start doesn't move
            new_start = end - chunk_overlap
            if new_start <= start:
                start = end
            else:
                start = new_start

        return chunks

    # =========================================================================
    # STEP 2: GENERATE EMBEDDINGS
    # =========================================================================
    def generate_embeddings(self, text_list: list[str]) -> np.ndarray:
        """
        Generate vector embeddings for a list of text chunks.

        Args:
            text_list: List of strings to embed

        Returns:
            NumPy array of shape (num_texts, embedding_dim)
        """
        if not text_list:
            return np.array([], dtype="float32").reshape(0, self.embedding_dim)

        print(f"Generating embeddings for {len(text_list)} chunks...")
        
        try:
            # Gemini embedding API (processes in batches)
            # Use task_type='retrieval_document' for indexing chunks
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text_list,
                task_type="retrieval_document"
            )
            
            # Convert list of embeddings to NumPy array
            embeddings = np.array(result['embedding'], dtype="float32")
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Fallback to zeros (not ideal, but prevents crash)
            return np.zeros((len(text_list), self.embedding_dim), dtype="float32")

    # =========================================================================
    # STEP 3: DOCUMENT INGESTION
    # =========================================================================
    def add_document(self, file_path: str, original_filename: str) -> dict:
        """
        Complete processing pipeline for a single file.

        Flow: Extract Text -> Chunk -> Embed -> Index -> Save
        """
        # Step 1: Extract text
        processor = DocumentProcessor()
        text = processor.extract_text(file_path)
        
        if not text:
            raise ValueError(f"Could not extract any text from '{original_filename}'")

        # Step 2: Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Step 3: Metadata
        file_size = os.path.getsize(file_path)
        doc_info = {
            "doc_id": doc_id,
            "original_filename": original_filename,
            "processed_filename": os.path.basename(file_path),
            "file_size": file_size,
            "file_size_readable": f"{file_size / 1024:.1f} KB",
            "text_length": len(text),
            "chunk_count": 0,
            "summary": "Processing..."
        }

        # Step 4: Chunk text
        chunks = self.chunk_text(text)
        doc_info["chunk_count"] = len(chunks)

        # Step 5: Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Step 6: Add to vector index and storage
        if self.vectors.shape[0] == 0:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "original_filename": original_filename,
            })

        # Step 7: Generate an AI Summary (Resilient Triple-Retry)
        print(f"Generating AI auto-summary for '{original_filename}'...")
        import time
        doc_info["summary"] = "AI is busy..." 
        for attempt in range(3):
            try:
                sample_text = chunks[0] if chunks else ""
                if len(chunks) > 1:
                    sample_text += "\n" + chunks[len(chunks)//2]
                
                # Dynamic prompt: Focus on high-quality synthesis
                prompt_text = "Provide a comprehensive, professional summary of the document." if attempt == 0 else "Summarize this in 1-2 detailed sentences."
                summary_prompt = f"{prompt_text}\n\nExcerpt:\n{sample_text}"
                
                summary_response = self.generate_content_resilient(summary_prompt, generation_config=genai.GenerationConfig(max_output_tokens=512))
                doc_info["summary"] = summary_response.text.strip()
                break # Success!
            except Exception as e:
                if attempt < 2:
                    print(f"Summary attempt {attempt+1} failed ({e}). Retrying in 2s...")
                    time.sleep(2)
                else:
                    print(f"Warning: AI Summary failed after 3 attempts. Setting fallback.")
                    doc_info["summary"] = "Summary not available (AI is busy)."

        # Step 8: Store document info
        self.documents[doc_id] = doc_info

        # Save state to disk
        self._save_state()
        print(f"Indexed '{original_filename}': {len(chunks)} chunks, {len(text)} chars")

        return doc_info

    # =========================================================================
    # STEP 4: RETRIEVAL (Search)
    # =========================================================================
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the most relevant chunks for a question.

        Uses cosine similarity search: dot(query_v, chunk_v)
        """
        if self.vectors.shape[0] == 0:
            return []

        # 1. Embed the query (using task_type='retrieval_query')
        query_result = genai.embed_content(
            model=self.embedding_model_name,
            content=query,
            task_type="retrieval_query"
        )
        query_vector = np.array(query_result['embedding'], dtype="float32")

        # 2. Compute similarity (Dot product of normalized vectors = Cosine Similarity)
        # Normalize index vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized_vectors = self.vectors / norms
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Calculate dot product
        similarities = np.dot(normalized_vectors, query_vector)

        # 3. Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 4. Return formatted results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "score": float(similarities[idx]),
                "metadata": self.chunk_metadata[idx]
            })

        return results

    # =========================================================================
    # STEP 5: GENERATE RESPONSE WITH LLM
    # =========================================================================
    def generate_content_resilient(self, prompt: str, generation_config=None) -> Any:
        """
        Final Resilient Orchestrator: Tries multiple models + Exponential Backoff.
        This provides maximal uptime for the Alactic interview demo.
        """
        import time
        from google.api_core import exceptions
        
        last_error = None
        for model_name in self.model_priorities:
            # 2-stage retry for EACH model (to handle temporary 'minute' quotas)
            for attempt in range(2):
                try:
                    temp_model = genai.GenerativeModel(model_name)
                    response = temp_model.generate_content(prompt, generation_config=generation_config)
                    # Successful response!
                    self.llm = temp_model
                    return response
                except exceptions.ResourceExhausted as e:
                    # 429 Quota Error: Wait 2 seconds and retry once for this model
                    if attempt == 0:
                        print(f"Model {model_name} hit Quota (429). Waiting 2s for backoff...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"Model {model_name} exhausted. Trying next fallback...")
                        last_error = e
                except Exception as e:
                    # Other errors (404, etc.): Skip immediately
                    print(f"Model {model_name} failed: {e}. Moving to next tier.")
                    last_error = e
                    break 
        
        raise last_error if last_error else Exception("All LLM models failed.")

    def generate_response(self, query: str, chat_history: list = None) -> dict:
        """
        Generate a response using the full RAG pipeline.
        """
        # 1. Search for context (Increased TOP_K for better accuracy)
        search_results = self.search(query, top_k=8)
        
        if not search_results:
            return {
                "answer": "I don't have any documents in my memory yet. Please upload some files or use /web so I can help you!",
                "sources": [],
                "has_context": False,
                "chunks_used": 0
            }

        # 2. Build context string
        context_string = ""
        sources = set()
        for i, res in enumerate(search_results):
            context_string += f"\n--- SOURCE {i+1} ({res['metadata']['original_filename']}) ---\n{res['text']}\n"
            sources.add(res['metadata']['original_filename'])

        # 3. Create RAG prompt (Strict Context Enforcement)
        system_prompt = """You are DocuMind AI, a strict Document Intelligence Assistant. 

STRICT RULES:
1. USE ONLY THE PROVIDED CONTEXT. If the information is not there, then you can say you don't have it.
2. SYNTHESIZE & SUMMARIZE: If the information is present (even from headers/metadata), provide a clean, professional summary.
3. NO APOLOGIES: Avoid phrases like "I don't have enough information but..." if you can provide a partial answer based on what you *do* have.
4. CITE YOUR SOURCES.
"""

        full_prompt = f"{system_prompt}\n\nUSER QUESTION: {query}\n\nCONTEXT FROM DOCUMENTS:\n{context_string}"

        # 5. Generate AI Response using resilient fallback
        response = self.generate_content_resilient(
            full_prompt,
            generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=2048)
        )

        return {
            "answer": response.text,
            "sources": list(sources),
            "has_context": True,
            "chunks_used": len(search_results)
        }

    # =========================================================================
    # ADDITIONAL FEATURES (Interview Plus)
    # =========================================================================

    def add_website(self, url: str) -> dict:
        """
        Agentic Automation: Scrape a website and index it into RAG.
        """
        import requests
        from bs4 import BeautifulSoup
        import urllib3
        urllib3.disable_warnings()
        
        print(f"Scraping website: {url}")
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) DocuMind/1.0'}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. SITE-IDENTITY EXTRACTION (For Identity Injection)
        title = soup.title.string if soup.title else url
        meta_desc = soup.find("meta", attrs={"name": "description"})
        desc_content = meta_desc["content"] if meta_desc and meta_desc.has_attr("content") else ""
        
        # 2. PORTAL-AWARE DEEP EXTRACTION
        # Instead of just get_text(), we scan specifically for meaningful blocks
        # This captures links/headers on portals like wikipedia.org
        content_blocks = []
        for tag_name in ["h1", "h2", "h3", "p", "li"]:
            for tag in soup.find_all(tag_name):
                # Only keep blocks with actual text
                block_text = tag.get_text(strip=True)
                if len(block_text) > 20: 
                    content_blocks.append(block_text)
                    
        text = "\n".join(content_blocks)
        
        # 3. SPA/PORTAL FALLBACK: If body is empty or very thin, use Meta/Title
        if len(text) < 300:
            print(f"Body content thin for {url}. Using Metadata + OpenGraph Fallback...")
            text = f"SOURCE IDENTITY: {title}\nDESCRIPTION: {desc_content}\n" + text
            
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            if og_desc and og_desc.has_attr("content"):
                text += f"ABOUT THE SITE: {og_desc['content']}\n"
            
            if len(text.strip()) < 50:
                raise ValueError("Website content too short or protected (No Metadata).")

        # 4. SITE-IDENTITY INJECTION
        # We prepend the identity to the START of the text to ensure it's in EVERY chunk
        identity_prefix = f"[SOURCE: {title} | DESC: {desc_content[:200]}]\n"
        full_text = identity_prefix + text

        doc_id = f"web_{uuid.uuid4().hex[:8]}"
        doc_info = {
            "doc_id": doc_id,
            "original_filename": url,
            "processed_filename": url,
            "file_size": len(response.text),
            "file_size_readable": f"{len(response.text)/1024:.1f} KB",
            "text_length": len(full_text),
            "chunk_count": 0,
            "summary": "Scraping..."
        }

        chunks = self.chunk_text(full_text)
        doc_info["chunk_count"] = len(chunks)
        embeddings = self.generate_embeddings(chunks)

        if self.vectors.shape[0] == 0:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "original_filename": url,
            })
            
        # Optional AI auto-summary (Triple-Retry)
        import time
        doc_info["summary"] = "AI is busy..."
        for attempt in range(3):
            try:
                sample_text = chunks[0] if chunks else text[:500]
                summary_prompt = f"Summarize this website in 1 professional sentence. Be concise.\n\nExcerpt:\n{sample_text}"
                summary_response = self.generate_content_resilient(summary_prompt)
                doc_info["summary"] = summary_response.text.strip()
                break # Success!
            except Exception as e:
                if attempt < 2:
                    print(f"Website Summary attempt {attempt+1} failed. Retrying in 2s...")
                    time.sleep(2)
                else:
                    doc_info["summary"] = "Website scraped (AI summary busy)."
            
        self.documents[doc_id] = doc_info
        self._save_state()
        return doc_info

    def export_report(self, query: str) -> str:
        """
        Agentic Automation: Generate a full Markdown report based on RAG context.
        """
        search_results = self.search(query, top_k=15)
        if not search_results:
            raise ValueError("No context found for report generation.")
            
        context_string = ""
        for i, res in enumerate(search_results):
            context_string += f"\n--- DATA SEGMENT {i+1} ---\n{res['text']}\n"

        export_prompt = f"""You are an expert technical analyst. The user requested an automated report on: 
'{query}'

RULES:
1. Write a comprehensive, multi-paragraph Markdown document based ONLY on the provided context.
2. Include an executive summary, main findings, and bullet points.
3. Cite your sources clearly throughout.

DOCUMENT CONTEXT:
{context_string}
"""
        
        response = self.generate_content_resilient(
            export_prompt,
            generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=4096)
        )

        exports_dir = os.path.join(self.data_dir, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        filename = f"insight_report_{uuid.uuid4().hex[:8]}.md"
        file_path = os.path.join(exports_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
            
        return filename

    def get_documents(self) -> list[dict]:
        """Return list of all uploaded documents."""
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and rebuild the vector index."""
        if doc_id not in self.documents:
            return False

        new_chunks = []
        new_metadata = []
        keep_indices = []
        for i, (chunk, meta) in enumerate(zip(self.chunks, self.chunk_metadata)):
            if meta["doc_id"] != doc_id:
                new_chunks.append(chunk)
                new_metadata.append(meta)
                keep_indices.append(i)

        self.chunks = new_chunks
        self.chunk_metadata = new_metadata
        del self.documents[doc_id]

        if keep_indices:
            self.vectors = self.vectors[keep_indices]
        else:
            self.vectors = np.array([], dtype="float32").reshape(0, self.embedding_dim)

        self._save_state()
        return True

    # =========================================================================
    # STATE PERSISTENCE (Save/Load)
    # =========================================================================
    def _save_state(self):
        """Save chunks, metadata, documents, and vectors to disk."""
        state = {
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "documents": self.documents,
        }
        state_path = os.path.join(self.data_dir, "rag_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # PERSISTENCE FIX: Always save (or overwrite) the vectors file!
        vectors_path = os.path.join(self.data_dir, "vectors.npy")
        if self.vectors.shape[0] > 0:
            np.save(vectors_path, self.vectors)
        elif os.path.exists(vectors_path):
            # If empty but file exists, delete it or save empty array
            np.save(vectors_path, self.vectors)

    def _load_state(self):
        """Load previously saved state from disk with Sync Integrity Check."""
        state_path = os.path.join(self.data_dir, "rag_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                
                temp_chunks = state.get("chunks", [])
                temp_metadata = state.get("chunk_metadata", [])
                temp_docs = state.get("documents", {})

                # Load vectors
                vectors_path = os.path.join(self.data_dir, "vectors.npy")
                temp_vectors = np.array([], dtype="float32").reshape(0, self.embedding_dim)
                if os.path.exists(vectors_path):
                    temp_vectors = np.load(vectors_path)

                # SYNC INTEGRITY CHECK: 
                # If chunks and vectors don't match, the state is corrupted (Index Out of Range risk).
                if len(temp_chunks) != temp_vectors.shape[0]:
                    print(f"CRITICAL: RAG State Desync Detected! (Chunks: {len(temp_chunks)}, Vectors: {temp_vectors.shape[0]})")
                    print("Repairing state: Starting with a clean index to prevent 'Index Out of Range' errors.")
                    # Return and keep empty state
                    return

                # Success: Commit loaded state
                self.chunks = temp_chunks
                self.chunk_metadata = temp_metadata
                self.documents = temp_docs
                self.vectors = temp_vectors

                print(f"Restored {len(self.documents)} documents, {len(self.chunks)} chunks.")
                if self.vectors.shape[0] > 0:
                    print(f"Loaded {self.vectors.shape[0]} vectors from saved index.")
            except Exception as e:
                print(f"Warning: Could not load saved state: {e}")

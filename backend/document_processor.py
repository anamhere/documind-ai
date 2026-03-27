"""
Document Processor Module
Handles extraction and cleaning of text from various document formats (PDF, TXT).
"""

import os
from PyPDF2 import PdfReader
from typing import Optional


class DocumentProcessor:
    """Processes uploaded documents and extracts clean text content."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from a document based on its file extension.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return DocumentProcessor._extract_from_pdf(file_path)
        elif ext == ".docx":
            return DocumentProcessor._extract_from_docx(file_path)
        elif ext in {".txt", ".md", ".csv"}:
            return DocumentProcessor._extract_from_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(DocumentProcessor.SUPPORTED_EXTENSIONS)}"
            )

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        """Extract text from Word document using python-docx."""
        import docx
        doc = docx.Document(file_path)
        text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(text_parts)
        return DocumentProcessor._clean_text(full_text)

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2.

        How it works:
        - PdfReader reads the PDF structure (pages, fonts, text objects)
        - Each page's text is extracted and combined
        - PyPDF2 handles various PDF encodings and layouts
        """
        reader = PdfReader(file_path)
        text_parts = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

        full_text = "\n\n".join(text_parts)
        return DocumentProcessor._clean_text(full_text)

    @staticmethod
    def _extract_from_text(file_path: str) -> str:
        """Extract text from plain text files (TXT, MD, CSV)."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return DocumentProcessor._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize extracted text.

        Steps:
        1. Replace multiple whitespace/newlines with single space
        2. Strip leading/trailing whitespace
        3. Remove null bytes and control characters
        """
        # Remove null bytes
        text = text.replace("\x00", "")

        # Replace multiple newlines with double newline (preserve paragraphs)
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)

        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    @staticmethod
    def get_document_info(file_path: str) -> dict:
        """Get metadata about a document."""
        ext = os.path.splitext(file_path)[1].lower()
        size = os.path.getsize(file_path)

        info = {
            "file_name": os.path.basename(file_path),
            "file_type": ext,
            "file_size_bytes": size,
            "file_size_readable": DocumentProcessor._format_size(size),
        }

        if ext == ".pdf":
            try:
                reader = PdfReader(file_path)
                info["page_count"] = len(reader.pages)
            except Exception:
                info["page_count"] = 0

        return info

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

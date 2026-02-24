import os
import tempfile
import logging
from datetime import datetime

import httpx
import pymupdf4llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Config ---
API_KEY = os.environ.get("DOCLING_API_KEY", "")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "250"))

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docling-api")

# --- App ---
app = FastAPI(
    title="PDF Extractor API",
    description="Extracts structured Markdown from PDFs using PyMuPDF4LLM. Built for n8n workflow integration.",
    version="2.0.0",
)


# --- Models ---
class ExtractRequest(BaseModel):
    url: str
    api_key: str = ""


class ExtractResponse(BaseModel):
    markdown: str
    char_count: int
    page_count: int
    status: str
    processing_time_seconds: float
    source_url: str


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Routes ---
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint for Railway."""
    return HealthResponse(status="ok", version="2.0.0")


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(req: ExtractRequest):
    """
    Download a PDF from the given URL and extract its content
    as structured Markdown using PyMuPDF4LLM.
    """
    start_time = datetime.now()

    # Auth check
    if API_KEY and req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.url:
        raise HTTPException(status_code=400, detail="URL is required")

    logger.info(f"Processing PDF: {req.url[:100]}...")

    try:
        # Download PDF to temp file (streaming for large files)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            total_bytes = 0
            max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0),
                follow_redirects=True,
            ) as client:
                async with client.stream("GET", req.url) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            os.unlink(tmp.name)
                            raise HTTPException(
                                status_code=413,
                                detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit",
                            )
                        tmp.write(chunk)

            tmp_path = tmp.name
            logger.info(f"Downloaded {total_bytes / (1024*1024):.1f}MB to {tmp_path}")

        # Extract markdown using PyMuPDF4LLM
        logger.info("Starting PDF extraction...")
        markdown = pymupdf4llm.to_markdown(tmp_path)
        
        # Get page count
        import pymupdf
        doc = pymupdf.open(tmp_path)
        page_count = len(doc)
        doc.close()
        
        logger.info(f"Extraction complete: {len(markdown)} chars, {page_count} pages")

        # Clean up temp file
        os.unlink(tmp_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {elapsed:.1f}s")

        return ExtractResponse(
            markdown=markdown,
            char_count=len(markdown),
            page_count=page_count,
            status="success",
            processing_time_seconds=round(elapsed, 1),
            source_url=req.url,
        )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download PDF: HTTP {e.response.status_code}",
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

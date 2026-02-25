import os
import tempfile
import logging
from datetime import datetime

import httpx
import pymupdf  # PyMuPDF
import pymupdf4llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Config ---
API_KEY = os.environ.get("DOCLING_API_KEY", "")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "250"))

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-extract-api")

# --- App ---
app = FastAPI(
    title="PDF Extractor API",
    description="Extracts structured Markdown and hyperlinks from PDFs. Built for n8n workflow integration.",
    version="3.0.0",
)


# --- Models ---
class ExtractRequest(BaseModel):
    url: str
    api_key: str = ""
    page_start: int = 1
    page_end: int = 0  # 0 = all pages


class LinkData(BaseModel):
    page_num: int
    text: str
    url: str


class ExtractResponse(BaseModel):
    markdown: str
    char_count: int
    page_count: int
    pages_extracted: str
    links: list[LinkData]
    link_count: int
    status: str
    processing_time_seconds: float
    source_url: str


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Helpers ---
def extract_links_from_pdf(doc) -> list[dict]:
    """Extract all hyperlinks from every page of the PDF."""
    links = []
    seen = set()

    for page_num in range(len(doc)):
        page = doc[page_num]

        for link in page.get_links():
            uri = link.get("uri", "")
            if not uri or not uri.startswith("http"):
                continue

            # Get the text near the link rectangle
            rect = link.get("from")
            text = ""
            if rect:
                text = page.get_text("text", clip=rect).strip()

            # Skip duplicate URL+text combos
            key = f"{uri}|{text}"
            if key in seen:
                continue
            seen.add(key)

            links.append({
                "page_num": page_num + 1,
                "text": text or uri,
                "url": uri,
            })

    return links


# --- Routes ---
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="3.0.0")


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(req: ExtractRequest):
    """
    Download a PDF and extract:
    - Structured Markdown from a page range (default: all pages)
    - All hyperlinks from the entire document
    """
    start_time = datetime.now()

    # Auth
    if API_KEY and req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.url:
        raise HTTPException(status_code=400, detail="URL is required")

    logger.info(f"Processing PDF: {req.url[:100]}...")

    try:
        # Download PDF
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

        # Open document
        doc = pymupdf.open(tmp_path)
        page_count = len(doc)
        logger.info(f"PDF has {page_count} pages")

        # Determine page range for markdown extraction
        start = max(0, req.page_start - 1)  # Convert to 0-indexed
        end = page_count if req.page_end <= 0 else min(req.page_end, page_count)
        page_range = list(range(start, end))
        pages_label = f"{start + 1}-{end} of {page_count}"

        # Extract markdown from page range only
        logger.info(f"Extracting markdown from pages {pages_label}...")
        markdown = pymupdf4llm.to_markdown(tmp_path, pages=page_range)
        logger.info(f"Markdown: {len(markdown)} chars from {len(page_range)} pages")

        # Extract hyperlinks from ALL pages
        logger.info("Extracting hyperlinks from full document...")
        raw_links = extract_links_from_pdf(doc)
        links = [
            LinkData(page_num=lnk["page_num"], text=lnk["text"], url=lnk["url"])
            for lnk in raw_links
        ]
        logger.info(f"Found {len(links)} unique hyperlinks")

        doc.close()
        os.unlink(tmp_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {elapsed:.1f}s")

        return ExtractResponse(
            markdown=markdown,
            char_count=len(markdown),
            page_count=page_count,
            pages_extracted=pages_label,
            links=links,
            link_count=len(links),
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

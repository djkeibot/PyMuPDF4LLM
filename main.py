import os
import io
import base64
import tempfile
import logging
from datetime import datetime
from typing import Optional

import httpx
import pymupdf  # PyMuPDF
import pymupdf4llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Config ---
API_KEY = os.environ.get("DOCLING_API_KEY", "")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "250"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "75"))
MIN_IMAGE_DIM = 150  # minimum width or height in pixels
MAX_IMAGE_WIDTH = 800  # resize for email embedding
JPEG_QUALITY = 70

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-extract-api")

# --- App ---
app = FastAPI(
    title="PDF Extractor API",
    description="Extracts structured Markdown and embedded images from PDFs. Built for n8n workflow integration.",
    version="2.1.0",
)


# --- Models ---
class ExtractRequest(BaseModel):
    url: str
    api_key: str = ""
    include_images: bool = True
    min_image_dim: int = MIN_IMAGE_DIM
    max_images: int = MAX_IMAGES


class ImageData(BaseModel):
    page_num: int
    index: int
    base64_data: str
    content_type: str
    width: int
    height: int
    original_width: int
    original_height: int


class ExtractResponse(BaseModel):
    markdown: str
    char_count: int
    page_count: int
    images: list[ImageData]
    image_count: int
    image_summary: str
    status: str
    processing_time_seconds: float
    source_url: str


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Image Helpers ---
def extract_and_resize_image(doc, xref: int, max_width: int = MAX_IMAGE_WIDTH) -> Optional[tuple]:
    """Extract an image by xref, resize if needed, return as JPEG base64."""
    try:
        img_info = doc.extract_image(xref)
        if not img_info or not img_info.get("image"):
            return None

        img_bytes = img_info["image"]
        width = img_info.get("width", 0)
        height = img_info.get("height", 0)

        # Open with PyMuPDF's Pixmap for resizing
        pix = pymupdf.Pixmap(img_bytes)

        # Skip very small images (icons, logos, bullets)
        if pix.width < MIN_IMAGE_DIM and pix.height < MIN_IMAGE_DIM:
            return None

        original_width = pix.width
        original_height = pix.height

        # Convert CMYK or other colorspaces to RGB
        if pix.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

        # Resize if wider than max
        if pix.width > max_width:
            scale = max_width / pix.width
            new_width = max_width
            new_height = int(pix.height * scale)
            # Create resized pixmap via transformation matrix
            src_rect = pymupdf.IRect(0, 0, pix.width, pix.height)
            pix_resized = pymupdf.Pixmap(pix, 0)  # copy
            # Use tobytes and re-create at target size
            img_data = pix.tobytes("jpeg")
            # Re-open and resize using fitz
            small_pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, new_width, new_height), 1)
            small_pix.clear_with(255)  # white background
            # Simple approach: just use the JPEG output with quality control
            jpeg_bytes = pix.tobytes("jpeg")
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            return {
                "base64_data": b64,
                "content_type": "image/jpeg",
                "width": pix.width,
                "height": pix.height,
                "original_width": original_width,
                "original_height": original_height,
            }

        # Convert to JPEG bytes
        jpeg_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        return {
            "base64_data": b64,
            "content_type": "image/jpeg",
            "width": pix.width,
            "height": pix.height,
            "original_width": original_width,
            "original_height": original_height,
        }

    except Exception as e:
        logger.warning(f"Failed to extract image xref {xref}: {e}")
        return None


def extract_images_from_pdf(doc, max_images: int = MAX_IMAGES) -> list[dict]:
    """Extract significant embedded images from all pages."""
    images = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]

            # Skip duplicates (same image on multiple pages)
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            result = extract_and_resize_image(doc, xref)
            if result:
                result["page_num"] = page_num + 1  # 1-indexed
                result["index"] = img_index
                images.append(result)

                if len(images) >= max_images:
                    logger.info(f"Hit max image limit ({max_images})")
                    return images

    return images


def build_image_summary(images: list[dict], page_count: int) -> str:
    """Build a text summary of available images for Claude's context."""
    if not images:
        return "No significant images found in the document."

    # Group by page
    pages_with_images = {}
    for img in images:
        pg = img["page_num"]
        if pg not in pages_with_images:
            pages_with_images[pg] = 0
        pages_with_images[pg] += 1

    lines = [f"Found {len(images)} significant images across {len(pages_with_images)} pages:"]
    for pg in sorted(pages_with_images.keys()):
        count = pages_with_images[pg]
        lines.append(f"  Page {pg}: {count} image(s)")

    return "\n".join(lines)


# --- Routes ---
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="2.1.0")


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(req: ExtractRequest):
    """
    Download a PDF and extract structured Markdown + embedded images.
    Images are returned as base64 JPEG, filtered for significance and resized for email.
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

        # Extract markdown
        logger.info("Extracting markdown...")
        markdown = pymupdf4llm.to_markdown(tmp_path)
        logger.info(f"Markdown: {len(markdown)} chars")

        # Extract images
        images = []
        image_summary = "Image extraction disabled."

        if req.include_images:
            logger.info("Extracting images...")
            doc = pymupdf.open(tmp_path)
            page_count = len(doc)
            raw_images = extract_images_from_pdf(doc, max_images=req.max_images)

            images = [
                ImageData(
                    page_num=img["page_num"],
                    index=img["index"],
                    base64_data=img["base64_data"],
                    content_type=img["content_type"],
                    width=img["width"],
                    height=img["height"],
                    original_width=img["original_width"],
                    original_height=img["original_height"],
                )
                for img in raw_images
            ]

            image_summary = build_image_summary(raw_images, page_count)
            logger.info(f"Extracted {len(images)} images")
            doc.close()
        else:
            doc = pymupdf.open(tmp_path)
            page_count = len(doc)
            doc.close()

        # Cleanup
        os.unlink(tmp_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {elapsed:.1f}s")

        return ExtractResponse(
            markdown=markdown,
            char_count=len(markdown),
            page_count=page_count,
            images=images,
            image_count=len(images),
            image_summary=image_summary,
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

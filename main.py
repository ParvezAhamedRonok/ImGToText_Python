# :::writing{variant="standard" id="74163" title="Final FastAPI OCR Backend — main.py"}

from __future__ import annotations

import asyncio
import os
from typing import Optional

# =========================================================
# PADDLE / PADDLEX COMPATIBILITY
# IMPORTANT:
# These MUST be set before importing anything that imports
# Paddle/PaddleOCR.
# =========================================================

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault(
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
    "True",
)

# Reduce noisy/unsafe threading behaviour on small CPU servers.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# =========================================================
# IMPORTS
# =========================================================

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

from ocr_engine import process_image


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000",
).strip()

MAX_FILE_SIZE_MB = int(
    os.getenv(
        "MAX_FILE_SIZE_MB",
        "15",
    )
)

MAX_FILE_SIZE = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)

MAX_ANSWER_IMAGES = 4


# =========================================================
# ALLOWED IMAGE TYPES
# =========================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="IELTS Plus OCR API",
    description="CPU OCR API using PaddleOCR",
    version="3.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "IELTS Plus OCR API",
        "ocr": "PaddleOCR",
        "device": "CPU",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }


# =========================================================
# OCR HEALTH
# =========================================================

@app.get("/api/ocr/health")
async def ocr_health():
    return {
        "status": "ready",
        "engine": "PaddleOCR",
        "device": "CPU",
    }


# =========================================================
# READ IMAGE FILE
# =========================================================

async def read_image_file(
    file: UploadFile,
) -> bytes:
    """
    Read and validate one uploaded image.

    Validation happens before OCR so invalid files never
    reach PaddleOCR.
    """

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Image file is required.",
        )

    # -----------------------------------------------------
    # Content type
    # -----------------------------------------------------

    content_type = (
        file.content_type or ""
    ).lower().strip()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG, PNG and WEBP "
                "images are supported."
            ),
        )

    # -----------------------------------------------------
    # Read bytes
    # -----------------------------------------------------

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    # -----------------------------------------------------
    # File size
    # -----------------------------------------------------

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large. Maximum size is "
                f"{MAX_FILE_SIZE_MB} MB."
            ),
        )

    return data


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

async def process_one_image(
    image_bytes: bytes,
    label: str,
) -> dict:
    """
    Run the synchronous PaddleOCR code in a worker thread.

    This is important because PaddleOCR/OpenCV inference is
    blocking CPU work. Calling process_image() directly
    inside an async FastAPI endpoint would block the event
    loop.
    """

    print()
    print(
        "----------------------------------------"
    )
    print(
        f"[OCR] Processing {label}"
    )
    print(
        "[OCR] File size:",
        round(
            len(image_bytes) / 1024 / 1024,
            2,
        ),
        "MB",
    )
    print(
        "----------------------------------------"
    )

    try:
        result = await asyncio.to_thread(
            process_image,
            image_bytes,
        )

    except Exception as exc:
        print(
            f"[OCR] {label} processing failed:"
        )
        print(
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to process {label}."
            ),
        )

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                f"OCR returned an invalid result "
                f"for {label}."
            ),
        )

    text = (
        result.get("text", "")
        or ""
    )

    confidence = result.get(
        "confidence"
    )

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # -----------------------------------------------------
    # Normalize confidence
    # -----------------------------------------------------

    if confidence is not None:
        try:
            confidence = float(
                confidence
            )

            if confidence < 0:
                confidence = 0.0

            if confidence > 100:
                confidence = 100.0

            confidence = round(
                confidence,
                2,
            )

        except (
            TypeError,
            ValueError,
        ):
            confidence = None

    print(
        f"[OCR] {label} completed."
    )

    print(
        "[OCR] Confidence:",
        confidence,
    )

    print(
        "[OCR] Text length:",
        len(text),
    )

    return {
        "text": text,
        "confidence": confidence,
    }


# =========================================================
# OCR BATCH ENDPOINT
# =========================================================

@app.post("/api/ocr/batch")
async def ocr_batch(
    question: Optional[UploadFile] = File(
        default=None
    ),
    answers: list[UploadFile] | None = File(
        default=None
    ),
):
    """
    Frontend-compatible OCR endpoint.

    Expected multipart fields:

        question
        answers
        answers
        answers
        answers

    Maximum:
        1 question
        4 answers
    """

    answers = answers or []

    # =====================================================
    # VALIDATE ANSWER COUNT
    # =====================================================

    if len(answers) > MAX_ANSWER_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {MAX_ANSWER_IMAGES} "
                f"answer images are allowed."
            ),
        )

    # =====================================================
    # AT LEAST ONE IMAGE
    # =====================================================

    if (
        question is None
        and not answers
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload at least one image."
            ),
        )

    # =====================================================
    # RESULT VARIABLES
    # =====================================================

    question_text = ""
    question_confidence = None

    answer_texts: list[str] = []
    answer_confidences: list[float] = []

    # =====================================================
    # QUESTION
    # =====================================================

    if question is not None:

        question_bytes = (
            await read_image_file(
                question
            )
        )

        question_result = (
            await process_one_image(
                question_bytes,
                "question image",
            )
        )

        question_text = (
            question_result["text"]
        )

        question_confidence = (
            question_result["confidence"]
        )

    # =====================================================
    # ANSWERS
    # =====================================================

    for index, answer in enumerate(
        answers,
        start=1,
    ):
        answer_bytes = (
            await read_image_file(
                answer
            )
        )

        answer_result = (
            await process_one_image(
                answer_bytes,
                f"answer image {index}",
            )
        )

        text = answer_result["text"]
        confidence = (
            answer_result["confidence"]
        )

        # -------------------------------------------------
        # Keep only non-empty OCR results.
        # This preserves the frontend's existing behaviour.
        # -------------------------------------------------

        if text:
            answer_texts.append(
                text
            )

            if confidence is not None:
                answer_confidences.append(
                    confidence
                )

    # =====================================================
    # ANSWER CONFIDENCE
    # =====================================================

    answer_confidence = None

    if answer_confidences:
        answer_confidence = round(
            sum(answer_confidences)
            / len(answer_confidences),
            2,
        )

    # =====================================================
    # COMBINED ANSWER TEXT
    # =====================================================

    answer_text = "\n\n".join(
        answer_texts
    ).strip()

    # =====================================================
    # FRONTEND-COMPATIBLE RESPONSE
    # =====================================================

    response = {
        "success": True,

        # Frontend reads this as a string.
        "question": question_text,

        # Frontend already supports this field.
        "question_confidence":
            question_confidence,

        # Keep array for compatibility/future use.
        "answers": answer_texts,

        # Frontend's primary answer field.
        "answer_text":
            answer_text,

        # Average confidence across detected answers.
        "answer_confidence":
            answer_confidence,

        "total_answers":
            len(answer_texts),
    }

    print()
    print(
        "========================================"
    )
    print(
        "[OCR] BATCH COMPLETE"
    )
    print(
        "[OCR] Question detected:",
        bool(question_text),
    )
    print(
        "[OCR] Answers detected:",
        len(answer_texts),
    )
    print(
        "[OCR] Question confidence:",
        question_confidence,
    )
    print(
        "[OCR] Answer confidence:",
        answer_confidence,
    )
    print(
        "========================================"
    )
    print()

    return response


# =========================================================
# GLOBAL HTTP ERROR HANDLER
# =========================================================

@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request,
    exc: HTTPException,
):
    """
    FastAPI's default HTTPException response is already
    JSON-compatible. This handler is intentionally kept
    simple so the frontend always receives a `detail` field.
    """

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": str(
                exc.detail
            ),
        },
    )


# =========================================================
# GLOBAL UNEXPECTED ERROR HANDLER
# =========================================================

@app.exception_handler(
    Exception
)
async def unexpected_exception_handler(
    request,
    exc: Exception,
):
    """
    Last safety net.

    An unexpected exception should produce a clean JSON
    response instead of an HTML error page or server crash.
    """

    from fastapi.responses import JSONResponse

    print(
        "[API] Unexpected server error:"
    )
    print(
        repr(exc)
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": (
                "Internal OCR server error. "
                "Please try again."
            ),
        },
    )

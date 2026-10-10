# :::writing{variant="standard" id="31684" title="Final CPU PaddleOCR Engine — ocr_engine.py"}

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

# =========================================================
# PADDLE CONFIGURATION
# IMPORTANT:
# MUST BE SET BEFORE IMPORTING PaddleOCR.
# =========================================================

os.environ.setdefault(
    "FLAGS_use_mkldnn",
    "0",
)

os.environ.setdefault(
    "FLAGS_enable_pir_api",
    "0",
)

os.environ.setdefault(
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
    "True",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "4",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "4",
)


# =========================================================
# IMPORTS
# =========================================================

import cv2
import numpy as np

from paddleocr import PaddleOCR


# =========================================================
# CONFIGURATION
# =========================================================

OCR_LANGUAGE = "en"

OCR_DEVICE = "cpu"

ENABLE_MKLDNN = False

CPU_THREADS = 4

# ---------------------------------------------------------
# Large images are downscaled before OCR.
#
# We NEVER upscale small images.
# ---------------------------------------------------------

MAX_IMAGE_DIMENSION = 1800


# ---------------------------------------------------------
# PaddleOCR is intentionally run ONCE per uploaded image.
#
# The previous version used:
#
#   pass 1 -> original
#   pass 2 -> thresholded
#
# That could double CPU inference time.
#
# To keep CPU usage predictable and stay comfortably within
# the frontend's 180-second timeout, this version performs
# one OCR inference per image.
# ---------------------------------------------------------


# =========================================================
# OCR RESULT
# =========================================================

@dataclass
class OCRResult:
    text: str
    confidence: float


# =========================================================
# CREATE PADDLE OCR
# =========================================================

def create_ocr() -> PaddleOCR:
    """
    Initialize PaddleOCR exactly once.

    A compatibility fallback is included because PaddleOCR
    versions do not always expose exactly the same
    constructor arguments.
    """

    print(
        "[OCR] Initializing PaddleOCR..."
    )

    try:
        engine = PaddleOCR(
            lang=OCR_LANGUAGE,
            device=OCR_DEVICE,
            enable_mkldnn=ENABLE_MKLDNN,
            cpu_threads=CPU_THREADS,

            # Disable unnecessary document-level models.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    except TypeError as exc:
        print(
            "[OCR] Extended PaddleOCR configuration "
            "was not accepted."
        )

        print(
            "[OCR] Switching to compatibility mode."
        )

        print(
            "[OCR] Constructor error:",
            repr(exc),
        )

        try:
            engine = PaddleOCR(
                lang=OCR_LANGUAGE,
                device=OCR_DEVICE,
                enable_mkldnn=False,
            )

        except Exception as fallback_exc:
            print(
                "[OCR] PaddleOCR compatibility "
                "initialization failed."
            )

            print(
                repr(fallback_exc)
            )

            raise

    except Exception as exc:
        print(
            "[OCR] PaddleOCR initialization failed."
        )

        print(
            repr(exc)
        )

        raise

    print(
        "[OCR] PaddleOCR loaded successfully."
    )

    print(
        "[OCR] Device: CPU"
    )

    print(
        "[OCR] MKL-DNN: disabled"
    )

    print(
        "[OCR] CPU threads:",
        CPU_THREADS,
    )

    return engine


# =========================================================
# LOAD OCR ONCE
# =========================================================

ocr = create_ocr()


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(
    text: str,
) -> str:
    """
    Normalize OCR text without aggressively changing the
    actual content.
    """

    if not text:
        return ""

    text = str(
        text
    )

    # -----------------------------------------------------
    # Normalize line endings
    # -----------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # -----------------------------------------------------
    # Remove invalid control characters
    # -----------------------------------------------------

    text = "".join(
        char
        for char in text
        if (
            char == "\n"
            or char == "\t"
            or ord(char) >= 32
        )
    )

    # -----------------------------------------------------
    # Clean individual lines
    # -----------------------------------------------------

    lines: list[str] = []

    for line in text.split(
        "\n"
    ):
        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        )

        line = line.strip()

        if line:
            lines.append(
                line
            )

    result = "\n".join(
        lines
    )

    # -----------------------------------------------------
    # Prevent excessive blank lines
    # -----------------------------------------------------

    result = re.sub(
        r"\n{3,}",
        "\n\n",
        result,
    )

    return result.strip()


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(
    image_bytes: bytes,
) -> np.ndarray:
    """
    Decode uploaded bytes into a BGR OpenCV image.
    """

    if not image_bytes:
        raise ValueError(
            "Image data is empty."
        )

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Unable to decode uploaded image."
        )

    return image


# =========================================================
# RESIZE IMAGE
# =========================================================

def resize_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Downscale very large images.

    Small/normal images are returned unchanged.

    No upscaling is performed because upscaling increases
    CPU cost and does not reliably improve OCR quality.
    """

    height, width = (
        image.shape[:2]
    )

    current_max = max(
        width,
        height,
    )

    # -----------------------------------------------------
    # Already small enough.
    # -----------------------------------------------------

    if (
        current_max
        <= MAX_IMAGE_DIMENSION
    ):
        return image

    # -----------------------------------------------------
    # Calculate downscale ratio.
    # -----------------------------------------------------

    scale = (
        MAX_IMAGE_DIMENSION
        / float(current_max)
    )

    new_width = max(
        1,
        int(
            width * scale
        ),
    )

    new_height = max(
        1,
        int(
            height * scale
        ),
    )

    print(
        "[OCR] Resizing:",
        f"{width}x{height}",
        "->",
        f"{new_width}x{new_height}",
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# =========================================================
# LIGHT IMAGE NORMALIZATION
# =========================================================

def prepare_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Perform cheap CPU preprocessing only.

    We deliberately avoid expensive multi-pass OCR.

    PaddleOCR itself performs the primary text detection
    and recognition.
    """

    # -----------------------------------------------------
    # Resize only when necessary.
    # -----------------------------------------------------

    image = resize_image(
        image
    )

    return image


# =========================================================
# GET RESULT JSON
# =========================================================

def get_result_json(
    result: Any,
) -> Any:
    """
    Extract PaddleOCR result data across common PaddleOCR
    3.x result representations.
    """

    # -----------------------------------------------------
    # .json
    # -----------------------------------------------------

    if hasattr(
        result,
        "json",
    ):
        try:
            data = result.json

            if callable(data):
                data = data()

            return data

        except Exception:
            pass

    # -----------------------------------------------------
    # .to_json()
    # -----------------------------------------------------

    if hasattr(
        result,
        "to_json",
    ):
        try:
            return result.to_json()

        except Exception:
            pass

    # -----------------------------------------------------
    # Already dict
    # -----------------------------------------------------

    if isinstance(
        result,
        dict,
    ):
        return result

    return None


# =========================================================
# PARSE SINGLE RESULT
# =========================================================

def parse_single_result(
    item: Any,
) -> tuple[
    list[str],
    list[float],
    list[list[float]],
]:
    """
    Parse one PaddleOCR result object.
    """

    texts: list[str] = []
    scores: list[float] = []
    boxes: list[list[float]] = []

    data = get_result_json(
        item
    )

    if data is None:
        return (
            texts,
            scores,
            boxes,
        )

    # -----------------------------------------------------
    # JSON string
    # -----------------------------------------------------

    if isinstance(
        data,
        str,
    ):
        try:
            data = json.loads(
                data
            )

        except Exception:
            return (
                texts,
                scores,
                boxes,
            )

    # -----------------------------------------------------
    # Some PaddleOCR structures use "res".
    # -----------------------------------------------------

    if isinstance(
        data,
        dict,
    ):
        if "res" in data:
            data = data["res"]

    if not isinstance(
        data,
        dict,
    ):
        return (
            texts,
            scores,
            boxes,
        )

    # -----------------------------------------------------
    # PaddleOCR 3.x fields.
    # -----------------------------------------------------

    rec_texts = data.get(
        "rec_texts",
        [],
    )

    rec_scores = data.get(
        "rec_scores",
        [],
    )

    rec_boxes = data.get(
        "rec_boxes",
        [],
    )

    # -----------------------------------------------------
    # Convert to lists safely.
    # -----------------------------------------------------

    try:
        rec_texts = list(
            rec_texts
        )
    except Exception:
        rec_texts = []

    try:
        rec_scores = list(
            rec_scores
        )
    except Exception:
        rec_scores = []

    try:
        rec_boxes = list(
            rec_boxes
        )
    except Exception:
        rec_boxes = []

    # -----------------------------------------------------
    # Extract detected text.
    # -----------------------------------------------------

    for index, raw_text in enumerate(
        rec_texts
    ):
        text = str(
            raw_text
        ).strip()

        if not text:
            continue

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        score = 0.0

        if index < len(
            rec_scores
        ):
            try:
                score = float(
                    rec_scores[index]
                )
            except (
                ValueError,
                TypeError,
            ):
                score = 0.0

        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        box: list[float] = []

        if index < len(
            rec_boxes
        ):
            try:
                box = (
                    np.asarray(
                        rec_boxes[index]
                    )
                    .reshape(-1)
                    .astype(float)
                    .tolist()
                )

            except Exception:
                box = []

        texts.append(
            text
        )

        scores.append(
            score
        )

        boxes.append(
            box
        )

    return (
        texts,
        scores,
        boxes,
    )


# =========================================================
# PARSE PADDLE RESULT
# =========================================================

def parse_paddle_result(
    result: Any,
) -> tuple[
    list[str],
    list[float],
    list[list[float]],
]:
    """
    Parse the iterable returned by PaddleOCR.predict().
    """

    texts: list[str] = []
    scores: list[float] = []
    boxes: list[list[float]] = []

    try:
        items = list(
            result
        )

    except Exception:
        items = [
            result
        ]

    for item in items:
        (
            item_texts,
            item_scores,
            item_boxes,
        ) = parse_single_result(
            item
        )

        texts.extend(
            item_texts
        )

        scores.extend(
            item_scores
        )

        boxes.extend(
            item_boxes
        )

    return (
        texts,
        scores,
        boxes,
    )


# =========================================================
# SORT TEXT
# =========================================================

def sort_text_items(
    texts: list[str],
    scores: list[float],
    boxes: list[list[float]],
) -> tuple[
    list[str],
    list[float],
]:
    """
    Sort detected text approximately top-to-bottom,
    then left-to-right.
    """

    if not texts:
        return [], []

    items = []

    for index, text in enumerate(
        texts
    ):
        score = (
            scores[index]
            if index < len(scores)
            else 0.0
        )

        box = (
            boxes[index]
            if index < len(boxes)
            else []
        )

        x = 0.0
        y = float(
            index
        )

        # -------------------------------------------------
        # rec_boxes normally contains:
        # [x1, y1, x2, y2]
        # -------------------------------------------------

        if len(box) >= 2:
            try:
                x = float(
                    box[0]
                )

                y = float(
                    box[1]
                )

            except Exception:
                pass

        items.append(
            (
                y,
                x,
                text,
                score,
            )
        )

    # -----------------------------------------------------
    # Top -> bottom
    # Left -> right
    # -----------------------------------------------------

    items.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    sorted_texts = [
        item[2]
        for item in items
    ]

    sorted_scores = [
        item[3]
        for item in items
    ]

    return (
        sorted_texts,
        sorted_scores,
    )


# =========================================================
# RUN PADDLE OCR
# =========================================================

def run_paddle_ocr(
    image: np.ndarray,
) -> OCRResult:
    """
    Exactly ONE PaddleOCR inference for this image.
    """

    try:
        print(
            "[OCR] Running PaddleOCR..."
        )

        result = ocr.predict(
            image
        )

    except Exception as exc:
        print(
            "[OCR] PaddleOCR inference failed:"
        )

        print(
            repr(exc)
        )

        # Return a safe result rather than allowing the
        # worker/server to crash.
        return OCRResult(
            text="",
            confidence=0.0,
        )

    # -----------------------------------------------------
    # Parse
    # -----------------------------------------------------

    (
        texts,
        scores,
        boxes,
    ) = parse_paddle_result(
        result
    )

    if not texts:
        return OCRResult(
            text="",
            confidence=0.0,
        )

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    (
        texts,
        scores,
    ) = sort_text_items(
        texts,
        scores,
        boxes,
    )

    # -----------------------------------------------------
    # Build text
    # -----------------------------------------------------

    final_text = "\n".join(
        texts
    )

    final_text = clean_text(
        final_text
    )

    if not final_text:
        return OCRResult(
            text="",
            confidence=0.0,
        )

    # -----------------------------------------------------
    # Average confidence
    # -----------------------------------------------------

    valid_scores = []

    for score in scores:
        try:
            value = float(
                score
            )

            if (
                np.isfinite(value)
                and value >= 0
            ):
                valid_scores.append(
                    value
                )

        except (
            ValueError,
            TypeError,
        ):
            continue

    if valid_scores:
        average_confidence = (
            sum(valid_scores)
            / len(valid_scores)
        )

        # PaddleOCR normally returns 0-1.
        if average_confidence <= 1.0:
            average_confidence *= 100.0

    else:
        average_confidence = 0.0

    # -----------------------------------------------------
    # Clamp confidence to frontend-friendly 0-100.
    # -----------------------------------------------------

    average_confidence = max(
        0.0,
        min(
            100.0,
            average_confidence,
        ),
    )

    return OCRResult(
        text=final_text,
        confidence=round(
            average_confidence,
            2,
        ),
    )


# =========================================================
# EXTRACT TEXT
# =========================================================

def extract_text(
    image_bytes: bytes,
) -> OCRResult:
    """
    Full processing pipeline for ONE image.

    Pipeline:

        bytes
          ↓
        OpenCV decode
          ↓
        downscale if necessary
          ↓
        ONE PaddleOCR inference
          ↓
        parse
          ↓
        sort
          ↓
        clean
    """

    # =====================================================
    # LOAD
    # =====================================================

    image = load_image(
        image_bytes
    )

    height, width = (
        image.shape[:2]
    )

    print(
        "[OCR] Original image:",
        f"{width}x{height}",
    )

    # =====================================================
    # LIGHT PREPROCESSING
    # =====================================================

    image = prepare_image(
        image
    )

    # =====================================================
    # ONE OCR PASS
    # =====================================================

    print(
        "[OCR] Single-pass OCR"
    )

    result = run_paddle_ocr(
        image
    )

    # =====================================================
    # FINAL LOG
    # =====================================================

    print(
        "[OCR] Final confidence:",
        result.confidence,
    )

    print(
        "[OCR] Final text length:",
        len(result.text),
    )

    return result


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def process_image(
    image_bytes: bytes,
) -> dict:
    """
    Public synchronous API used by FastAPI.

    main.py executes this through asyncio.to_thread(),
    so CPU-heavy OCR does not block FastAPI's event loop.
    """

    try:
        result = extract_text(
            image_bytes
        )

        return {
            "text": result.text,
            "confidence": result.confidence,
        }

    except Exception as exc:
        print(
            "[OCR] Processing error:"
        )

        print(
            repr(exc)
        )

        # -------------------------------------------------
        # Safe failure.
        #
        # The API remains alive and receives a predictable
        # empty OCR result.
        # -------------------------------------------------

        return {
            "text": "",
            "confidence": 0.0,
        }
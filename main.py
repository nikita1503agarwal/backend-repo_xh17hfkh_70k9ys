import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import io
from PIL import Image
import numpy as np
import cv2

from database import create_document, get_documents
from schemas import BodyProfile, Recommendation

app = FastAPI(title="AI Size Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    message: str


@app.get("/", response_model=HealthResponse)
def read_root():
    return {"message": "AI Size Recommender Backend Running"}


@app.get("/privacy")
def privacy_summary():
    return {
        "purpose": "Body size estimation from user-provided images to recommend clothing sizes.",
        "storage": "Images are processed in-memory and not stored. Derived measurements may be stored with consent.",
        "lawful_basis": "Consent",
        "gdpr": {
            "data_minimization": True,
            "storage_limitation": True,
            "privacy_by_default": True,
            "dpo_contact": "privacy@example.com",
        },
        "ccpa": {
            "sale_of_data": False,
            "opt_out": True,
        },
    }


def _estimate_scale_from_reference(height_cm: Optional[float], img_h: int) -> Optional[float]:
    if not height_cm:
        return None
    # naive assumption: full-body fits height of image; scale cm per pixel
    return height_cm / max(1, img_h)


def _edge_based_width(mask: np.ndarray, axis=1) -> int:
    # compute max width across rows in mask
    widths = []
    for r in range(mask.shape[0]):
        row = mask[r, :]
        cols = np.where(row > 0)[0]
        if cols.size:
            widths.append(cols.max() - cols.min())
    return int(np.median(widths)) if widths else 0


def _simple_silhouette_mask(img_bgr: np.ndarray) -> np.ndarray:
    # convert to HSV and separate person with simple threshold + morphology as fallback to real segmentation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Canny edges + close to approximate silhouette
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=2)
    mask = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def _estimate_measurements(img: Image.Image, height_cm: Optional[float]) -> BodyProfile:
    img = img.convert("RGB")
    img_np = np.array(img)[:, :, ::-1]  # RGB->BGR for OpenCV
    h, w, _ = img_np.shape

    mask = _simple_silhouette_mask(img_np)

    # Heuristic landmarks along body: chest ~ 20% from top, waist ~ 35%, hips ~ 50%, inseam approx from 65%
    chest_row = int(0.20 * h)
    waist_row = int(0.35 * h)
    hips_row = int(0.50 * h)

    chest_px = _edge_based_width(mask[:chest_row+5, :][chest_row:chest_row+1, :]) or _edge_based_width(mask[ chest_row:chest_row+1, :])
    waist_px = _edge_based_width(mask[ waist_row:waist_row+1, :])
    hips_px  = _edge_based_width(mask[  hips_row:hips_row+1, :])

    # Fallback: use overall median width if rows failed
    if not any([chest_px, waist_px, hips_px]):
        chest_px = waist_px = hips_px = _edge_based_width(mask)

    cm_per_px = _estimate_scale_from_reference(height_cm, h) or 0.5  # naive fallback scale

    chest_cm = max(0.0, float(chest_px * cm_per_px * 1.1))
    waist_cm = max(0.0, float(waist_px * cm_per_px * 1.1))
    hips_cm  = max(0.0, float(hips_px  * cm_per_px * 1.1))

    return BodyProfile(
        height_cm=height_cm,
        chest_cm=round(chest_cm, 1) if chest_cm else None,
        waist_cm=round(waist_cm, 1) if waist_cm else None,
        hips_cm=round(hips_cm, 1) if hips_cm else None,
        computed_from_image=True,
    )


DEFAULT_CHARTS = {
    "generic": {
        "tshirt": {
            "XS": {"chest_cm": (80, 86)},
            "S":  {"chest_cm": (86, 92)},
            "M":  {"chest_cm": (92, 98)},
            "L":  {"chest_cm": (98, 104)},
            "XL": {"chest_cm": (104, 110)},
            "XXL": {"chest_cm": (110, 118)},
        },
        "pants": {
            "28": {"waist_cm": (71, 74)},
            "30": {"waist_cm": (76, 79)},
            "32": {"waist_cm": (81, 84)},
            "34": {"waist_cm": (86, 89)},
            "36": {"waist_cm": (91, 94)},
            "38": {"waist_cm": (96, 99)},
        },
    }
}


def _match_size(category: str, profile: BodyProfile, brand: str = "generic") -> Recommendation:
    charts = DEFAULT_CHARTS.get(brand, DEFAULT_CHARTS["generic"]).get(category)
    if not charts:
        raise HTTPException(status_code=400, detail="No size chart for category")

    metric = None
    value = None
    if category in ("tshirt", "shirt", "hoodie", "jacket"):
        metric = "chest_cm"
        value = profile.chest_cm
    elif category in ("pants", "jeans", "skirt"):
        metric = "waist_cm"
        value = profile.waist_cm
    elif category == "dress":
        metric = "hips_cm"
        value = profile.hips_cm

    if value is None:
        raise HTTPException(status_code=400, detail="Insufficient measurements to recommend size")

    best = None
    for size_label, bounds in charts.items():
        lo, hi = list(bounds.values())[0]
        if lo <= value <= hi:
            best = size_label
            break

    if best is None:
        # choose nearest
        diffs = []
        for size_label, bounds in charts.items():
            lo, hi = list(bounds.values())[0]
            center = (lo + hi) / 2
            diffs.append((abs(value - center), size_label))
        diffs.sort()
        best = diffs[0][1]

    # naive confidence based on distance to center
    lo, hi = list(charts[best].values())[0]
    center = (lo + hi) / 2
    span = (hi - lo) / 2
    confidence = max(0.4, 1 - min(1.0, abs(value - center) / (span + 1e-6)))

    return Recommendation(
        brand=brand,
        category=category,
        suggested_size=best,
        confidence=round(float(confidence), 2),
        details={"metric": metric, "value_cm": value, "range_cm": [lo, hi]},
    )


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    height_cm: Optional[float] = Form(None),
    brand: str = Form("generic"),
    category: str = Form("tshirt"),
    consent: bool = Form(False),
):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    profile = _estimate_measurements(img, height_cm)

    rec = _match_size(category=category, profile=profile, brand=brand)

    # Privacy by default: don't store images; optionally store derived measurements and recommendation with consent
    saved_id = None
    if consent:
        try:
            doc = profile.model_dump()
            doc.update(rec.model_dump())
            saved_id = create_document("recommendation", doc)
        except Exception:
            saved_id = None

    return {
        "profile": profile.model_dump(),
        "recommendation": rec.model_dump(),
        "saved_id": saved_id,
        "privacy": {
            "image_stored": False,
            "derived_data_stored": bool(consent and saved_id),
        },
    }


@app.get("/schema")
def get_schema():
    # expose pydantic models for DB viewer
    return {
        "collections": [
            "user",
            "bodyprofile",
            "garment",
            "recommendation",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""FastAPI bridge for the evidence-grounded story pipeline."""

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components.pipeline_runner import run_pipeline
from src.config.settings import get_settings

app = FastAPI(title="Tamil Cultural Story Generator API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    config = get_settings()
    return {"status": "ok", "mode": "live" if config.is_live_mode else "mock"}


@app.post("/api/generate")
async def generate_story(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    target_words: int = Form(400),
    reading_level: str = Form("general"),
) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a valid image file.")
    if not 1 <= top_k <= 10:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 10.")
    if not 100 <= target_words <= 800:
        raise HTTPException(status_code=422, detail="target_words must be between 100 and 800.")
    if reading_level not in {"general", "simple", "academic"}:
        raise HTTPException(status_code=422, detail="Unsupported reading level.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="The uploaded file is not a readable image.") from exc

    try:
        result = run_pipeline(
            pil_image=pil_image,
            config=get_settings(),
            top_k=top_k,
            target_words=target_words,
            reading_level=reading_level,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return {
        "observation": result["observation"].model_dump(mode="json"),
        "retrieval": result["retrieval"].model_dump(mode="json"),
        "trace": result["trace"].model_dump(mode="json"),
        "story": result["story"].model_dump(mode="json"),
        "consistency": result["consistency"].model_dump(mode="json"),
        "baselines": {
            key: value.model_dump(mode="json")
            for key, value in result.get("baselines", {}).items()
        },
    }

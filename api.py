"""
Neuro Fusion-RAG — FastAPI service
------------------------------------
A plain REST API over the same text / speech / text+speech models used in
app.py (Gradio UI), meant for programmatic integration -- e.g. a teammate's
MRI-branch pipeline can call these endpoints and combine the results with
their own model's output.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health
    POST /predict/text          (JSON: {"transcript": "..."})
    POST /predict/speech        (multipart: audio file)
    POST /predict/text-speech   (multipart: transcript field + audio file, either optional)

All prediction endpoints return JSON like:
    {"label": "Control (no dementia signal)", "probability": 0.1234}

See README.md for the known limitation: live text/speech embeddings may not
exactly match the original training pipeline (the exact pooling/tokenization
code wasn't included in the project files), so predictions are best-effort.
"""

import os
import shutil
import tempfile
from typing import Optional

import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.inference import predict_text, predict_speech, predict_text_speech
from ui import demo as gradio_demo

app = FastAPI(
    title="Neuro Fusion-RAG — Speech & Text API",
    description="Text-only, speech-only, and combined dementia-signal prediction endpoints.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


class TextRequest(BaseModel):
    transcript: str


class PredictionResponse(BaseModel):
    label: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/text", response_model=PredictionResponse)
def predict_text_endpoint(payload: TextRequest):
    if not payload.transcript or not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is required.")
    return predict_text(payload.transcript)


@app.post("/predict/speech", response_model=PredictionResponse)
async def predict_speech_endpoint(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_speech(tmp_path)
    finally:
        os.remove(tmp_path)

    return result


@app.post("/predict/text-speech")
async def predict_text_speech_endpoint(
    transcript: str = Form(default=""),
    audio: Optional[UploadFile] = File(default=None),
):
    tmp_path = None
    try:
        if audio is not None:
            suffix = os.path.splitext(audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(audio.file, tmp)
                tmp_path = tmp.name

        try:
            return predict_text_speech(transcript, tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path is not None:
            os.remove(tmp_path)


# Mount the Gradio UI at /ui, so ONE process serves both:
#   /docs                -> Swagger UI (API docs)
#   /predict/text, etc.  -> REST API endpoints
#   /ui                  -> Gradio interface
# This keeps everything on a single Railway service (useful on the free plan,
# which limits how many services you can run).
app = gr.mount_gradio_app(app, gradio_demo, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

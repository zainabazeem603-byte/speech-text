"""
Shared model-loading and prediction logic for the Neuro Fusion-RAG
Speech & Text Branch demo. Used by both:
  - app.py  (Gradio UI)
  - api.py  (FastAPI service, for programmatic / other-teammate integration)

See app.py's module docstring for background on the known limitations
(embedding pooling mismatch, missing acoustic-feature extractor, etc).
"""
import torch
import librosa
from torch.quantization import quantize_dynamic
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2Processor, Wav2Vec2Model,
)

from src.model import UnimodalClassifier, MultimodalSiameseNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "Control (no dementia signal)", 1: "Dementia signal detected"}

# ablation.py evaluated/reported accuracy (92.14% text, 80.71% speech, 91.43%
# fusion) using decision threshold 0.4, not the sigmoid default of 0.5.
DECISION_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Load normalization stats (computed on training split only)
# ---------------------------------------------------------------------------
norm_stats = torch.load("preprocessing_config/norm_stats.pt", map_location="cpu", weights_only=False)


def normalize(x, mean_key, std_key):
    mean = norm_stats[mean_key]
    std = norm_stats[std_key]
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Load classifiers (loaded once at import time)
# ---------------------------------------------------------------------------
text_model = UnimodalClassifier(input_dim=768).to(DEVICE)
text_model.load_state_dict(torch.load("models/best_model_text.pth", map_location=DEVICE))
text_model.eval()

speech_model = UnimodalClassifier(input_dim=768).to(DEVICE)
speech_model.load_state_dict(torch.load("models/best_model_speech.pth", map_location=DEVICE))
speech_model.eval()

# Real trained fusion model (text + speech + acoustic), reported 91.43% accuracy.
fusion_model = MultimodalSiameseNetwork().to(DEVICE)
fusion_model.load_state_dict(torch.load("models/best_model_fusion.pth", map_location=DEVICE))
fusion_model.eval()

# ---------------------------------------------------------------------------
# Load embedding backbones
# ---------------------------------------------------------------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()
bert_model = quantize_dynamic(bert_model, {torch.nn.Linear}, dtype=torch.qint8)

wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
wav2vec_model.eval()
wav2vec_model = quantize_dynamic(wav2vec_model, {torch.nn.Linear}, dtype=torch.qint8)


def get_text_embedding(transcript: str, pooling_method: str = "mean"):
    """Returns the normalized 768-dim text embedding tensor (on DEVICE, batch dim=1)."""
    with torch.no_grad():
        tokens = bert_tokenizer(
            transcript, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(DEVICE)
        outputs = bert_model(**tokens)
        if pooling_method == "cls":
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        elif pooling_method == "pooler":
            embedding = outputs.pooler_output.squeeze(0).cpu()
        else:
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        return normalize(embedding, "text_mean", "text_std").unsqueeze(0).to(DEVICE)


def get_speech_embedding(audio_path: str):
    """Returns the normalized 768-dim speech embedding tensor (on DEVICE, batch dim=1)."""
    with torch.no_grad():
        speech_array, _ = librosa.load(audio_path, sr=16000)
        inputs = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        hidden = wav2vec_model(**inputs).last_hidden_state
        embedding = hidden.mean(dim=1).squeeze(0).cpu()
        return normalize(embedding, "speech_mean", "speech_std").unsqueeze(0).to(DEVICE)


def predict_text(transcript: str) -> dict:
    """Returns {'label': str, 'probability': float} for a transcript."""
    with torch.no_grad():
        embedding = get_text_embedding(transcript, pooling_method="mean")
        logit = text_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return {"label": LABELS[pred], "probability": round(prob, 4)}


def predict_speech(audio_path: str) -> dict:
    """Returns {'label': str, 'probability': float} for an audio file path."""
    with torch.no_grad():
        embedding = get_speech_embedding(audio_path)
        logit = speech_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return {"label": LABELS[pred], "probability": round(prob, 4)}


def predict_text_speech(transcript: str, audio_path: str) -> dict:
    """
    Returns a combined result by simple-averaging the text-only and
    speech-only probabilities. NOT the real trained fusion model.
    {'label': str, 'probability': float, 'text': {...}, 'speech': {...}}
    """
    text_result = predict_text(transcript) if transcript and transcript.strip() else None
    speech_result = predict_speech(audio_path) if audio_path else None

    if text_result is None and speech_result is None:
        raise ValueError("Provide a transcript and/or an audio file.")
    if text_result is None:
        return {**speech_result, "text": None, "speech": speech_result}
    if speech_result is None:
        return {**text_result, "text": text_result, "speech": None}

    combined_prob = round((text_result["probability"] + speech_result["probability"]) / 2, 4)
    combined_pred = int(combined_prob > DECISION_THRESHOLD)

    return {
        "label": LABELS[combined_pred],
        "probability": combined_prob,
        "text": text_result,
        "speech": speech_result,
    }


def predict_fusion(transcript: str, audio_path: str, acoustic_pt_path: str) -> dict:
    """
    Runs the REAL trained fusion model (best_model_fusion.pth). Requires a
    precomputed 47-dim acoustic-features .pt file alongside transcript+audio,
    since this project has no code to extract acoustic features from raw
    audio. Returns {'label': str, 'probability': float}.
    """
    acoustic_raw = torch.load(acoustic_pt_path, map_location="cpu", weights_only=False)
    if isinstance(acoustic_raw, dict) and "acoustic_features" in acoustic_raw:
        acoustic_raw = acoustic_raw["acoustic_features"]
    acoustic_tensor = torch.as_tensor(acoustic_raw).float().reshape(-1)
    if acoustic_tensor.numel() != 47:
        raise ValueError(
            f"Acoustic file has {acoustic_tensor.numel()} values, expected 47."
        )

    with torch.no_grad():
        text_embedding = get_text_embedding(transcript)
        speech_embedding = get_speech_embedding(audio_path)
        acoustic_embedding = normalize(
            acoustic_tensor, "acoustic_mean", "acoustic_std"
        ).unsqueeze(0).to(DEVICE)

        logit = fusion_model(acoustic_embedding, speech_embedding, text_embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return {"label": LABELS[pred], "probability": round(prob, 4)}

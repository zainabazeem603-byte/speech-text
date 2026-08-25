"""
Neuro Fusion-RAG — Speech & Text Branch Demo
----------------------------------------------
Live demo for the text-only, speech-only, and fused ablation models trained on
the Pitt Corpus (Cookie Theft / Verbal Fluency / Sentence Construction tasks).

  - Text branch   : BERT (bert-base-uncased), mean-pooled last hidden state
  - Speech branch : Wav2Vec2 (facebook/wav2vec2-base-960h), mean-pooled over time
  - Fusion branch : the ORIGINAL trained MultimodalSiameseNetwork
                     (best_model_fusion.pth, reported 91.43% accuracy),
                     which needs text (768-d) + speech (768-d) + acoustic (47-d)
                     features together.

IMPORTANT — acoustic features: the code that extracts the 47-dim acoustic
feature vector from a raw audio file was not included in the project files
(only the trained norm stats and the trained model checkpoint were). So the
"Text + Speech" tab CANNOT call the real fusion model directly from raw
audio — instead:
  - By default it falls back to a simple average of the independently-trained
    text-only and speech-only models (an approximation, not the real fusion
    model).
  - If you separately have a precomputed acoustic-features `.pt` file for a
    sample (a 47-length vector, e.g. produced by whatever tool built the
    original dataset), you can upload it in the "Advanced" section to run the
    real trained fusion model end-to-end.

Because the exact pooling/tokenization used to build the original training
text/speech embeddings isn't recorded in the project files either, live
text/speech predictions are best-effort — accuracy may differ from the
reported test-set numbers (92.14% text-only, 80.71% speech-only) if the
original pipeline pooled differently (e.g. CLS token instead of mean pooling).
"""

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2Processor, Wav2Vec2Model,
)
import librosa
import numpy as np

from src.model import UnimodalClassifier, MultimodalSiameseNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load normalization stats (computed on training split only)
# ---------------------------------------------------------------------------
norm_stats = torch.load("preprocessing_config/norm_stats.pt", map_location="cpu", weights_only=False)


def normalize(x, mean_key, std_key):
    mean = norm_stats[mean_key]
    std = norm_stats[std_key]
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Load classifiers
# ---------------------------------------------------------------------------
text_model = UnimodalClassifier(input_dim=768).to(DEVICE)
text_model.load_state_dict(torch.load("models/best_model_text.pth", map_location=DEVICE))
text_model.eval()

speech_model = UnimodalClassifier(input_dim=768).to(DEVICE)
speech_model.load_state_dict(torch.load("models/best_model_speech.pth", map_location=DEVICE))
speech_model.eval()

# Real trained fusion model (text + speech + acoustic), reported 91.43% accuracy.
# Needs a 47-dim acoustic feature vector we can't compute live from raw audio
# (see module docstring) -- only usable via the "Advanced" acoustic upload.
fusion_model = MultimodalSiameseNetwork().to(DEVICE)
fusion_model.load_state_dict(torch.load("models/best_model_fusion.pth", map_location=DEVICE))
fusion_model.eval()

# ---------------------------------------------------------------------------
# Load embedding backbones (lazy globals, loaded once at startup)
# ---------------------------------------------------------------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()

wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
wav2vec_model.eval()


LABELS = {0: "Control (no dementia signal)", 1: "Dementia signal detected"}

# The training/ablation script (ablation.py) evaluated and reported the
# checkpoints' accuracy (92.14% text, 80.71% speech, 91.43% fusion) using
# decision threshold 0.4, NOT the default 0.5:
#   def evaluate(model, loader, modality, device, threshold=0.4): ...
# Using 0.5 here would silently disagree with how these models were tuned
# and evaluated, so we match that threshold at inference time too.
DECISION_THRESHOLD = 0.4


def get_text_embedding(transcript: str, pooling_method: str = "mean"):
    """
    Returns the normalized 768-dim text embedding tensor (on DEVICE, batch dim=1).
    pooling_method:
      - "mean"   : mean-pool the last hidden state over tokens
      - "cls"    : raw [CLS] token from the last hidden state
      - "pooler" : BERT's pooler_output (CLS token passed through a dense+tanh
                   layer) -- what many pipelines mean by "the BERT embedding"
    """
    with torch.no_grad():
        tokens = bert_tokenizer(
            transcript, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(DEVICE)
        outputs = bert_model(**tokens)
        if pooling_method == "cls":
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # raw [CLS] -> [768]
        elif pooling_method == "pooler":
            embedding = outputs.pooler_output.squeeze(0).cpu()  # dense+tanh over CLS -> [768]
        else:
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()  # mean-pooled -> [768]
        return normalize(embedding, "text_mean", "text_std").unsqueeze(0).to(DEVICE)


def get_speech_embedding(audio_path: str):
    """Returns the normalized 768-dim speech embedding tensor (on DEVICE, batch dim=1)."""
    with torch.no_grad():
        speech_array, _ = librosa.load(audio_path, sr=16000)
        inputs = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        hidden = wav2vec_model(**inputs).last_hidden_state  # [1, T, 768]
        embedding = hidden.mean(dim=1).squeeze(0).cpu()  # mean-pooled -> [768]
        return normalize(embedding, "speech_mean", "speech_std").unsqueeze(0).to(DEVICE)


def predict_text(transcript: str):
    if not transcript or not transcript.strip():
        return "Please enter a transcript.", None

    with torch.no_grad():
        embedding = get_text_embedding(transcript, pooling_method="mean")
        logit = text_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return LABELS[pred], round(prob, 4)


def predict_speech(audio_path: str):
    if audio_path is None:
        return "Please upload or record audio.", None

    with torch.no_grad():
        embedding = get_speech_embedding(audio_path)
        logit = speech_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return LABELS[pred], round(prob, 4)


def predict_fusion(transcript: str, audio_path: str, acoustic_pt_path: str):
    """
    Runs the REAL trained fusion model (best_model_fusion.pth). Requires all
    three modalities: transcript, audio, AND a precomputed acoustic-features
    .pt file (a 47-length vector, or a dict/sample containing
    'acoustic_features'), since we have no code to extract acoustic features
    from raw audio ourselves.
    """
    if not transcript or not transcript.strip():
        return "Please enter a transcript.", None
    if audio_path is None:
        return "Please upload or record audio.", None
    if acoustic_pt_path is None:
        return "Please upload a precomputed acoustic-features .pt file.", None

    try:
        acoustic_raw = torch.load(acoustic_pt_path, map_location="cpu", weights_only=False)
        if isinstance(acoustic_raw, dict) and "acoustic_features" in acoustic_raw:
            acoustic_raw = acoustic_raw["acoustic_features"]
        acoustic_tensor = torch.as_tensor(acoustic_raw).float().reshape(-1)
        if acoustic_tensor.numel() != 47:
            return (
                f"Acoustic file has {acoustic_tensor.numel()} values, expected 47. "
                "Please upload the correct precomputed acoustic-features vector.",
                None,
            )
    except Exception as e:
        return f"Could not read acoustic .pt file: {e}", None

    with torch.no_grad():
        text_embedding = get_text_embedding(transcript)
        speech_embedding = get_speech_embedding(audio_path)
        acoustic_embedding = normalize(
            acoustic_tensor, "acoustic_mean", "acoustic_std"
        ).unsqueeze(0).to(DEVICE)

        logit = fusion_model(acoustic_embedding, speech_embedding, text_embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return LABELS[pred], round(prob, 4)


def predict_text_speech(transcript: str, audio_path: str):
    if (not transcript or not transcript.strip()) and audio_path is None:
        return "Please enter a transcript and/or upload audio.", None, "", ""

    text_label_str, text_prob_val = (None, None)
    speech_label_str, speech_prob_val = (None, None)

    if transcript and transcript.strip():
        text_label_str, text_prob_val = predict_text(transcript)

    if audio_path is not None:
        speech_label_str, speech_prob_val = predict_speech(audio_path)

    # If only one modality was provided, fall back to that single result
    if text_prob_val is None:
        return speech_label_str, speech_prob_val, "(no transcript provided)", f"{speech_label_str} ({speech_prob_val})"
    if speech_prob_val is None:
        return text_label_str, text_prob_val, f"{text_label_str} ({text_prob_val})", "(no audio provided)"

    # NOTE: This is a simple average ensemble of the two independently-trained
    # unimodal models, NOT the original trained fusion (text+speech+acoustic)
    # model — that checkpoint isn't available (see README). Treat this as an
    # approximation only.
    combined_prob = round((text_prob_val + speech_prob_val) / 2, 4)
    combined_pred = int(combined_prob > DECISION_THRESHOLD)

    return (
        LABELS[combined_pred],
        combined_prob,
        f"{text_label_str} ({text_prob_val})",
        f"{speech_label_str} ({speech_prob_val})",
    )


with gr.Blocks(title="Neuro Fusion-RAG — Speech & Text Branch") as demo:
    gr.Markdown(
        "# Neuro Fusion-RAG — Speech & Text Branch Demo\n"
        "Text-only and speech-only ablation models from the Pitt Corpus study, plus "
        "a simple-average ensemble of the two ('Text + Speech' tab). The **real "
        "trained fusion model** (text+speech+acoustic, 91.43% reported accuracy) is "
        "also loaded and available under 'Text + Speech' → Advanced, for samples "
        "where you have precomputed acoustic features — see model card below."
    )

    with gr.Tab("Text"):
        transcript_in = gr.Textbox(
            label="Transcript",
            placeholder="Paste a Cookie Theft description or other speech transcript...",
            lines=6,
        )
        text_btn = gr.Button("Analyze Text")
        text_label = gr.Textbox(label="Prediction")
        text_prob = gr.Number(label="Dementia probability (0-1)")
        text_btn.click(
            predict_text,
            inputs=transcript_in,
            outputs=[text_label, text_prob],
        )

    with gr.Tab("Speech"):
        audio_in = gr.Audio(label="Upload or record audio", type="filepath")
        speech_btn = gr.Button("Analyze Audio")
        speech_label = gr.Textbox(label="Prediction")
        speech_prob = gr.Number(label="Dementia probability (0-1)")
        speech_btn.click(predict_speech, inputs=audio_in, outputs=[speech_label, speech_prob])

    with gr.Tab("Text + Speech"):
        gr.Markdown(
            "**Note:** by default this combines the independently-trained "
            "text-only and speech-only models by simple average — it is *not* "
            "the original trained fusion model. Provide either or both inputs."
        )
        ts_transcript_in = gr.Textbox(
            label="Transcript (optional)",
            placeholder="Paste a Cookie Theft description or other speech transcript...",
            lines=6,
        )
        ts_audio_in = gr.Audio(label="Upload or record audio (optional)", type="filepath")
        ts_btn = gr.Button("Analyze Text + Speech (average)")
        ts_label = gr.Textbox(label="Combined prediction")
        ts_prob = gr.Number(label="Combined dementia probability (0-1, simple average)")
        ts_text_detail = gr.Textbox(label="Text branch result")
        ts_speech_detail = gr.Textbox(label="Speech branch result")
        ts_btn.click(
            predict_text_speech,
            inputs=[ts_transcript_in, ts_audio_in],
            outputs=[ts_label, ts_prob, ts_text_detail, ts_speech_detail],
        )

        with gr.Accordion("Advanced: run the REAL trained fusion model (91.43%)", open=False):
            gr.Markdown(
                "The original trained fusion checkpoint (`best_model_fusion.pth`) "
                "**is** included and loaded in this app. It needs a 47-dim acoustic "
                "feature vector alongside the transcript and audio, but this app "
                "has no code to extract that vector from raw audio (that "
                "extraction step wasn't part of the project files). If you have a "
                "precomputed acoustic-features `.pt` file for this sample (e.g. "
                "from the original dataset-prep pipeline, containing a 47-length "
                "vector or a `{'acoustic_features': ...}` dict), upload it below "
                "to get the real fusion model's prediction."
            )
            ts_acoustic_in = gr.File(label="Acoustic features .pt file (47-dim)", type="filepath")
            ts_fusion_btn = gr.Button("Run REAL Fusion Model")
            ts_fusion_label = gr.Textbox(label="Fusion model prediction")
            ts_fusion_prob = gr.Number(label="Fusion model dementia probability (0-1)")
            ts_fusion_btn.click(
                predict_fusion,
                inputs=[ts_transcript_in, ts_audio_in, ts_acoustic_in],
                outputs=[ts_fusion_label, ts_fusion_prob],
            )

    gr.Markdown(
        "### Model card\n"
        "- Trained on the Pitt Corpus (Cookie Theft, Verbal Fluency, Sentence Construction tasks), "
        "929 samples, evaluated with 4-fold cross-validation plus external ADReSS/Addresso validation.\n"
        "- Reported test accuracy: **92.14%** (text-only), **80.71%** (speech-only), **91.43%** (fusion).\n"
        "- The **Text + Speech** tab's default button combines the text-only and speech-only "
        "models' predictions by simple averaging.\n"
        "- The full trained fusion model is available under the 'Advanced' section of the "
        "Text + Speech tab for samples with precomputed acoustic features.\n"
        "- **Not a medical device.** Research demo only — not for clinical use."
    )

demo.launch()
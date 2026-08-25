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

NOTE ON MEMORY: all heavy models (BERT, Wav2Vec2, and the three trained
checkpoints) are LAZY-LOADED — only loaded into memory the first time
they're actually needed, not all at once at startup. This matters on
low-RAM hosts (e.g. Railway free tier, 1GB limit).
"""

import os
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

LABELS = {0: "Control (no dementia signal)", 1: "Dementia signal detected"}
DECISION_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Lazy-loaded globals — nothing heavy is loaded at import/startup time.
# ---------------------------------------------------------------------------
_norm_stats = None
_text_model = None
_speech_model = None
_fusion_model = None
_bert_tokenizer = None
_bert_model = None
_wav2vec_processor = None
_wav2vec_model = None


def get_norm_stats():
    global _norm_stats
    if _norm_stats is None:
        _norm_stats = torch.load(
            "preprocessing_config/norm_stats.pt", map_location="cpu", weights_only=False
        )
    return _norm_stats


def normalize(x, mean_key, std_key):
    stats = get_norm_stats()
    mean = stats[mean_key]
    std = stats[std_key]
    return (x - mean) / std


def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = UnimodalClassifier(input_dim=768).to(DEVICE)
        _text_model.load_state_dict(torch.load("models/best_model_text.pth", map_location=DEVICE))
        _text_model.eval()
    return _text_model


def get_speech_model():
    global _speech_model
    if _speech_model is None:
        _speech_model = UnimodalClassifier(input_dim=768).to(DEVICE)
        _speech_model.load_state_dict(torch.load("models/best_model_speech.pth", map_location=DEVICE))
        _speech_model.eval()
    return _speech_model


def get_fusion_model():
    global _fusion_model
    if _fusion_model is None:
        _fusion_model = MultimodalSiameseNetwork().to(DEVICE)
        _fusion_model.load_state_dict(torch.load("models/best_model_fusion.pth", map_location=DEVICE))
        _fusion_model.eval()
    return _fusion_model


def get_bert():
    global _bert_tokenizer, _bert_model
    if _bert_model is None:
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


def get_wav2vec():
    global _wav2vec_processor, _wav2vec_model
    if _wav2vec_model is None:
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
        _wav2vec_model.eval()
    return _wav2vec_processor, _wav2vec_model


def get_text_embedding(transcript: str, pooling_method: str = "mean"):
    bert_tokenizer, bert_model = get_bert()
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
    wav2vec_processor, wav2vec_model = get_wav2vec()
    with torch.no_grad():
        speech_array, _ = librosa.load(audio_path, sr=16000)
        inputs = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        hidden = wav2vec_model(**inputs).last_hidden_state
        embedding = hidden.mean(dim=1).squeeze(0).cpu()
        return normalize(embedding, "speech_mean", "speech_std").unsqueeze(0).to(DEVICE)


def predict_text(transcript: str):
    if not transcript or not transcript.strip():
        return "Please enter a transcript.", None

    with torch.no_grad():
        embedding = get_text_embedding(transcript, pooling_method="mean")
        logit = get_text_model()(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return LABELS[pred], round(prob, 4)


def predict_speech(audio_path: str):
    if audio_path is None:
        return "Please upload or record audio.", None

    with torch.no_grad():
        embedding = get_speech_embedding(audio_path)
        logit = get_speech_model()(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > DECISION_THRESHOLD)
    return LABELS[pred], round(prob, 4)


def predict_fusion(transcript: str, audio_path: str, acoustic_pt_path: str):
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

        logit = get_fusion_model()(acoustic_embedding, speech_embedding, text_embedding)
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

    if text_prob_val is None:
        return speech_label_str, speech_prob_val, "(no transcript provided)", f"{speech_label_str} ({speech_prob_val})"
    if speech_prob_val is None:
        return text_label_str, text_prob_val, f"{text_label_str} ({text_prob_val})", "(no audio provided)"

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
        "a simple-average ensemble of the two ('Text + Speech' tab). The real "
        "trained fusion model (text+speech+acoustic, 91.43% reported accuracy) is "
        "also loaded and available under 'Text + Speech' → Advanced, for samples "
        "where you have precomputed acoustic features — see model card below.\n\n"
        "_Models load on first use, so the first prediction in each tab may take "
        "a little longer than the rest._"
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
            "text-only and speech-only models by simple average — it is not "
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
                "The original trained fusion checkpoint (best_model_fusion.pth) "
                "is included and loaded in this app. It needs a 47-dim acoustic "
                "feature vector alongside the transcript and audio, but this app "
                "has no code to extract that vector from raw audio (that "
                "extraction step wasn't part of the project files). If you have a "
                "precomputed acoustic-features .pt file for this sample (e.g. "
                "from the original dataset-prep pipeline, containing a 47-length "
                "vector or a {'acoustic_features': ...} dict), upload it below "
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
        "- Reported test accuracy: 92.14% (text-only), 80.71% (speech-only), 91.43% (fusion).\n"
        "- The Text + Speech tab's default button combines the text-only and speech-only "
        "models' predictions by simple averaging.\n"
        "- The full trained fusion model is available under the 'Advanced' section of the "
        "Text + Speech tab for samples with precomputed acoustic features.\n"
        "- Not a medical device. Research demo only — not for clinical use."
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
    )

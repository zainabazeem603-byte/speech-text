"""
Gradio UI (Blocks) for the Neuro Fusion-RAG Speech & Text Branch demo.

This module only BUILDS the `demo` Gradio Blocks object -- it does not call
.launch(). That lets it be used two ways:
  - app.py   : `from ui import demo; demo.launch(...)` for a standalone Gradio server
  - api.py   : mounts `demo` inside the FastAPI app at /ui, so one process/
               one Railway service serves both the UI and the REST API
               (useful on Railway's free plan, which limits you to one
               service).
"""

import gradio as gr

from src.inference import (
    predict_text as _predict_text_dict,
    predict_speech as _predict_speech_dict,
    predict_fusion as _predict_fusion_dict,
    predict_text_speech as _predict_text_speech_dict,
)


# -----------------------------------------------------------------------
# Thin adapters: src.inference functions return dicts (API-friendly);
# Gradio callbacks here return plain (label, probability, ...) tuples.
# -----------------------------------------------------------------------
def predict_text(transcript: str):
    if not transcript or not transcript.strip():
        return "Please enter a transcript.", None
    r = _predict_text_dict(transcript)
    return r["label"], r["probability"]


def predict_speech(audio_path: str):
    if audio_path is None:
        return "Please upload or record audio.", None
    r = _predict_speech_dict(audio_path)
    return r["label"], r["probability"]


def predict_fusion(transcript: str, audio_path: str, acoustic_pt_path: str):
    if not transcript or not transcript.strip():
        return "Please enter a transcript.", None
    if audio_path is None:
        return "Please upload or record audio.", None
    if acoustic_pt_path is None:
        return "Please upload a precomputed acoustic-features .pt file.", None
    try:
        r = _predict_fusion_dict(transcript, audio_path, acoustic_pt_path)
    except Exception as e:
        return f"Could not run fusion model: {e}", None
    return r["label"], r["probability"]


def predict_text_speech(transcript: str, audio_path: str):
    if (not transcript or not transcript.strip()) and audio_path is None:
        return "Please enter a transcript and/or upload audio.", None, "", ""
    try:
        r = _predict_text_speech_dict(transcript, audio_path)
    except ValueError as e:
        return str(e), None, "", ""

    text_detail = f"{r['text']['label']} ({r['text']['probability']})" if r.get("text") else "(no transcript provided)"
    speech_detail = f"{r['speech']['label']} ({r['speech']['probability']})" if r.get("speech") else "(no audio provided)"
    return r["label"], r["probability"], text_detail, speech_detail


# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------
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
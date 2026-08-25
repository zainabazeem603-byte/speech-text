# Neuro Fusion-RAG — Speech & Text Branch (Live Demo)

Live inference demo for the **text-only** and **speech-only** ablation classifiers
from the Neuro Fusion-RAG project — a multimodal dementia-detection system trained
on the Pitt Corpus (Cookie Theft, Verbal Fluency, and Sentence Construction tasks).

This repo is deployed as a [Gradio](https://gradio.app) app on
[Hugging Face Spaces](https://huggingface.co/spaces). See **Deploy** below to connect
this repo to a Space.

## What this demo does

- **Text tab** — paste a speech transcript, get a Control vs. Dementia prediction
  using a BERT-based embedding + trained MLP classifier.
- **Speech tab** — upload/record audio, get a Control vs. Dementia prediction using
  a Wav2Vec2-based embedding + trained MLP classifier.
- **Text + Speech tab** — combines the above two by simple average (approximation,
  not the real fusion model), plus an "Advanced" accordion that runs the **real
  trained fusion model** (`best_model_fusion.pth`, reported 91.43% accuracy) if you
  supply a precomputed 47-dim acoustic-features `.pt` file for the sample.

## What this demo does *not* include

The full Neuro Fusion-RAG pipeline also has an **MRI branch** (Swin Transformer,
4-class dementia staging) — not part of this repo.

The real fusion model's weights and architecture *are* included, but this demo
has no code to extract the 47-dim acoustic feature vector from raw audio — that
extraction step wasn't part of the project files, only pre-computed `.pt`
samples and the trained normalization stats/checkpoint were. So the fusion
model can only run end-to-end if you separately provide that acoustic vector.

## Project structure

```
.
├── app.py                        # Gradio app (entry point for HF Spaces)
├── requirements.txt              # Python dependencies
├── src/
│   └── model.py                  # Encoder / UnimodalClassifier architecture
├── models/
│   ├── best_model_text.pth       # Trained text-only classifier
│   ├── best_model_speech.pth     # Trained speech-only classifier
│   └── best_model_fusion.pth     # Trained fusion (text+speech+acoustic) model
└── preprocessing_config/
    └── norm_stats.pt             # Train-set mean/std for z-score normalization
```

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

The first run downloads `bert-base-uncased` and `facebook/wav2vec2-base-960h`
from the Hugging Face Hub (a few hundred MB combined) — this needs an internet
connection the first time.

## Deploy on Hugging Face Spaces (recommended)

1. Push this repo to GitHub.
2. On [huggingface.co](https://huggingface.co), create a **New Space** →
   SDK: **Gradio** → Hardware: **CPU basic** (free).
3. Choose **"Create Space from GitHub"** (or link the repo under Space settings →
   Repository) so the Space stays in sync with this repo — every push rebuilds
   the live demo automatically.
4. Wait for the build to finish (first build installs `torch` + `transformers`,
   ~3–5 min). The Space then serves the app at a public URL.

## Model performance (reported, on held-out Pitt Corpus test split)

| Model | Accuracy | F1-score | AUC |
|---|---|---|---|
| Text-only | 92.14% | 94.93% | 96.77% |
| Speech-only | 80.71% | 86.70% | 89.64% |
| Fusion (text+speech+acoustic) | 91.43% | 94.39% | 96.80% |

## Limitations

- This demo builds text/speech embeddings on the fly (mean-pooled BERT /
  Wav2Vec2 hidden states). If the original training pipeline used a different
  pooling strategy (e.g. CLS token) or checkpoint variant, live predictions may
  not exactly match the reported test accuracy above.
- **Research demo only — not a medical device, not for clinical diagnosis.**

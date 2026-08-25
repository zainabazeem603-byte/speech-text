import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.network(x)


class UnimodalClassifier(nn.Module):
    """
    Single-modality classifier -- used for ablation study to compare
    text-only vs speech-only vs fused (multimodal) performance.
    Reuses the exact same Encoder block as the multimodal model, so the
    comparison is fair (same capacity per branch).
    """

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = Encoder(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output


class MultimodalSiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.acoustic_encoder = Encoder(47)
        self.speech_encoder = Encoder(768)
        self.text_encoder = Encoder(768)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, acoustic, speech, text):

        acoustic = self.acoustic_encoder(acoustic)
        speech = self.speech_encoder(speech)
        text = self.text_encoder(text)

        fused = torch.cat(
            [acoustic, speech, text],
            dim=1
        )

        output = self.classifier(fused)

        return output
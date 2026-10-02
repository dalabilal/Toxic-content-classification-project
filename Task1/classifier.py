"""Loads the model trained by train_LSTM_simple.py and predicts the toxic category."""

import json
import re
from pathlib import Path

import torch
import torch.nn as nn


ARTIFACT_DIR = Path(__file__).parent / "artifact"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vocab = None
cfg = None
model = None


class LSTMClassifier(nn.Module):
    # This must match train_LSTM_simple.py exactly.
    def __init__(self, vocab_size, n_classes, cfg):
        super().__init__()
        self.emb = nn.Embedding(
            vocab_size,
            cfg["emb_dim"],
            padding_idx=cfg["pad"]
        )
        self.lstm = nn.LSTM(
            cfg["emb_dim"],
            cfg["hidden"],
            batch_first=True,
            bidirectional=True
        )
        self.drop = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(cfg["hidden"] * 2, n_classes)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            e,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)

        # Last hidden state from both directions.
        h = torch.cat((h[0], h[1]), dim=1)

        return self.fc(self.drop(h))


def clean(s):
    # Same cleaning as in train_LSTM_simple.py.
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def load_classifier():
    """Load config, vocabulary and model only once."""
    global vocab, cfg, model

    if model is not None:
        return

    cfg = json.loads(
        (ARTIFACT_DIR / "config.json").read_text(encoding="utf-8")
    )

    vocab = json.loads(
        (ARTIFACT_DIR / "vocab.json").read_text(encoding="utf-8")
    )

    model = LSTMClassifier(
        len(vocab),
        len(cfg["labels"]),
        cfg
    )

    model.load_state_dict(
        torch.load(
            ARTIFACT_DIR / "lstm_model.pt",
            map_location=DEVICE
        )
    )

    model.to(DEVICE).eval()


def encode(query="", image_text=""):
    # Must use the same format as training:
    # query <SEP> image description
    q = [
        vocab.get(w, cfg["unk"])
        for w in clean(query).split()
    ][:cfg["max_q"]]

    i = [
        vocab.get(w, cfg["unk"])
        for w in clean(image_text).split()
    ][:cfg["max_i"]]

    seq = q + [cfg["sep"]] + i

    x = torch.zeros(
        1,
        cfg["seq_len"],
        dtype=torch.long
    )

    x[0, :len(seq)] = torch.tensor(seq)

    lengths = torch.tensor([len(seq)])

    return x.to(DEVICE), lengths


def predict(query="", image_text=""):
    """
    Return:
        label,
        confidence,
        {label: probability}
    """
    load_classifier()

    x, lengths = encode(query, image_text)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(x, lengths),
            dim=-1
        )[0].tolist()

    best = probabilities.index(max(probabilities))

    return (
        cfg["labels"][best],
        probabilities[best],
        dict(zip(cfg["labels"], probabilities))
    )
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report

torch.manual_seed(42)
np.random.seed(42)

# settings
EMB_DIM = 128
HIDDEN = 128
DROPOUT = 0.4
LR = 0.002
BATCH = 32
EPOCHS = 60
PATIENCE = 8

# 1. load the data
df = pd.read_csv("cellula toxic data.csv")
df = df[["query", "image descriptions", "Toxic Category"]]
df.columns = ["query", "img", "label"]
df = df.dropna()


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


df["query"] = df["query"].map(clean)
df["img"] = df["img"].map(clean)

# one row for each different (query, image, label), n = how many times it appears
pairs = df.groupby(["query", "img", "label"], as_index=False).size()
pairs = pairs.rename(columns={"size": "n"})

labels = sorted(pairs["label"].unique())
label2id = {name: i for i, name in enumerate(labels)}
pairs["y"] = pairs["label"].map(label2id)
print(pairs["label"].value_counts())

# 2. split each class into train / validation / test
train_parts, val_parts, test_parts = [], [], []
for label, group in pairs.groupby("label"):
    group = group.sample(frac=1, random_state=42)
    n_test = max(1, round(0.15 * len(group)))
    n_val = max(1, round(0.15 * len(group)))
    test_parts.append(group.iloc[:n_test])
    val_parts.append(group.iloc[n_test:n_test + n_val])
    train_parts.append(group.iloc[n_test + n_val:])

train_u = pd.concat(train_parts)
val_df = pd.concat(val_parts)
test_df = pd.concat(test_parts)

# in train only, repeat each row like the csv does (at most 30 times)
train_df = train_u.loc[train_u.index.repeat(train_u["n"].clip(upper=30))]
print("train", len(train_df), "val", len(val_df), "test", len(test_df))

# 3. vocabulary (words from train only)
counter = Counter()
for text in list(train_u["query"]) + list(train_u["img"]):
    counter.update(text.split())

vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
for word in counter:
    vocab[word] = len(vocab)

max_q = int(min(80, np.percentile(train_u["query"].str.split().str.len(), 99)))
max_i = int(train_u["img"].str.split().str.len().max())
seq_len = max_q + 1 + max_i


def encode(frame):
    # query words + <sep> + image words, padded with 0
    seqs = []
    for q, i in zip(frame["query"], frame["img"]):
        ids = [vocab.get(w, 1) for w in q.split()][:max_q]
        ids = ids + [2] + [vocab.get(w, 1) for w in i.split()][:max_i]
        seqs.append(ids)
    x = torch.zeros(len(seqs), seq_len, dtype=torch.long)
    for r, ids in enumerate(seqs):
        x[r, :len(ids)] = torch.tensor(ids)
    lens = torch.tensor([len(s) for s in seqs])
    y = torch.tensor(frame["y"].values, dtype=torch.long)
    return x, lens, y


xtr, ltr, ytr = encode(train_df)
xva, lva, yva = encode(val_df)
xte, lte, yte = encode(test_df)

# class weights so the small classes are not ignored
counts = np.maximum(np.bincount(ytr.numpy(), minlength=len(labels)), 1)
weights = torch.tensor(len(ytr) / (len(labels) * counts), dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=weights)


# 4. model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN * 2, n_classes)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat((h[0], h[1]), dim=1)   # last state of both directions
        return self.fc(self.drop(h))


model = LSTMClassifier(len(vocab), len(labels))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 5. training with early stopping
best_score = 0
best_state = None
wait = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(xtr))
    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(model(xtr[idx], ltr[idx]), ytr[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(xva, lva).argmax(1).numpy()
    score = f1_score(yva.numpy(), val_pred, average="weighted", zero_division=0)
    print(f"epoch {epoch}  loss {loss.item():.4f}  val weighted f1 {score:.4f}")

    if score > best_score:
        best_score = score
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("early stopping")
            break

# 6. test the best model
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(xte, lte).argmax(1).numpy()
y_true = yte.numpy()

present = sorted(set(y_true.tolist()))
print("\ntest accuracy:", round(accuracy_score(y_true, pred), 4))
print("test weighted f1:", round(f1_score(y_true, pred, average="weighted", zero_division=0), 4))
print(classification_report(y_true, pred, labels=present,
                            target_names=[labels[i] for i in present], zero_division=0))

# 7. save for the app
# All files needed by the classifier are stored in one folder.
ARTIFACT_DIR = Path(__file__).parent / "artifact"
ARTIFACT_DIR.mkdir(exist_ok=True)

torch.save(best_state, ARTIFACT_DIR / "lstm_model.pt")

config = {
    "labels": labels,
    "max_q": max_q,
    "max_i": max_i,
    "seq_len": seq_len,
    "pad": 0,
    "unk": 1,
    "sep": 2,
    "emb_dim": EMB_DIM,
    "hidden": HIDDEN,
    "dropout": DROPOUT
}

with open(ARTIFACT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

with open(ARTIFACT_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"saved model, config and vocab to: {ARTIFACT_DIR}")

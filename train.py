"""Train + evaluate baseline (TF-IDF+LR) and DistilBERT on the same split.
Usage: python train.py --model both [--sample 8000] [--epochs 2]"""
import argparse, json, os
import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from utils import load_isot, preprocess

MET = "models/metrics.json"


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary")
    return dict(accuracy=float(accuracy_score(y, p)), precision=float(pr), recall=float(rc),
                f1=float(f1), confusion=confusion_matrix(y, p).tolist())


def train_transformer(tr, te, name, epochs):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    class DS(torch.utils.data.Dataset):
        def __init__(s, df):
            s.enc = tok(df.text_full.tolist(), truncation=True, max_length=256, padding="max_length")
            s.y = df.label.values
        def __len__(s): return len(s.y)
        def __getitem__(s, i):
            d = {k: torch.tensor(v[i]) for k, v in s.enc.items()}
            d["labels"] = torch.tensor(int(s.y[i]))
            return d

    args = TrainingArguments("models/_tmp", num_train_epochs=epochs, per_device_train_batch_size=16,
                             learning_rate=2e-5, weight_decay=0.01, save_strategy="no",
                             fp16=torch.cuda.is_available(), report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=DS(tr))
    trainer.train()
    logits = trainer.predict(DS(te)).predictions
    model.save_pretrained("models/transformer"); tok.save_pretrained("models/transformer")
    return metrics(te.label.values, np.argmax(logits, -1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["baseline", "transformer", "both"])
    ap.add_argument("--sample", type=int, default=0, help="subsample rows for quick/CPU runs")
    ap.add_argument("--epochs", type=int, default=2)
    # multilingual checkpoint -> English training transfers (zero-shot) to Hindi
    ap.add_argument("--hf_model", default="distilbert-base-multilingual-cased")
    a = ap.parse_args()

    df = load_isot()
    if a.sample: df = df.sample(a.sample, random_state=1)
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    os.makedirs("models", exist_ok=True)
    res = json.load(open(MET)) if os.path.exists(MET) else {}

    if a.model in ("baseline", "both"):
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
        lr = LogisticRegression(max_iter=1000).fit(vec.fit_transform(tr.text_full.map(preprocess)), tr.label)
        pred = lr.predict(vec.transform(te.text_full.map(preprocess)))
        res["Baseline (TF-IDF + LR)"] = metrics(te.label, pred)
        joblib.dump({"vec": vec, "lr": lr}, "models/baseline.joblib")
        print("Baseline:", {k: round(v, 4) for k, v in res["Baseline (TF-IDF + LR)"].items() if k != "confusion"})

    if a.model in ("transformer", "both"):
        res["DistilBERT"] = train_transformer(tr, te, a.hf_model, a.epochs)
        print("DistilBERT:", {k: round(v, 4) for k, v in res["DistilBERT"].items() if k != "confusion"})

    json.dump(res, open(MET, "w"), indent=2)

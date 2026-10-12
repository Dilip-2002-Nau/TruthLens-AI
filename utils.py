"""Shared NLP preprocessing + data loading for TruthLens AI."""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

LEMM, STOP = WordNetLemmatizer(), set(stopwords.words("english"))


def is_hindi(t: str) -> bool:
    """True if >30% of non-space characters are Devanagari."""
    chars = re.sub(r"\s", "", t)
    return len(re.findall(r"[\u0900-\u097F]", t)) > 0.3 * max(1, len(chars))


def clean(t: str) -> str:
    """Light cleaning (keeps casing/punctuation for the transformer).
    Also strips the 'CITY (Reuters) -' dateline, a known label-leak in ISOT."""
    t = re.sub(r"^.{0,60}?\(Reuters\)\s*-?\s*", "", str(t))
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def preprocess(t: str) -> str:
    """Classic pipeline for the baseline: lowercase -> tokenize -> stop-words -> lemmatize."""
    t = clean(t)
    if is_hindi(t):  # NLTK English resources don't apply to Hindi
        return t
    toks = [w for w in word_tokenize(t.lower()) if w.isalpha()]
    return " ".join(LEMM.lemmatize(w) for w in toks if w not in STOP)


def load_isot(folder="data") -> pd.DataFrame:
    """ISOT dataset: data/Fake.csv + data/True.csv. Label 1 = Fake, 0 = Real."""
    fake, real = pd.read_csv(f"{folder}/Fake.csv"), pd.read_csv(f"{folder}/True.csv")
    fake["label"], real["label"] = 1, 0
    df = pd.concat([fake, real], ignore_index=True)
    df["text_full"] = (df.title.fillna("") + ". " + df.text.fillna("")).map(clean)
    df = df[df.text_full.str.len() > 30].drop_duplicates("text_full")
    return df.sample(frac=1, random_state=42)[["text_full", "label"]].reset_index(drop=True)

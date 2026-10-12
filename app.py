"""TruthLens AI - Streamlit dashboard. Run: streamlit run app.py"""
import html, json, os, re, sqlite3
from collections import Counter
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st
from utils import STOP, clean, is_hindi, preprocess

BASE, TRF = "Baseline (TF-IDF + LR)", "DistilBERT"
st.set_page_config("TruthLens AI", "🔍", layout="wide")

DB = sqlite3.connect("history.db", check_same_thread=False)
DB.execute("""CREATE TABLE IF NOT EXISTS h(ts TEXT DEFAULT CURRENT_TIMESTAMP, title TEXT,
              lang TEXT, model TEXT, label TEXT, conf REAL, summary TEXT)""")


@st.cache_resource
def load_models():
    R = {}
    if os.path.exists("models/baseline.joblib"):
        R[BASE] = joblib.load("models/baseline.joblib")
    if os.path.isdir("models/transformer"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        R[TRF] = (AutoTokenizer.from_pretrained("models/transformer"),
                  AutoModelForSequenceClassification.from_pretrained("models/transformer").eval())
    return R


R = load_models()


def p_fake(texts, name):
    """Probability that each text is FAKE."""
    if name == BASE:
        X = R[BASE]["vec"].transform([preprocess(t) for t in texts])
        return R[BASE]["lr"].predict_proba(X)[:, 1]
    import torch
    tok, model = R[TRF]
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits, -1)[:, 1].tolist()
    return np.array(out)


def explain(text, name, k=4, max_words=160):
    """Occlusion explainability: drop each k-word phrase, measure change in P(fake).
    delta > 0 -> phrase pushes toward FAKE; delta < 0 -> toward REAL."""
    w = text.split()[:max_words]
    chunks = [w[i:i + k] for i in range(0, len(w), k)]
    base = p_fake([" ".join(w)], name)[0]
    variants = [" ".join(sum(chunks[:j] + chunks[j + 1:], [])) for j in range(len(chunks))]
    return chunks, base - p_fake(variants, name)


def highlight(chunks, delta):
    m = max(np.abs(delta).max(), 1e-6)
    spans = []
    for c, d in zip(chunks, delta):
        a = min(abs(d) / m, 1) * 0.75
        rgb = "220,38,38" if d > 0 else "22,163,74"
        spans.append(f"<span style='background:rgba({rgb},{a:.2f});padding:2px 1px;border-radius:3px'>"
                     f"{html.escape(' '.join(c))}</span>")
    return " ".join(spans)


def summarize(text, n=3):
    """Extractive summary: rank sentences by average word frequency."""
    sents = [s for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
    if len(sents) <= n: return text
    tokens = lambda s: [w for w in re.findall(r"[\w\u0900-\u097F]+", s.lower()) if w not in STOP]
    freq = Counter(tokens(text))
    score = [sum(freq[w] for w in tokens(s)) / (len(s.split()) + 1) for s in sents]
    top = sorted(sorted(range(len(sents)), key=lambda i: -score[i])[:n])
    return " ".join(sents[i] for i in top)


st.title("🔍 TruthLens AI")
st.caption("Fake news detection with TF-IDF + Logistic Regression and DistilBERT · English / Hindi")
t1, t2, t3 = st.tabs(["Analyze", "Model Performance", "History"])

with t1:
    if not R:
        st.error("No trained models found. Run `python train.py --model both` first."); st.stop()
    c1, c2 = st.columns([3, 1])
    title = c1.text_input("Headline")
    model_name = c2.selectbox("Model", list(R), index=list(R).index(TRF) if TRF in R else 0)
    body = st.text_area("Article text (English or Hindi)", height=220)

    if st.button("Analyze", type="primary") and (title + body).strip():
        text = clean(f"{title}. {body}")
        hindi = is_hindi(text)
        if hindi and model_name == BASE:
            if TRF in R:
                st.info("Hindi detected - switching to the multilingual DistilBERT model."); model_name = TRF
            else:
                st.error("Baseline is English-only; train the transformer for Hindi."); st.stop()
        with st.spinner("Analyzing..."):
            p = float(p_fake([text], model_name)[0])
            label, conf = ("FAKE", p) if p >= 0.5 else ("REAL", 1 - p)
            summary = summarize(text)
            chunks, delta = explain(text, model_name)

        a, b = st.columns([1, 2])
        a.markdown(f"### {'🚨' if label == 'FAKE' else '✅'} {label}")
        a.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=conf * 100,
            number={"suffix": "%"}, title={"text": "Confidence"},
            gauge={"axis": {"range": [50, 100]}, "bar": {"color": "#dc2626" if label == "FAKE" else "#16a34a"}})
            ).update_layout(height=250, margin=dict(t=50, b=0)), use_container_width=True)
        if conf < 0.65: a.warning("Low confidence - verify with trusted sources.")
        b.subheader("Summary"); b.write(summary)
        b.caption(f"Language: {'Hindi' if hindi else 'English'} · Model: {model_name}")

        st.subheader("Why this prediction?")
        st.markdown("<small>🟥 pushes toward **Fake** · 🟩 pushes toward **Real** (darker = stronger)</small>",
                    unsafe_allow_html=True)
        st.markdown(highlight(chunks, delta), unsafe_allow_html=True)
        top = sorted(zip([" ".join(c) for c in chunks], delta), key=lambda x: -abs(x[1]))[:8]
        st.plotly_chart(px.bar(pd.DataFrame(top, columns=["phrase", "impact"]).iloc[::-1], x="impact", y="phrase",
            orientation="h", color="impact", color_continuous_scale="RdYlGn_r", title="Most influential phrases"
            ).update_layout(height=350), use_container_width=True)
        st.caption("⚠️ The model detects writing patterns learned from its training data; "
                   "it does not fact-check claims. Treat results as a signal, not a verdict.")
        DB.execute("INSERT INTO h(title,lang,model,label,conf,summary) VALUES(?,?,?,?,?,?)",
                   (title or text[:80], "hi" if hindi else "en", model_name, label, round(conf, 4), summary))
        DB.commit()

with t2:
    if not os.path.exists("models/metrics.json"):
        st.info("Run train.py to generate metrics.")
    else:
        M = json.load(open("models/metrics.json"))
        df = pd.DataFrame({k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1"]} for k, v in M.items()}).T
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
        st.plotly_chart(px.bar(df.reset_index().melt("index"), x="variable", y="value", color="index",
            barmode="group", labels={"index": "Model", "variable": "Metric", "value": "Score"},
            title="Baseline vs Transformer").update_yaxes(range=[0.8, 1.0]), use_container_width=True)
        cols = st.columns(len(M))
        for col, (k, v) in zip(cols, M.items()):
            col.plotly_chart(px.imshow(v["confusion"], text_auto=True, x=["Real", "Fake"], y=["Real", "Fake"],
                labels=dict(x="Predicted", y="Actual"), title=k, color_continuous_scale="Blues"
                ).update_layout(height=320), use_container_width=True)

with t3:
    h = pd.read_sql("SELECT * FROM h ORDER BY ts DESC", DB)
    st.dataframe(h, use_container_width=True)
    if len(h) and st.button("Clear history"):
        DB.execute("DELETE FROM h"); DB.commit(); st.rerun()

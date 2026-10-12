# TruthLens AI - Fake News Detection (NLP + Transformers)

## Setup
```bash
pip install -r requirements.txt
mkdir data   # put ISOT's Fake.csv and True.csv here (Kaggle: "Fake and real news dataset")
python train.py --model both            # full run (GPU recommended)
python train.py --model both --sample 8000 --epochs 1   # quick CPU test
streamlit run app.py
```

## Files
- `utils.py` - cleaning, tokenization, stop-words, lemmatization, Hindi detection, dataset loader
- `train.py` - trains TF-IDF+LR baseline and DistilBERT on the same split; writes `models/metrics.json`
- `app.py` - Streamlit app: prediction, confidence gauge, summary, phrase highlighting, history, metrics dashboard

## Design notes
- **Label:** 1 = Fake, 0 = Real. Stratified 80/20 split, identical for both models.
- **Leakage:** ISOT real articles start with "CITY (Reuters) -", which inflates accuracy to ~99%+.
  `clean()` strips it; expect somewhat lower, more honest scores.
- **Explainability:** occlusion - each 4-word phrase is removed and the change in P(fake) is measured.
  Works for both models, no extra library needed.
- **Hindi:** uses `distilbert-base-multilingual-cased`, fine-tuned on English only, so Hindi is zero-shot
  cross-lingual transfer. For reliable Hindi accuracy, add a labeled Hindi dataset to `load_isot()`.
  The baseline is English-only; the app auto-switches to the transformer for Hindi input.
- **Limits:** the model learns the style of its training data and does not verify facts.
  It may generalize poorly to other sources/time periods.

## Extensions
URL article fetching (newspaper3k), LIME/SHAP, model-drift monitoring, Docker deployment.

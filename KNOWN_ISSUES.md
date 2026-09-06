# Known Issues

## Train/inference feature mismatch — amount_bin, amount_ratio

**Status:** Resolved (2026-08). Fixed via option 2 below — no retraining
needed since the training data was recoverable from `data/processed/sar_dataset.csv`.

`preprocessing.py` (used at training time) and `predict.py` (used at
inference time) previously computed `amount_bin` and `amount_ratio`
differently:

| Feature | preprocessing.py (train) | predict.py (inference) |
|---|---|---|
| `amount_bin` | `pd.cut(df['amount'], bins=5)` — data-driven, equal-width bins based on the training set's min/max | `pd.cut(data['amount'], bins=[0,100,1000,10000,100000,inf])` — fixed edges |
| `amount_ratio` | `amount / amount.mean()` — training set mean | `amount / 5000` — hardcoded constant |

**Impact:** the model was trained on one feature distribution and is
served a differently-distributed version of the same named feature.
This likely degrades (rather than helps) the contribution of these two
features to the fraud probability score.

**Fix applied:** `predict.py` now uses `TRAIN_AMOUNT_BIN_EDGES` and
`TRAIN_AMOUNT_MEAN` constants derived directly from the processed
training dataset (`data/processed/sar_dataset.csv`), so `amount_bin`
and `amount_ratio` are computed identically at inference time as they
were during training. Verified by `test_amount_bin_matches_training_edges`
and `test_amount_ratio_uses_training_mean` in `test_predict.py`.

Original fix options considered:
1. Retrain the model using the fixed bins/divisor from `predict.py`, or
2. Compute the training-set mean and equal-width bin edges once, hardcode
   them into `predict.py` to match exactly what the model saw during training.

Option 2 was chosen — lower risk, no retraining required.

---

## Unseeded random.choice() in preprocessing.py

**Status:** Resolved (2026-08).

`location` and `transaction_type` columns were generated with
`random.choice()` without ever calling `random.seed()`, so even though
`df.sample(..., random_state=42)` and the SMOTE/train_test_split calls
elsewhere were reproducible, these two columns were not — rerunning
the pipeline could silently produce a different dataset.

**Fix:** added `random.seed(42)` immediately before the `random.choice()`
calls in `preprocessing.py`.

---

## Silently swallowed Groq API errors

**Status:** Resolved (2026-08).

`generate_narrative()` caught `Exception as e` when the Groq API call
failed but never logged `e` — errors vanished silently and there was
no way to tell from logs whether the LLM path was working or always
falling back to the local template.

**Fix:** the exception is now printed to stderr before falling back.

---

## Leftover "EDI" branding in SAR fallback template

**Status:** Resolved (2026-08).

The fallback SAR narrative (used when Groq isn't configured or fails)
referenced "EDI Compliance Operations Center" and report IDs like
`SAR-2026-EDI-####` — a leftover from the project's original hackathon
codename before it became Aegis. Any fraud narrative shown to a user
or reviewer displayed the wrong institution name.

**Fix:** replaced with "Aegis Compliance Operations Center" and
`SAR-2026-AEGIS-####`. Covered by
`test_fraud_narrative_uses_aegis_branding_not_edi`.

---

## README claimed unused technologies (FastAPI, React, GNN, ChromaDB, LangChain)

**Status:** Resolved (2026-08).

The README's Features, Stack, and Architecture sections described a
FastAPI backend, GNN risk scoring, ChromaDB vector storage, LangChain
orchestration, and a React frontend. None of these exist in this repo —
the actual implementation is Streamlit + XGBoost + TF-IDF retrieval +
direct Groq API calls. Anyone reading the README then opening the code
would immediately see the mismatch.

**Fix:** rewrote the Features/Stack/Architecture sections to accurately
describe the implemented system, and noted that the GNN approach was
explored in the published research paper but is not what's deployed
here. Covered by `test_readme_accuracy.py`.

---

## data/raw/ directory created but never read

**Status:** Resolved (2026-08).

`preprocessing.py` created a `data/raw/` directory on every run via
`os.makedirs(...)`, but the actual `creditcard.csv` load only ever
checked the repo root — `data/raw/` was dead: created, documented in
`data/README.md`, but never actually read.

**Fix:** `preprocessing.py` now checks `data/raw/creditcard.csv` first,
falling back to the repo root for backward compatibility.

---

## Unpinned xgboost/scikit-learn risk breaking the live deployment

**Status:** Mitigated (2026-08). Not fully resolved — see "long-term
fix" below.

`model.pkl` and `features.pkl` are loaded via `joblib.load()`, which
pickles the exact internal object graph of whatever xgboost/scikit-learn
version trained them. Loading `model.pkl` already emits:

```
UserWarning: If you are loading a serialized model (like pickle in
Python, RDS in R)... please export the model by calling
Booster.save_model() from that version first...
```

`requirements.txt` previously had no version bounds on `xgboost` or
`scikit-learn` at all. Streamlit Community Cloud reinstalls dependencies
fresh on every rebuild (redeploy, or periodic cache invalidation) — a
new major xgboost/scikit-learn release landing on PyPI could silently
break `joblib.load()` on the next rebuild, taking down the live demo
with no code change on this end.

**Mitigation applied:** pinned `scikit-learn>=1.4,<2.0` and
`xgboost>=2.0,<4.0` in `requirements.txt` — compatible ranges around
the versions the model was actually trained/verified against.

**Long-term fix (not yet done):** migrate `predict.py`/`train.py` to
xgboost's native `Booster.save_model()` / `Booster.load_model()`
(JSON/UBJSON format), which is explicitly designed to be
version-portable, instead of relying on pickle compatibility at all.
This requires re-exporting the model and updating `predict.py`'s
loading code — bigger change, deliberately not rushed into this pass.

---

## Model metrics panel shows hardcoded, not live, numbers

**Status:** Open — flagged, not fixed. Documenting the risk rather than
rushing a bigger feature.

The "Model Training Logistics" panel in `app.py` (Module 6) displays
static text — "Accuracy: 99.63%", "F1-Score: 0.87" — and a hardcoded
confusion matrix (`z = [[19958, 9], [1, 32]]`) for the heatmap
visualization. None of this is computed from the actual current
`model.pkl`; it's frozen from whenever this UI section was written.

**Risk:** if the model is ever retrained via `train.py` with different
data, hyperparameters, or even just a different random seed, this
panel will silently continue showing the old numbers with zero
indication they're stale or disconnected from what's actually deployed.
A user reading this panel has no way to tell "these are real current
metrics" from "these are frozen from an earlier training run."

**Why not fixed now:** computing genuinely live metrics would mean
loading a held-out test set at runtime, running predictions through
the current model, and rendering the real confusion matrix/accuracy —
a real feature addition (deciding on a canonical held-out set, handling
the load-time cost on every page view, etc.), not a small bug fix. That
deserves its own deliberate pass rather than a rushed implementation
bundled into an unrelated fix.

**Interim mitigation applied:** added a code comment directly above the
hardcoded values in `app.py` explaining they're static, so a future
contributor immediately sees the caveat instead of assuming the numbers
are live.

**Proper fix (future work):** add a small `evaluate_model()` helper
(likely in `train_utils.py` alongside `FEATURE_COLUMNS`) that loads a
fixed held-out test split, scores it with the current `model.pkl`, and
returns real accuracy/F1/confusion-matrix values — then have `app.py`
call it once per session (cached via `st.cache_data`) instead of
hardcoding the panel's contents.

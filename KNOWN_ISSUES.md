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

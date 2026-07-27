# Known Issues

## Train/inference feature mismatch — amount_bin, amount_ratio

**Status:** Open — needs retraining or code alignment before considered resolved.

`preprocessing.py` (used at training time) and `predict.py` (used at
inference time) compute `amount_bin` and `amount_ratio` differently:

| Feature | preprocessing.py (train) | predict.py (inference) |
|---|---|---|
| `amount_bin` | `pd.cut(df['amount'], bins=5)` — data-driven, equal-width bins based on the training set's min/max | `pd.cut(data['amount'], bins=[0,100,1000,10000,100000,inf])` — fixed edges |
| `amount_ratio` | `amount / amount.mean()` — training set mean | `amount / 5000` — hardcoded constant |

**Impact:** the model was trained on one feature distribution and is
served a differently-distributed version of the same named feature.
This likely degrades (rather than helps) the contribution of these two
features to the fraud probability score.

**Fix options:**
1. Retrain the model using the fixed bins/divisor from `predict.py`, or
2. Compute the training-set mean and equal-width bin edges once, hardcode
   them into `predict.py` to match exactly what the model saw during training.

Option 2 is lower-risk (no retraining required) and should be the next
concrete task on this repo.

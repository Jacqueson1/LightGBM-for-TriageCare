import os
import re
import time
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

SEED = 42
RUN = 3   # 1, 2 = earlier tests | 3 = mild guard | 4 = run 3 + faster learning

all_settings = {
    1: {"min_child_samples": 5},
    2: {"min_child_samples": 5, "min_child_weight": 0},   # crashed before
    3: {"min_child_samples": 5, "min_child_weight": 1e-6, "min_split_gain": 0},
    4: {"min_child_samples": 5, "min_child_weight": 1e-6, "min_split_gain": 0,
        "learning_rate": 0.3},
}
settings = all_settings[RUN]

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train.drop(columns=["diseases", "label_id"]).astype("int8")
y_train = train["label_id"]
X_test = test.drop(columns=["diseases", "label_id"]).astype("int8")
y_test = test["label_id"]

def clean(name):
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(name))

X_train.columns = [clean(c) for c in X_train.columns]
X_test.columns = [clean(c) for c in X_test.columns]
assert X_train.columns.is_unique, "Column names clash after cleaning"

print(f"RUN {RUN} settings:", settings)
print("Training started (can take a while with 587 classes)...")

# verbose=-1 hides the repeated -inf warnings (they are not errors)
model = LGBMClassifier(
    n_jobs=-1, random_state=SEED, verbose=-1, **settings
)

t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training done in {train_time:.1f}s")

t0 = time.time()
pred = model.predict(X_test)
predict_time = time.time() - t0

# Sanity check: a broken model predicts only a few different diseases
print("Different diseases predicted:", len(set(pred)), "of", y_train.nunique())

results = {
    "model": f"LightGBM (run {RUN})",
    "settings": str(settings),
    "accuracy": accuracy_score(y_test, pred),
    "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
    "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    "train_time_sec": round(train_time, 1),
    "predict_time_sec": round(predict_time, 1),
}

os.makedirs("results", exist_ok=True)
pd.DataFrame([results]).to_csv(f"results/lgbm_run{RUN}.csv", index=False)

print()
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
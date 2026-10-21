# predict-shap: put YOUR trained model into production, and explain every prediction with SHAP.
#
# Every hour (:25) this function:
#   1. loads the model you trained in Colab:   gs://<bucket>/models/model.joblib  (+ models/model_card.json)
#   2. reads your master table:                gs://<bucket>/structured/datasets/listings_master.csv
#      (or the table named in your model card's "data_key", e.g. listings_master_llm.csv for a Gemini model)
#   3. predicts every car this model version has not predicted yet
#   4. computes SHAP values for those predictions (summed back to each original feature)
# Which features? The "features" list in your model card (Lab 4: year, mileage, make, model). A midterm model
# trained on Gemini fields just lists more features; nothing here needs editing.
#   5. appends both to:                        gs://<bucket>/structured/preds_lab4/preds_master.csv
#                                              gs://<bucket>/structured/preds_lab4/shap_master.csv
# No model uploaded yet? It returns {"status": "no_model"} and does nothing (safe to deploy before Lab 4).
# HTTP entrypoint: predict_shap_http

import io
import json
import logging
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_master.csv")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "structured/preds_lab4")
MAX_PER_RUN_TREE = int(os.getenv("MAX_PER_RUN_TREE", "500"))   # tree SHAP is fast
MAX_PER_RUN_OTHER = int(os.getenv("MAX_PER_RUN_OTHER", "40"))   # sampled (Kernel) SHAP is slow: cap it

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("shap").setLevel(logging.WARNING)   # KernelExplainer logs every row at INFO

FEATURES = ["year", "mileage", "make", "model"]      # default, used when the model card has no "features" list
NUMERIC = ["year", "mileage"]                         # every other feature is treated as text
NICKNAMES = {"chevy": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz", "benz": "mercedes-benz"}


# -------------------- the SAME cleaning the Lab 4 notebook uses --------------------
def clean_cars(df: pd.DataFrame, text_cols=("make", "model")) -> pd.DataFrame:
    out = df.copy()
    for col in ["price", "year", "mileage"]:
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(r"[^\d.]+", "", regex=True), errors="coerce")
    for col in text_cols:
        out[col] = out[col].astype(str).str.strip().str.lower()
        out.loc[out[col].isin(["", "nan", "none"]), col] = np.nan
    out["make"] = out["make"].replace(NICKNAMES)
    return out


# -------------------- the SAME SHAP helper the Lab 4 notebook uses --------------------
def original_feature(transformed_name: str, text_cols=("make", "model")) -> str:
    name = transformed_name.split("__", 1)[-1]
    owner = None
    for col in text_cols:                      # longest matching name wins ("model" vs "model_year", say)
        if name.startswith(col + "_") and (owner is None or len(col) > len(owner)):
            owner = col
    if owner:
        return owner
    for col in ["year", "mileage"]:
        if name == col or name.startswith(col):
            return col
    return name


def explain_rows(pipe, X: pd.DataFrame, background: pd.DataFrame, features=FEATURES):
    """SHAP values for each row of X, summed back to the original features. Returns (base_value, DataFrame)."""
    pre = pipe.named_steps["pre"]
    est = pipe.named_steps["model"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = list(pre.get_feature_names_out())
    if hasattr(est, "tree_") or hasattr(est, "estimators_"):
        explainer = shap.TreeExplainer(est)
        values = explainer.shap_values(Xt, check_additivity=False)
        base = float(np.ravel(explainer.expected_value)[0])
    else:
        Bt = pre.transform(background)
        if hasattr(Bt, "toarray"):
            Bt = Bt.toarray()
        summary = shap.kmeans(Bt, min(10, len(Bt)))
        explainer = shap.KernelExplainer(est.predict, summary)
        values = explainer.shap_values(Xt, nsamples=100, silent=True)
        base = float(np.ravel(explainer.expected_value)[0])
    values = np.asarray(values)
    text_cols = [f for f in features if f not in NUMERIC]
    grouped = pd.DataFrame(0.0, index=X.index, columns=["shap_" + f for f in features])
    for j, name in enumerate(names):
        grouped["shap_" + original_feature(name, text_cols)] += values[:, j]
    return base, grouped


# -------------------- GCS helpers --------------------
def _read_csv(bucket, key, **kw):
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()), **kw)


def _write_csv(bucket, key, df):
    bucket.blob(key).upload_from_string(df.to_csv(index=False), content_type="text/csv")


def run_once(dry_run: bool = False) -> dict:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)

    model_blob = bucket.blob(f"{MODEL_PREFIX}/model.joblib")
    if not model_blob.exists():
        return {"status": "no_model", "message": f"Upload your model to gs://{GCS_BUCKET}/{MODEL_PREFIX}/model.joblib (Lab 4, Part 6)."}
    pipe = joblib.load(io.BytesIO(model_blob.download_as_bytes()))
    card_blob = bucket.blob(f"{MODEL_PREFIX}/model_card.json")
    card = json.loads(card_blob.download_as_text()) if card_blob.exists() else {}
    version = str(card.get("trained_at", "unknown"))
    trained_ids = set(str(p) for p in card.get("train_post_ids", []))
    model_type = card.get("model_type", type(pipe.named_steps["model"]).__name__)
    warning = None
    if card.get("sklearn_version") and card["sklearn_version"] != sklearn.__version__:
        warning = f"model saved with scikit-learn {card['sklearn_version']}, production runs {sklearn.__version__}"
        logging.warning(warning)

    features = list(card.get("features") or FEATURES)        # what this model was trained on
    text_cols = [f for f in features if f not in NUMERIC]
    data_key = card.get("data_key") or DATA_KEY              # a Gemini model's card points at listings_master_llm.csv
    raw = _read_csv(bucket, data_key, dtype={"post_id": str})
    if raw is None:
        return {"status": "error", "error": f"gs://{GCS_BUCKET}/{data_key} not found (run materialize-master / materialize-llm first)"}
    missing = [f for f in features if f not in raw.columns]
    if missing:
        return {"status": "error", "error": f"your model needs columns {missing} that {data_key} does not have. "
                "A Gemini-field model needs data_key = structured/datasets/listings_master_llm.csv in its model card."}
    cars = clean_cars(raw, text_cols)

    preds_key = f"{OUTPUT_PREFIX}/preds_master.csv"
    shap_key = f"{OUTPUT_PREFIX}/shap_master.csv"
    old_preds = _read_csv(bucket, preds_key, dtype={"post_id": str, "model_version": str})
    old_shap = _read_csv(bucket, shap_key, dtype={"post_id": str, "model_version": str})
    done = set()
    if old_preds is not None and len(old_preds) > 0:
        same_version = old_preds[old_preds["model_version"].astype(str) == version]
        done = set(same_version["post_id"].astype(str))

    todo = cars[~cars["post_id"].astype(str).isin(done)].copy()
    est = pipe.named_steps["model"]
    is_tree = hasattr(est, "tree_") or hasattr(est, "estimators_")
    cap = MAX_PER_RUN_TREE if is_tree else MAX_PER_RUN_OTHER
    todo = todo.tail(cap)                       # newest cars first in line
    if len(todo) == 0:
        return {"status": "ok", "new_predictions": 0, "model_version": version, "model_type": model_type, "warning": warning}

    X = todo[features]
    y_hat = pipe.predict(X)
    usable = cars[features].dropna(subset=NUMERIC)           # text blanks are fine: the model fills them itself
    background = usable.sample(min(50, len(usable)), random_state=0)
    base, shap_df = explain_rows(pipe, X, background, features)

    now = pd.Timestamp.now(tz="UTC").isoformat()
    preds = todo[["post_id", "scraped_at", "year", "mileage", "make", "model", "price"]].copy()
    preds = preds.rename(columns={"price": "actual_price"})
    preds["pred_price"] = np.round(y_hat, 2)
    preds["abs_error"] = (preds["pred_price"] - preds["actual_price"]).abs().round(2)
    preds["seen_in_training"] = preds["post_id"].astype(str).isin(trained_ids)
    preds["model_type"] = model_type
    preds["model_version"] = version
    preds["predicted_at"] = now

    shaps = pd.concat([todo[["post_id"]].reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
    shaps["base_value"] = round(base, 2)
    shaps["pred_price"] = np.round(y_hat, 2)
    shaps["model_version"] = version
    shaps["predicted_at"] = now

    if not dry_run:
        _write_csv(bucket, preds_key, pd.concat([old_preds, preds], ignore_index=True) if old_preds is not None else preds)
        _write_csv(bucket, shap_key, pd.concat([old_shap, shaps], ignore_index=True) if old_shap is not None else shaps)

    new_cars = preds[~preds["seen_in_training"]]
    return {
        "status": "ok",
        "model_type": model_type,
        "model_version": version,
        "new_predictions": int(len(preds)),
        "mae_all": round(float(preds["abs_error"].mean()), 2) if preds["abs_error"].notna().any() else None,
        "mae_unseen": round(float(new_cars["abs_error"].mean()), 2) if len(new_cars) and new_cars["abs_error"].notna().any() else None,
        "preds_key": preds_key,
        "shap_key": shap_key,
        "warning": warning,
        "dry_run": dry_run,
    }


def predict_shap_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})

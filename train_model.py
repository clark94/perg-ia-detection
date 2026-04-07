from pathlib import Path
import zipfile
import joblib
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "a-comprehensive-dataset-of-pattern-electroretinograms-for-ocular-electrophysiology-research-the-perg-ioba-dataset-1.0.0.zip"

MODEL_PATH = BASE_DIR / "model_v2_erg.joblib"
METRICS_PATH = BASE_DIR / "metrics_v2_erg.json"
DATASET_EXPORT_PATH = BASE_DIR / "dataset_v2_erg.csv"

ROOT = "a-comprehensive-dataset-of-pattern-electroretinograms-for-ocular-electrophysiology-research-the-perg-ioba-dataset-1.0.0/"
CSV_INSIDE_ZIP = ROOT + "csv/participants_info.csv"

CLINICAL_FEATURES = ["age_years", "sex", "va_re_logMar", "va_le_logMar"]
TARGET_COL = "target"


def safe_float_series(series):
    return pd.to_numeric(series, errors="coerce")


def load_participants():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open(CSV_INSIDE_ZIP) as f:
            df = pd.read_csv(f)

    df = df.copy()
    df["id_record"] = pd.to_numeric(df["id_record"], errors="coerce").astype("Int64")
    df["age_years"] = safe_float_series(df["age_years"])
    df["va_re_logMar"] = safe_float_series(df["va_re_logMar"])
    df["va_le_logMar"] = safe_float_series(df["va_le_logMar"])
    df["sex"] = df["sex"].fillna("Unknown").astype(str)
    df["diagnosis1"] = df["diagnosis1"].fillna("Unknown").astype(str)
    df[TARGET_COL] = (df["diagnosis1"] != "Normal").astype(int)
    return df


def load_signal_from_zip(zf, patient_id: int):
    signal_path = f"{ROOT}csv/{patient_id:04d}.csv"
    try:
        with zf.open(signal_path) as f:
            sig = pd.read_csv(f)
        return sig
    except KeyError:
        return None


def zero_crossings(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    signs = np.sign(x)
    return np.sum(signs[:-1] * signs[1:] < 0)


def slope_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    dx = np.diff(x)
    return np.mean(np.abs(dx)), np.max(np.abs(dx))


def area_abs(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.sum(np.abs(x))


def signal_energy(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.sum(x ** 2)


def dominant_fft_magnitude(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan
    centered = x - np.mean(x)
    fft_vals = np.abs(np.fft.rfft(centered))
    if len(fft_vals) <= 1:
        return np.nan
    return np.max(fft_vals[1:])


def extract_signal_features(values, prefix):
    values = np.asarray(values, dtype=float)

    mean_abs_slope, max_abs_slope = slope_stats(values)

    features = {
        f"{prefix}_mean": np.nanmean(values),
        f"{prefix}_std": np.nanstd(values),
        f"{prefix}_min": np.nanmin(values),
        f"{prefix}_max": np.nanmax(values),
        f"{prefix}_median": np.nanmedian(values),
        f"{prefix}_q25": np.nanpercentile(values, 25),
        f"{prefix}_q75": np.nanpercentile(values, 75),
        f"{prefix}_range": np.nanmax(values) - np.nanmin(values),
        f"{prefix}_energy": signal_energy(values),
        f"{prefix}_area_abs": area_abs(values),
        f"{prefix}_zero_crossings": zero_crossings(values),
        f"{prefix}_mean_abs_slope": mean_abs_slope,
        f"{prefix}_max_abs_slope": max_abs_slope,
        f"{prefix}_fft_peak": dominant_fft_magnitude(values),
        f"{prefix}_first": values[0] if len(values) > 0 else np.nan,
        f"{prefix}_last": values[-1] if len(values) > 0 else np.nan,
    }

    # Pics simples
    if len(values) > 0:
        idx_max = int(np.nanargmax(values))
        idx_min = int(np.nanargmin(values))
        features[f"{prefix}_idx_max"] = idx_max
        features[f"{prefix}_idx_min"] = idx_min
    else:
        features[f"{prefix}_idx_max"] = np.nan
        features[f"{prefix}_idx_min"] = np.nan

    return features


def build_erg_features():
    rows = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for patient_id in range(1, 337):
            sig = load_signal_from_zip(zf, patient_id)

            row = {"id_record": patient_id}

            if sig is None or sig.empty:
                rows.append(row)
                continue

            # Colonnes attendues dans le dataset
            re_col = "RE_1"
            le_col = "LE_1"

            if re_col not in sig.columns or le_col not in sig.columns:
                rows.append(row)
                continue

            re_vals = pd.to_numeric(sig[re_col], errors="coerce").to_numpy()
            le_vals = pd.to_numeric(sig[le_col], errors="coerce").to_numpy()

            row.update(extract_signal_features(re_vals, "re"))
            row.update(extract_signal_features(le_vals, "le"))

            # Features de symétrie entre yeux
            if len(re_vals) == len(le_vals) and len(re_vals) > 0:
                diff = re_vals - le_vals
                row.update(extract_signal_features(diff, "re_le_diff"))
                row["corr_re_le"] = np.corrcoef(
                    np.nan_to_num(re_vals, nan=0.0),
                    np.nan_to_num(le_vals, nan=0.0)
                )[0, 1]
            else:
                row["corr_re_le"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def make_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


def build_models(preprocessor):
    return {
        "RandomForest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced",
                max_depth=None,
                min_samples_leaf=2
            ))
        ]),
        "LogisticRegression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),
        "GradientBoosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                random_state=42
            ))
        ]),
    }


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {ZIP_PATH}")

    print("Chargement des données cliniques...")
    df_participants = load_participants()

    print("Extraction des features ERG...")
    df_erg = build_erg_features()

    df = df_participants.merge(df_erg, on="id_record", how="left")

    # Export utile pour inspection
    df.to_csv(DATASET_EXPORT_PATH, index=False)

    categorical_features = ["sex"]
    numeric_features = [c for c in df.columns if c not in [
        "id_record", "date", "diagnosis1", "diagnosis2", "diagnosis3",
        "unilateral", "rep_record", "comments", TARGET_COL, "sex"
    ]]

    features = numeric_features + categorical_features

    df_model = df[features + [TARGET_COL]].copy()
    X = df_model[features]
    y = df_model[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    results = {}
    best_model = None
    best_name = None
    best_auc = -1
    best_pred = None
    best_proba = None

    print("\nEntraînement des modèles...")
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": float(acc),
            "auc": float(auc)
        }

        print(f"{name:<20} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model
            best_pred = y_pred
            best_proba = y_proba

    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "version": "V2_clinique_plus_ERG",
        "best_model": best_name,
        "accuracy": float(accuracy_score(y_test, best_pred)),
        "auc": float(roc_auc_score(y_test, best_proba)),
        "report": classification_report(y_test, best_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "models_comparison": results,
        "n_samples": int(len(df)),
        "n_features": int(len(features)),
        "clinical_features": CLINICAL_FEATURES,
        "erg_feature_count": int(len([f for f in features if f not in CLINICAL_FEATURES and f != "sex"])),
        "saved_model_path": str(MODEL_PATH),
        "saved_dataset_path": str(DATASET_EXPORT_PATH),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Meilleur modèle : {best_name}")
    print(f"✅ Modèle sauvegardé : {MODEL_PATH}")
    print(f"✅ Métriques sauvegardées : {METRICS_PATH}")
    print(f"✅ Dataset fusionné sauvegardé : {DATASET_EXPORT_PATH}")


if __name__ == "__main__":
    main()
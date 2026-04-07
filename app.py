from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Diagnostic IA Oculaire V2",
    page_icon="👁️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_v2_erg.csv"
MODEL_PATH = BASE_DIR / "model_v2_erg.joblib"
METRICS_PATH = BASE_DIR / "metrics_v2_erg.json"

TARGET_COL = "target"
CLINICAL_FEATURES = ["age_years", "sex", "va_re_logMar", "va_le_logMar"]

# =========================
# UI TOOLS
# =========================
def kpi_card(titre: str, valeur: str):
    st.metric(label=titre, value=valeur)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def afficher_jauge_risque(proba: float):
    st.progress(int(proba * 100))
    if proba >= 0.80:
        st.error(f"Risque élevé : {proba:.1%}")
    elif proba >= 0.50:
        st.warning(f"Risque modéré : {proba:.1%}")
    else:
        st.success(f"Risque faible : {proba:.1%}")


# =========================
# DATA
# =========================
@st.cache_data
def charger_dataset():
    df = pd.read_csv(DATASET_PATH)

    df["id_record"] = pd.to_numeric(df["id_record"], errors="coerce").astype("Int64")
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["va_re_logMar"] = pd.to_numeric(df["va_re_logMar"], errors="coerce")
    df["va_le_logMar"] = pd.to_numeric(df["va_le_logMar"], errors="coerce")

    df["sex"] = df["sex"].fillna("Unknown").astype(str)
    df["diagnosis1"] = df["diagnosis1"].fillna("Unknown").astype(str)

    return df


@st.cache_resource
def charger_modele():
    return joblib.load(MODEL_PATH)


@st.cache_data
def charger_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def build_full_dataset():
    df_full = charger_dataset()

    categorical_features = ["sex"]

    excluded_cols = {
        "id_record", "date", "diagnosis1", "diagnosis2", "diagnosis3",
        "unilateral", "rep_record", "comments", TARGET_COL
    }

    numeric_features = [c for c in df_full.columns if c not in excluded_cols and c not in categorical_features]
    feature_cols = numeric_features + categorical_features

    return df_full, feature_cols


# =========================
# CHECK FILES
# =========================
if not DATASET_PATH.exists():
    st.error("dataset_v2_erg.csv introuvable")
    st.stop()

if not MODEL_PATH.exists():
    st.error("model_v2_erg.joblib introuvable")
    st.stop()

modele = charger_modele()
metrics = charger_metrics()
df_full, feature_cols = build_full_dataset()
df_participants = df_full.copy()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("👁️ Diagnostic IA V2")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Dashboard", "🔬 Analyse patient", "📈 Performances"]
)

st.sidebar.write(f"Patients : {len(df_participants)}")
st.sidebar.write(f"Malades : {int(df_participants['target'].sum())}")

if metrics:
    st.sidebar.write(f"AUC : {metrics.get('auc', 0):.2%}")


# =========================
# ACCUEIL
# =========================
if page == "🏠 Accueil":

    st.title("👁️ Diagnostic IA Oculaire V2")

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(df_participants))
    c2.metric("Cas malades", int(df_participants["target"].sum()))
    c3.metric("Taux", f"{df_participants['target'].mean()*100:.1f}%")

    st.divider()

    st.write("Modèle utilisant données cliniques + ERG")

    st.caption("Outil d’aide à la décision uniquement")


# =========================
# DASHBOARD
# =========================
elif page == "📊 Dashboard":

    st.title("Dashboard")

    fig, ax = plt.subplots()
    df_participants["target"].value_counts().plot(kind="bar", ax=ax)
    show_fig(fig)

    st.dataframe(df_participants.head())


# =========================
# ANALYSE PATIENT
# =========================
elif page == "🔬 Analyse patient":

    st.title("Analyse patient")

    patient_ids = df_participants["id_record"].dropna().astype(int).tolist()
    patient_id = st.selectbox("Patient", patient_ids)

    row = df_full[df_full["id_record"] == patient_id]

    if st.button("Prédire"):

        X = row[feature_cols]

        pred = modele.predict(X)[0]
        proba = modele.predict_proba(X)[0][1]

        st.metric("Probabilité maladie", f"{proba:.2%}")

        afficher_jauge_risque(proba)


# =========================
# PERFORMANCE
# =========================
else:

    st.title("Performances")

    if metrics:
        st.metric("AUC", f"{metrics.get('auc',0):.2%}")
        st.metric("Accuracy", f"{metrics.get('accuracy',0):.2%}")

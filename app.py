from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_v2_erg.csv"
MODEL_PATH = BASE_DIR / "model_v2_erg.joblib"
METRICS_PATH = BASE_DIR / "metrics_v2_erg.json"

TARGET_COL = "target"
CLINICAL_FEATURES = ["age_years", "sex", "va_re_logMar", "va_le_logMar"]

st.set_page_config(
    page_title="Diagnostic IA Oculaire V2",
    page_icon="👁️",
    layout="wide",
)

st.markdown("""
<style>
.main > div {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    font-weight: 700 !important;
}
.stButton > button {
    width: 100%;
    height: 46px;
    border-radius: 12px;
    font-weight: 600;
}
.kpi-card {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0.6rem;
}
.kpi-title {
    font-size: 0.92rem;
    opacity: 0.85;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
}
.small-note {
    font-size: 0.92rem;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)


# =========================
# OUTILS UI
# =========================
def kpi_card(titre: str, valeur: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{titre}</div>
            <div class="kpi-value">{valeur}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


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
# CHARGEMENT DONNÉES
# =========================
@st.cache_data
def charger_dataset():
    df = pd.read_csv(DATASET_PATH)

    if "id_record" in df.columns:
        df["id_record"] = pd.to_numeric(df["id_record"], errors="coerce").astype("Int64")

    if "age_years" in df.columns:
        df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

    if "va_re_logMar" in df.columns:
        df["va_re_logMar"] = pd.to_numeric(df["va_re_logMar"], errors="coerce")

    if "va_le_logMar" in df.columns:
        df["va_le_logMar"] = pd.to_numeric(df["va_le_logMar"], errors="coerce")

    if "sex" in df.columns:
        df["sex"] = df["sex"].fillna("Unknown").astype(str)

    if "diagnosis1" in df.columns:
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

    categorical_features = ["sex"] if "sex" in df_full.columns else []

    excluded_cols = {
        "id_record", "date", "diagnosis1", "diagnosis2", "diagnosis3",
        "unilateral", "rep_record", "comments", TARGET_COL
    }

    numeric_features = [c for c in df_full.columns if c not in excluded_cols and c not in categorical_features]
    feature_cols = numeric_features + categorical_features

    return df_full, feature_cols


if not DATASET_PATH.exists():
    st.error("Le fichier dataset_v2_erg.csv est introuvable.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Le modèle V2 est introuvable. Lance d’abord `python train_model.py`.")
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

st.sidebar.divider()
st.sidebar.markdown("### Informations")
st.sidebar.write(f"Patients : **{len(df_participants)}**")
st.sidebar.write(f"Cas malades : **{int(df_participants['target'].sum())}**")
st.sidebar.write(f"Cas normaux : **{int((df_participants['target'] == 0).sum())}**")

if metrics is not None:
    st.sidebar.divider()
    st.sidebar.markdown("### Modèle V2")
    st.sidebar.write(f"Meilleur modèle : **{metrics.get('best_model', 'N/A')}**")
    st.sidebar.write(f"AUC : **{metrics.get('auc', 0):.2%}**")
    st.sidebar.write(f"Accuracy : **{metrics.get('accuracy', 0):.2%}**")


# =========================
# GRAPHIQUES DASHBOARD
# =========================
def graphique_repartition(df_local):
    repartition = df_local["target"].value_counts().sort_index()
    labels = ["Normal", "Maladie"]
    valeurs = [repartition.get(0, 0), repartition.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, valeurs)
    ax.set_title("Répartition des cas")
    ax.set_ylabel("Nombre de patients")
    show_fig(fig)


def graphique_age(df_local):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_local["age_years"].dropna(), bins=20)
    ax.set_title("Distribution des âges")
    ax.set_xlabel("Âge")
    ax.set_ylabel("Fréquence")
    show_fig(fig)


def graphique_acuite(df_local):
    resume = pd.DataFrame({
        "Œil droit": [df_local["va_re_logMar"].dropna().mean()],
        "Œil gauche": [df_local["va_le_logMar"].dropna().mean()]
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(resume.columns, resume.iloc[0].values)
    ax.set_title("Acuité visuelle moyenne (logMAR)")
    ax.set_ylabel("Valeur moyenne")
    show_fig(fig)


def get_patient_row(patient_id: int):
    row = df_full[df_full["id_record"] == patient_id].copy()
    if row.empty:
        return None
    return row.iloc[0]


def get_feature_dataframe_for_patient(patient_id: int):
    row = df_full[df_full["id_record"] == patient_id].copy()
    if row.empty:
        return None
    return row[feature_cols]


# =========================
# PAGE ACCUEIL
# =========================
if page == "🏠 Accueil":
    st.title("👁️ Diagnostic IA des maladies oculaires — V2")
    st.markdown("### Version clinique + features ERG extraites du dataset PERG-IOBA")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Patients", str(len(df_participants)))
    with c2:
        kpi_card("Cas malades", str(int(df_participants["target"].sum())))
    with c3:
        taux = df_participants["target"].mean() * 100 if len(df_participants) > 0 else 0
        kpi_card("Taux de cas malades", f"{taux:.1f}%")

    st.divider()

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("## 🎯 Objectif")
        st.write(
            "Cette V2 exploite les variables cliniques tabulaires ainsi que les "
            "features ERG déjà extraites et stockées dans le dataset enrichi."
        )
        st.write(
            "Le modèle apprend à partir de caractéristiques issues des deux yeux : "
            "statistiques du signal, énergie, pentes, pics, différences entre yeux "
            "et corrélation œil droit / œil gauche."
        )

    with g2:
        st.markdown("## 🩺 Intérêt scientifique")
        st.info(
            "Cette version repose sur des données réelles enrichies issues des signaux ERG, "
            "ce qui la rend plus crédible qu’une simple approche basée uniquement sur les variables cliniques."
        )
        st.markdown(
            '<p class="small-note">Outil d’aide à la décision uniquement. Il ne remplace pas un diagnostic médical.</p>',
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("## 🚀 Ce qui change par rapport à la V1")
    st.write("- V1 : âge, sexe, acuité visuelle")
    st.write("- V2 : âge, sexe, acuité visuelle + features ERG extraites")
    st.write("- V2 : possibilité d’analyser un patient réel du dataset enrichi")

# =========================
# PAGE DASHBOARD
# =========================
elif page == "📊 Dashboard":
    st.title("📊 Tableau de bord du dataset")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Patients", str(len(df_participants)))
    with c2:
        kpi_card("Cas malades", str(int(df_participants["target"].sum())))
    with c3:
        kpi_card("Cas normaux", str(int((df_participants["target"] == 0).sum())))
    with c4:
        taux = (df_participants["target"].mean() * 100) if len(df_participants) > 0 else 0
        kpi_card("Taux maladie", f"{taux:.1f}%")

    st.divider()

    col_filtre1, col_filtre2 = st.columns(2)
    with col_filtre1:
        sexe_filtre = st.selectbox(
            "Filtrer par sexe",
            ["Tous", "Male", "Female", "Unknown"],
            format_func=lambda x: {
                "Tous": "Tous",
                "Male": "Homme",
                "Female": "Femme",
                "Unknown": "Inconnu"
            }[x]
        )
    with col_filtre2:
        type_filtre = st.selectbox("Filtrer par type de cas", ["Tous", "Normal", "Maladie"])

    df_filtre = df_participants.copy()

    if sexe_filtre != "Tous":
        df_filtre = df_filtre[df_filtre["sex"] == sexe_filtre]

    if type_filtre == "Normal":
        df_filtre = df_filtre[df_filtre["target"] == 0]
    elif type_filtre == "Maladie":
        df_filtre = df_filtre[df_filtre["target"] == 1]

    st.markdown(f"**Nombre de lignes après filtre :** {len(df_filtre)}")

    st.divider()

    g1, g2 = st.columns(2)
    with g1:
        graphique_repartition(df_filtre)
    with g2:
        graphique_age(df_filtre)

    st.divider()

    g3, g4 = st.columns(2)
    with g3:
        graphique_acuite(df_filtre)
    with g4:
        st.subheader("Répartition par sexe")
        repartition_sexe = df_filtre["sex"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(repartition_sexe.index.astype(str), repartition_sexe.values)
        ax.set_title("Répartition par sexe")
        ax.set_ylabel("Nombre")
        show_fig(fig)

    st.divider()

    st.subheader("🗂️ Aperçu des données cliniques")
    colonnes_affichage = [
        col for col in [
            "id_record", "age_years", "sex", "va_re_logMar",
            "va_le_logMar", "diagnosis1", "target"
        ] if col in df_filtre.columns
    ]
    st.dataframe(df_filtre[colonnes_affichage].head(30), use_container_width=True)

# =========================
# PAGE ANALYSE PATIENT
# =========================
elif page == "🔬 Analyse patient":
    st.title("🔬 Analyse patient")

    patient_ids = [int(x) for x in df_participants["id_record"].dropna().tolist()]
    patient_id = st.selectbox("Choisir un patient", patient_ids, format_func=lambda x: f"Patient {x:04d}")

    patient = get_patient_row(patient_id)
    patient_features_df = get_feature_dataframe_for_patient(patient_id)

    if patient is None:
        st.error("Patient introuvable.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("ID patient", f"{patient_id:04d}")
    with c2:
        kpi_card("Âge", f"{patient.get('age_years', 'N/A')}")
    with c3:
        sexe_aff = patient.get("sex", "Unknown")
        sexe_aff = "Homme" if sexe_aff == "Male" else "Femme" if sexe_aff == "Female" else "Inconnu"
        kpi_card("Sexe", sexe_aff)
    with c4:
        kpi_card("Diagnostic réel", str(patient.get("diagnosis1", "N/A")))

    st.divider()

    gauche, droite = st.columns([1.2, 1])

    with gauche:
        st.subheader("Features ERG disponibles")
        colonnes_features = [
            "re_mean", "re_std", "re_energy", "re_fft_peak",
            "le_mean", "le_std", "le_energy", "le_fft_peak",
            "corr_re_le"
        ]
        colonnes_dispo = [c for c in colonnes_features if c in df_full.columns]
        if colonnes_dispo:
            st.dataframe(
                df_full.loc[df_full["id_record"] == patient_id, ["id_record"] + colonnes_dispo],
                use_container_width=True
            )
        else:
            st.info("Features ERG non disponibles.")

    with droite:
        st.subheader("Variables cliniques")
        st.write(f"- **Acuité œil droit (logMAR)** : {patient.get('va_re_logMar', np.nan)}")
        st.write(f"- **Acuité œil gauche (logMAR)** : {patient.get('va_le_logMar', np.nan)}")
        st.write(f"- **Classe cible** : {'Maladie' if patient.get('target', 0) == 1 else 'Normal'}")

        if patient_features_df is not None and st.button("🚀 Lancer la prédiction V2"):
            pred = int(modele.predict(patient_features_df)[0])
            probas = modele.predict_proba(patient_features_df)[0]
            proba_normal = float(probas[0])
            proba_maladie = float(probas[1])

            st.markdown("### Résultat du modèle")
            c5, c6 = st.columns(2)
            c5.metric("Probabilité Normal", f"{proba_normal:.1%}")
            c6.metric("Probabilité Maladie", f"{proba_maladie:.1%}")

            afficher_jauge_risque(proba_maladie)

            if pred == 1:
                st.error(f"⚠️ Profil prédit : pathologique — probabilité {proba_maladie:.1%}")
            else:
                st.success(f"✅ Profil prédit : normal — probabilité de normalité {proba_normal:.1%}")

# =========================
# PAGE PERFORMANCES
# =========================
else:
    st.title("📈 Performances du modèle V2")

    if metrics is None:
        st.warning("Le fichier metrics_v2_erg.json est introuvable. Relance `python train_model.py`.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Meilleur modèle", metrics.get("best_model", "N/A"))
    with c2:
        kpi_card("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with c3:
        kpi_card("AUC", f"{metrics.get('auc', 0):.2%}")

    st.divider()

    report = metrics.get("report", {})
    normal = report.get("0", {})
    maladie = report.get("1", {})

    st.markdown("### 🔎 Métriques par classe")
    a1, a2, a3 = st.columns(3)
    a1.metric("Précision Normal", f"{normal.get('precision', 0):.2%}")
    a2.metric("Rappel Normal", f"{normal.get('recall', 0):.2%}")
    a3.metric("F1-score Normal", f"{normal.get('f1-score', 0):.2%}")

    a4, a5, a6 = st.columns(3)
    a4.metric("Précision Maladie", f"{maladie.get('precision', 0):.2%}")
    a5.metric("Rappel Maladie", f"{maladie.get('recall', 0):.2%}")
    a6.metric("F1-score Maladie", f"{maladie.get('f1-score', 0):.2%}")

    st.divider()

    st.markdown("### 🧠 Analyse scientifique")
    st.write("- Cette version exploite à la fois les données cliniques et les features ERG.")
    st.write("- L’AUC permet d’évaluer la séparation entre cas normaux et pathologiques.")
    st.write("- Les features ERG apportent une information physiologique absente dans la V1.")
    st.write("- Cette approche est plus crédible pour un projet biomédical ou hackathon santé.")

    st.divider()

    if "models_comparison" in metrics:
        st.subheader("Comparaison des modèles")
        comparaison_df = pd.DataFrame(metrics["models_comparison"]).T
        st.dataframe(comparaison_df, use_container_width=True)

    st.divider()

    if "confusion_matrix" in metrics:
        st.subheader("Matrice de confusion")
        cm = np.array(metrics["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm)
        ax.set_title("Matrice de confusion")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Maladie"])
        ax.set_yticklabels(["Normal", "Maladie"])

        seuil = cm.max() / 2 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                couleur = "white" if cm[i, j] > seuil else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=couleur)

        fig.colorbar(im, ax=ax)
        show_fig(fig)

    st.divider()

    st.subheader("Rapport détaillé")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# 👁️ IA Oculaire — PERG-IOBA V2

## 🧠 Présentation
Ce projet propose un système d’intelligence artificielle pour la détection précoce d’anomalies oculaires à partir de données cliniques et de signaux ERG.

⚠️ Outil d’aide à la décision, non médical.

## 🎯 Objectif
Classifier :
- 0 → Sain
- 1 → Pathologie

## 🚀 Points forts
- Dataset PERG-IOBA
- Signaux ERG réels
- Feature engineering avancé
- Fusion données cliniques + signaux
- Comparaison de modèles ML
- Application Streamlit

## 📁 Structure
PERG_AI_PROJECT/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
└── outputs/

## ⚙️ Méthodologie
- Extraction features ERG
- Prétraitement (scaling, encoding)
- Modèles : RF, Logistic, GB

## ▶️ Exécution
python train_model.py
streamlit run app.py

## 📦 Dépendances
- pandas
- scikit-learn
- streamlit
- numpy

## 💡 Pitch
Pipeline complet data science avec visualisation interactive.

## ⚠️ Limites
- Dataset limité
- Binaire

## 🔮 Améliorations
- Deep Learning
- SHAP
- Déploiement cloud

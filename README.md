# PERG-AI — Prédiction erg & IA explicable

Application de Machine Learning dédiée à l’analyse de données électrophysiologiques oculaires (ERG) et de variables cliniques.

PERG-AI a pour objectif d’explorer l’utilisation de modèles de classification pour aider à identifier des profils pathologiques à partir de données ERG, tout en proposant une interprétation des prédictions grâce à SHAP.

---

## Démo

Application déployée sur Railway :

👉 https://perg-ai-v3-production. up.railway.app/

---

## Objectif du projet

Le projet repose sur un jeu de données composé de **336 enregistrements ERG**, associés à différentes variables électrophysiologiques et cliniques.

Le pipeline permet de :

- préparer et nettoyer les données ;
- construire des variables utilisables par les modèles ;
- entraîner des modèles de classification ;
- évaluer leurs performances ;
- générer une prédiction à partir d’un enregistrement ;
- afficher un score de confiance ;
- interpréter les facteurs influençant la prédiction avec SHAP ;
- rendre les résultats accessibles à travers une application web interactive.

---

## Fonctionnalités

### Classification

L’application permet de classifier un profil comme :

- normal ;
- pathologique.

### Score de confiance

Chaque prédiction est accompagnée d’un score permettant de visualiser le niveau de confiance du modèle.

### Explainable AI avec SHAP

Les prédictions peuvent être interprétées à l’aide de **SHAP**, afin d’identifier les variables ayant le plus contribué au résultat obtenu.

### Interface interactive

L’application permet de sélectionner un enregistrement, d’exécuter l’analyse et de visualiser directement :

- la classification ;
- le score de confiance ;
- le niveau de risque ;
- l’importance des différentes caractéristiques.

---

## Apprentissage automatique de pipeline

Le projet suit les principales étapes d’un pipeline de Data Science :

1. préparation des données ;
2. nettoyage et contrôle des valeurs ;
3. sélection et construction des variables ;
4. séparation des données d’entraînement et de test ;
5. entraînement des modèles ;
6. évaluation ;
7. interprétation des prédictions ;
8. intégration dans une application ;
9. déploiement.

---

## Technologies

### Apprentissage automatique

- Python
- scikit-learn
- XGBoost
- Pandas
- NumPy

### IA explicable

- SHAP

### Visualisation et application

- Streamlit
- Intrigue

### Déploiement

- Chemin de fer
- Vas-y
- GitHub

---

## Exemple de résultat

L’application fournit un rapport synthétique comprenant notamment :

- le résultat de classification ;
- le niveau de confiance ;
- une représentation du risque ;
- les principales variables influençant la prédiction.

L’objectif est de rendre les résultats du modèle plus compréhensibles et plus facilement exploitables.

---

## Compétences mises en pratique

Ce projet m’a permis de travailler sur plusieurs dimensions d’un projet d’intelligence artificielle :

- préparation de données ;
- Apprentissage automatique ;
- classification supervisée ;
- évaluation de modèles ;
- interprétabilité des modèles ;
- SHAP ;
- développement d’application ;
- visualisation de résultats ;
- déploiement ;
- versionnement avec Git/GitHub.

---

## Architecture générale

« ''Texte
Données ERG / cliniques
        |
        v
Préparation des données
        |
        v
Ingénierie des caractéristiques
        |
        v
Apprentissage automatique
        |
        v
Évaluation
        |
        v
Explicabilité — SHAP
        |
        v
Application Streamlit
        |
        v
Déploiement Railway

## Auteur
MAMPOUYA CLARK Chancy Loic Franlly
Master 2 SISE-Université Lumière Lyon2

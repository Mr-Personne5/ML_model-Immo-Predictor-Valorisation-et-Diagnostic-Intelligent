# Immo Predictor — Valorisation et Diagnostic Intelligent

**Auteur :** Djiba Kaba
**Contexte :** Master IA — M1 Semestre 1 — Bloc 2 Machine Learning
**Dataset :** [Ames Housing Dataset](https://www.kaggle.com/datasets/lespin/house-prices-dataset/data) (Kaggle)

---

## Objectifs

Ce projet aborde deux tâches de Machine Learning sur des données immobilières :

| Tâche | Cible | Type |
|---|---|---|
| Estimation de prix | `SalePrice` | Régression |
| Identification du type de bâtiment | `BldgType` | Classification |

---

## Structure du projet

```
Projet/
├── Djiba_Kaba_ML.ipynb       # Notebook principal
├── train.csv                 # Données d'entraînement (1 460 observations)
├── test.csv                  # Données de test
├── sample_submission.csv     # Format de soumission
├── data_description.txt      # Description des 81 variables
└── README.md
```

---

## Dataset

Le dataset **Ames Housing** décrit des biens immobiliers vendus à Ames, Iowa (USA).

- **1 460 observations**, **81 variables**
- **38 variables numériques** (surfaces, années, notes de qualité...)
- **43 variables catégorielles** (quartier, style architectural, type de vente...)

---

## Pipeline

### 1. Analyse Exploratoire (EDA)
- Distribution de `SalePrice` (asymétrie droite, transformation log)
- Matrice de corrélations des features avec le prix
- Distribution de `BldgType` (forte classe dominante : 84% `1Fam`)
- Nuages de points : features numériques vs prix

### 2. Pré-traitement
- Sélection des features pertinentes (15 pour la régression, 7 pour la classification)
- Aucune valeur manquante dans les features sélectionnées
- Encodage `LabelEncoder` pour les variables catégorielles (`Neighborhood`, `HouseStyle`)
- Standardisation `StandardScaler` uniquement pour le SVM (sensible aux échelles)

### 3. Modélisation

**Régression** (split 80/20, `random_state=42`) :
- Decision Tree Regressor (`max_depth=10`, `min_samples_split=5`)
- Random Forest Regressor (200 arbres, `max_depth=15`)

**Classification** (split 80/20 stratifié, `random_state=42`) :
- SVM avec noyau RBF (`C=10`, `gamma='scale'`)
- Random Forest Classifier (200 arbres, profondeur illimitée)

---

## Résultats

### Régression — Estimation de `SalePrice`

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Decision Tree | 25 174 $ | 39 233 $ | 0.799 |
| **Random Forest** | **18 163 $** | **30 039 $** | **0.882** |

Le **Random Forest** est nettement supérieur grâce à l'ensemble learning (agrégation de 200 arbres).
Features les plus importantes : `OverallQual`, `GrLivArea`, `GarageArea`.

### Classification — Identification de `BldgType`

| Modèle | Accuracy | F1-score (pondéré) |
|---|---|---|
| SVM (RBF) | 0.880 | 0.864 |
| **Random Forest** | **0.887** | **0.872** |

Le **Random Forest** gère mieux le déséquilibre des classes (84% `1Fam`).
Features les plus discriminantes : `Neighborhood`, `HouseStyle`.

> **Note :** Le F1-score pondéré est la métrique principale pour la classification, car l'accuracy seule est trompeuse avec un dataset aussi déséquilibré.

---

## Dépendances

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Installation :
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Pistes d'amélioration

1. **SMOTE** : sur-échantillonnage des classes minoritaires de `BldgType`
2. **GridSearchCV** : optimisation des hyperparamètres
3. **Feature engineering** : ratio surface/pièces, âge du bien
4. **Traitement des outliers** : biens atypiques dans `SalePrice` et `LotArea`
5. **Modèles avancés** : XGBoost, LightGBM pour les données tabulaires

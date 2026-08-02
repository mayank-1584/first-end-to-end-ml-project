# End-to-End ML Project — Student Exam Performance Prediction

An end-to-end machine learning pipeline that predicts a student's **math score** based on demographic and academic features (gender, ethnicity, parental education, lunch type, test preparation, and reading/writing scores). Built to practice a production-style ML project structure: data ingestion → data transformation → model training, with reusable logging and exception-handling modules.

## Project Structure

```
first-end-to-end-ml-project/
├── artifical/                      # Generated artifacts (raw/train/test data, trained model, preprocessor)
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
├── catboost_info/                  # CatBoost training logs (auto-generated)
├── notebook/
│   ├── EDA_model.ipynb             # Exploratory data analysis
│   ├── MODEL_TRAINING.ipynb        # Model experimentation
│   └── data/
│       └── stud.csv                # Source dataset
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads raw data, splits into train/test
│   │   ├── data_transformation.py  # Builds preprocessing pipeline (imputation, scaling, encoding)
│   │   └── model_trainer.py        # Trains & tunes multiple regressors, saves the best one
│   ├── pipeline/
│   │   ├── train_pipeline.py       # (WIP) end-to-end training entry point
│   │   └── predict_pipeline.py     # (WIP) inference entry point
│   ├── exception.py                # Custom exception with file/line-level error messages
│   ├── logger.py                   # Centralized logging setup
│   └── utils.py                    # Shared helpers (save/load objects, model evaluation)
├── requirements.txt
├── setup.py
└── README.md
```

## Dataset

The dataset (`notebook/data/stud.csv`) contains student records with the following columns:

| Column | Description |
|---|---|
| `gender` | Student's gender |
| `race_ethnicity` | Ethnicity group |
| `parental_level_of_education` | Highest education level of parent |
| `lunch` | Lunch type (standard / free-reduced) |
| `test_preparation_course` | Whether the test prep course was completed |
| `reading_score` | Reading score (feature) |
| `writing_score` | Writing score (feature) |
| `math_score` | Math score (**target**) |

## How the Pipeline Works

1. **Data Ingestion** (`src/components/data_ingestion.py`)
   Reads the raw CSV, saves a copy to `artifical/data.csv`, and splits it into train/test sets (80/20).

2. **Data Transformation** (`src/components/data_transformation.py`)
   Builds a `ColumnTransformer` pipeline:
   - Numerical columns (`reading_score`, `writing_score`) → median imputation + standard scaling
   - Categorical columns → most-frequent imputation + one-hot encoding + scaling
   The fitted preprocessor is saved as `artifical/preprocessor.pkl`.

3. **Model Training** (`src/components/model_trainer.py`)
   Trains and tunes several regression models via `GridSearchCV`:
   - Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, CatBoost, AdaBoost
   Selects the best-performing model (by R² score) and saves it as `artifical/model.pkl`.

## Modeling Approach

This section explains the reasoning behind the modeling decisions in this project.

**1. Problem framing**
Predicting `math_score` from the other student attributes is a **supervised regression** problem — the target is continuous, not a category, so regression models (not classifiers) are the right tool.

**2. Feature handling**
- **Numerical features** (`reading_score`, `writing_score`) are imputed with the **median** (robust to outliers) and then **standard-scaled**, since distance/gradient-based models (e.g. linear regression, KNN) are sensitive to feature scale.
- **Categorical features** (gender, ethnicity, parental education, lunch, test prep) are imputed with the **most frequent value** and **one-hot encoded**, since they have no inherent order — encoding them as raw integers would falsely imply a ranking.
- Both branches are combined with a single `ColumnTransformer`, and the *same* fitted transformer (not a freshly-fit one) is reused on the test set, which avoids data leakage.

**3. Train/test split**
An 80/20 split with a fixed `random_state` keeps results reproducible and gives the test set enough samples to estimate generalization error reliably.

**4. Model selection strategy**
Rather than betting on a single algorithm, the pipeline trains **seven different regressors** — from simple (Linear Regression) to ensemble/boosted (Random Forest, Gradient Boosting, XGBoost, CatBoost, AdaBoost) — and picks the best by test-set performance. This matters because:
- Linear models are a fast baseline and reveal whether relationships are roughly linear.
- Tree ensembles capture non-linear interactions (e.g. how test prep + parental education interact) without manual feature crosses.

**5. Hyperparameter tuning**
Each model is tuned with `GridSearchCV` (3-fold cross-validation) over a small, sensible parameter grid (e.g. `n_estimators`, `learning_rate`, `depth`) rather than trained with defaults — this avoids drawing conclusions from an unoptimized version of a model.

**6. Evaluation metric**
**R² score** is used because the goal is to explain variance in a continuous score, and R² is directly interpretable ("this model explains X% of the variation in math scores"). A minimum threshold (R² ≥ 0.6) is enforced before a model is accepted, so a poorly-fit model isn't silently deployed.

**7. Reproducibility & maintainability**
- The fitted preprocessor and final model are both serialized (`preprocessor.pkl`, `model.pkl`), so inference later uses the *exact* transformation learned during training — not a re-derived one.
- Custom logging (`logger.py`) and exception handling (`exception.py`, which reports the exact file and line number of a failure) make the pipeline debuggable rather than a black box.

**What I'd improve next:** add cross-validated confidence intervals around the R² estimate, log feature importances for the winning model, and add a held-out validation set separate from the test set used for model selection (to avoid subtly overfitting model *choice* to the test set).

## Tech Stack

- **Language:** Python
- **ML/Data:** scikit-learn, XGBoost, CatBoost, pandas, numpy
- **Visualization (EDA):** matplotlib, seaborn
- **Serialization:** dill

## Installation

```bash
git clone https://github.com/mayank-1584/first-end-to-end-ml-project.git
cd first-end-to-end-ml-project
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full training pipeline (ingestion → transformation → training):

```bash
python src/components/data_ingestion.py
```

This will:
1. Load and split the dataset
2. Fit the preprocessing pipeline
3. Train and evaluate multiple models
4. Save the best model and preprocessor to the `artifical/` folder

## Roadmap

- [ ] Complete `src/pipeline/train_pipeline.py` as a single clean entry point for training
- [ ] Complete `src/pipeline/predict_pipeline.py` for running inference on new data
- [ ] Add a simple web app (Flask/Streamlit) for interactive predictions
- [ ] Add unit tests

## Author

**Mayank**

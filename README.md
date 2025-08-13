APP Link - https://airbnbbamd-zhgxzpuwsytbgc2ueafw6p.streamlit.app/
# 🏡 Airbnb Demand & Listing Predictor — Streamlit App

This repository contains a Streamlit app with **two preloaded ML models** and a compact **EDA** workflow:

- **Model 1 (Listings model):** works with 8 fixed listing features and returns a prediction (the UI currently multiplies the raw prediction by `21` — see [Scaling / post-processing](#scaling--post-processing)).  
- **Model 2 (Demand model):** works with 8 fixed demand features and returns a demand prediction.

The app also **preloads a CSV** for a quick EDA and to **prefill** form inputs (medians/modes or selecting a row).

---

## ✨ What the app does

1. **Loads assets at startup**
   - Preloaded models from disk:
     - `MODEL_PATH = "classifier.pkl"` (Model 1)
     - `DEMAND_MODEL_PATH = "xgb_demand_model_final.pkl"` (Model 2 — Demand)
   - Preloaded dataset:
     - `CSV_PATH = "archive/train.csv"`
   - Optional renaming map for your CSV columns:
     - `FEATURE_RENAME_MAP = {}`

2. **Runs a compact EDA**
   - **Correlation heatmap** (Pearson/Spearman/Kendall) over numeric columns (booleans are cast to 0/1).
   - **Top Cities bar chart** with a slider to pick Top-N.
   - **Price distribution** histogram (+ KDE) if `price` column exists.
   - **Demand distribution** histogram if a `demand`-like column exists (tries: `demand`, `demand_target`, `demand_label`).

3. **Single‑row inference with two forms**
   - **App 1 (Listings model)** — fixed 8 features:
     ```text
     bathrooms, beds, review_score, essentials, kitchen, host_verified, is_superhost, city
     ```
   - **App 2 (Demand model)** — fixed 8 features:
     ```text
     reviews_per_month, number_of_reviews, host_is_superhost, cooking basics, availability_30, days_between_reviews, room_type_encoded, price
     ```
   - Each form supports **prefill**:
     - **Blank**
     - **Use medians/modes** (computed from CSV)
     - **Pick a row index** (read a row directly from the CSV)

---

## 🧠 Models & inputs

### Model 1 — Listings model
- **Path:** `classifier.pkl`
- **Inputs (8 fixed features):**
  - `bathrooms` *(numeric)*
  - `beds` *(numeric)*
  - `review_score` *(numeric)*
  - `essentials`, `kitchen`, `host_verified`, `is_superhost` *(booleans)*
  - `city` *(string)*
- **Prediction:** single value shown as a Streamlit metric.  
  See [Scaling / post-processing](#scaling--post-processing) for a note on the `×21` UI scaling.

### Model 2 — Demand model
- **Path:** `xgb_demand_model_final.pkl`
- **Inputs (8 fixed features):**
  - `reviews_per_month`, `number_of_reviews`, `availability_30`, `days_between_reviews`, `room_type_encoded`, `price` *(numeric)*
  - `host_is_superhost`, `cooking basics` *(booleans)*
- **Prediction:** single value shown as a Streamlit metric (no extra scaling in UI).

> **Pipelines recommended.** If your models were trained with preprocessing (imputers, encoders, one‑hot, etc.), save the entire **scikit-learn Pipeline** in the pickle so the app can pass raw features reliably.
>
> **Security:** Pickle files can execute arbitrary code on load. Only load files you trust.

---

## 🗂️ CSV expectations

- The app **loads the CSV at startup** and tries to coerce types:
  - Numerics → `pd.to_numeric(errors="coerce")`
  - Booleans (`essentials`, `kitchen`, `host_verified`, `is_superhost`, `host_is_superhost`, `cooking basics`) → parsed from strings like `"yes"/"no"`, `"true"/"false"`, `"1"/"0"`.
- If your column names differ, set them in `FEATURE_RENAME_MAP` or rename your CSV beforehand.
- **Optional columns** for EDA:
  - `price` (for price histogram)
  - `demand` / `demand_target` / `demand_label` (for demand histogram)
  - `city` (for top-cities table and bar chart)

---

## 🧮 Scaling / post-processing

- In `predict_and_show(...)` for **Model 1**, the displayed metric is **`raw_prediction * 21`**.  
  This mirrors your latest code. If you don’t intend this, remove the `* 21` multiplication in that function.

---

## ⚙️ How it works (under the hood)

- **Feature alignment**
  - If a model exposes `feature_names_in_` (or `get_feature_names_out()`), the app **builds a full‑schema row** in that order, filling any non‑used columns with `NaN`.
  - Otherwise, it sends **only** the fixed 8 columns.
- **Prefill logic**
  - **Medians/modes**: numeric columns use median; booleans use mode; `city` uses mode if present.
  - **Pick a row**: reads from the CSV and fills matching fields.
- **Charts**
  - Correlation heatmap (booleans cast to `float` = 0/1).  
  - Top N cities: `value_counts().head(N)` bar chart.  
  - Histograms via seaborn/matplotlib with adjustable bins for price and demand.

---

## 🚀 Run locally

1. **Create env & install deps**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Project files**
   - Put models at the paths specified in the app (or change them in code):
     ```text
     classifier.pkl
     xgb_demand_model_final.pkl
     archive/train.csv
     ```

3. **Launch**
   ```bash
   streamlit run app.py
   ```
   The app opens in your browser (usually http://localhost:8501).

### `requirements.txt`
Make sure your `requirements.txt` includes:
```txt
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn
```

### (Streamlit Cloud) `runtime.txt`
To avoid wheel issues on 3.13, pin Python:
```txt
3.12
```

---

## ☁️ Deploying & large files

- **Model file > 100 MB?** Use **Git LFS** or attach it as a **Release asset**.
  - **Git LFS (versioned)**:
    ```bash
    git lfs install
    git lfs track "*.pkl"
    git add .gitattributes
    git add classifier.pkl xgb_demand_model_final.pkl
    git commit -m "Add models via LFS"
    git push origin main
    ```
  - **Release asset (not versioned)**: create a GitHub Release and upload the model files there.

---

## 🧩 Configuration

Update these in the script or via `.streamlit/secrets.toml`:
```toml
MODEL_PATH = "classifier.pkl"
DEMAND_MODEL_PATH = "xgb_demand_model_final.pkl"
CSV_PATH = "archive/train.csv"
# Optional column renames for the CSV:
# FEATURE_RENAME_MAP = {"number_of_bathrooms": "bathrooms", "num_beds": "beds"}
```

---

## 🛠️ Troubleshooting

- **Module not found** at runtime → add it to `requirements.txt` and redeploy.
- **Pickle load error** → confirm model was saved with the same library versions (e.g., `xgboost`, `scikit-learn`) and that you’re on Python **3.12** (use `runtime.txt`).
- **Feature/shape mismatch** → ensure your pickled object is a **Pipeline** that includes preprocessing, or adapt `FEATURE_RENAME_MAP`/CSV columns to match training.
- **Model path not found** → verify file paths & that files are present in the repo (or LFS pulled on deploy).

---

## 📁 Suggested repo layout

```
.
├── app.py
├── requirements.txt
├── runtime.txt                 # (recommended for Streamlit Cloud)
├── classifier.pkl              # Model 1 (maybe via LFS)
├── xgb_demand_model_final.pkl  # Demand model (maybe via LFS)
├── archive/
│   └── train.csv               # Preloaded dataset
└── README.md
```

---

## ✅ License & credits

- This app uses Python, Streamlit, scikit‑learn, XGBoost, pandas, seaborn, and matplotlib.
- Data/model usage should comply with their respective licenses and your organization’s policies.

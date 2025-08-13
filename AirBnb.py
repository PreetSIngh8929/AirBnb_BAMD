import os
import pickle, joblib
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pickle Inference + EDA — 2 Preloaded Models", layout="wide")
st.title("AirBnb Price & Demand Prediction")

# ========= FIXED CONFIG =========
# You can override any of these via `.streamlit/secrets.toml`
# MODEL_PATH = st.secrets.get("MODEL_PATH", "classifier.pkl")
# DEMAND_MODEL_PATH = st.secrets.get("DEMAND_MODEL_PATH", "demand_model.pkl")
MODEL_PATH = "data/classifier.pkl"
CSV_PATH = "data/Train.csv"
DEMAND_MODEL_PATH = "data/xgb_demand_model_final.pkl"
FEATURE_RENAME_MAP = {}  # optional column rename for CSV

# App 1 — EXACT 8 features
FIXED_FEATURES = [
    "bathrooms", "beds", "review_score",
    "essentials", "kitchen", "host_verified",
    "is_superhost", "city",
]

# App 2 (Demand) — EXACT 8 features
DEMAND_FEATURES = [
    "reviews_per_month", "number_of_reviews", "host_is_superhost",
    "cooking basics", "availability_30", "days_between_reviews",
    "room_type_encoded", "price",
]
# =================================

# ---------- Utilities ----------
def coerce_bool(x):
    if pd.isna(x): return np.nan
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in {"1","true","t","yes","y","on"}: return True
    if s in {"0","false","f","no","n","off"}: return False
    return np.nan

def coerce_num(x):
    if x is None or (isinstance(x, str) and x.strip()==""): return np.nan
    try: return int(x)
    except Exception:
        try: return float(x)
        except Exception: return np.nan

def to_number(v):
    if v is None or (isinstance(v, str) and v.strip()==""): return np.nan
    if isinstance(v, (int,float,np.number)): return v
    return coerce_num(v)

@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        loaded = pickle.load(BytesIO(raw))
    except Exception:
        loaded = joblib.load(BytesIO(raw))
    model = getattr(loaded, "model", loaded)  # unwrap bundle if present
    meta = {
        "target_name": getattr(loaded, "target_name", None),
        "notes": getattr(loaded, "notes", None),
    }
    return model, meta

@st.cache_data(show_spinner=False)
def load_csv(path: str, rename_map: dict):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce dtypes for App 1 features
    for col in ["bathrooms","beds","review_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["essentials","kitchen","host_verified","is_superhost"]:
        if col in df.columns:
            df[col] = df[col].apply(coerce_bool)

    # Coerce dtypes for Demand features
    for col in ["reviews_per_month","number_of_reviews","availability_30","days_between_reviews","room_type_encoded","price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = df["host_is_superhost"].apply(coerce_bool)
    if "cooking basics" in df.columns:
        df["cooking basics"] = df["cooking basics"].apply(coerce_bool)

    return df

def infer_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    try:
        names = model.get_feature_names_out()
        return list(names) if names is not None else None
    except Exception:
        return None

def get_final_estimator(model):
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            return model.steps[-1][1]
    except Exception:
        pass
    return model

def detect_task(model):
    est = get_final_estimator(model)
    if hasattr(est, "predict_proba") or hasattr(est, "decision_function") or hasattr(est, "classes_"):
        return "classification", est.__class__.__name__
    return "regression", est.__class__.__name__

def build_X_generic(all_features, feature_list, row_fixed):
    if all_features:
        full = {f: np.nan for f in all_features}
        for f in feature_list:
            if f in full:
                full[f] = row_fixed.get(f, np.nan)
        return pd.DataFrame([full])[all_features]
    return pd.DataFrame([row_fixed])[feature_list]

# ---- Plot helpers ----
def corr_heatmap(df, method="pearson", include_bool=True, title="Correlation Heatmap"):
    data = df.copy()
    if include_bool:
        for col in data.select_dtypes(include=["boolean","bool"]).columns:
            data[col] = data[col].astype(float)
    num = data.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num.shape[1] < 2:
        st.info("Not enough numeric columns to compute correlation.")
        return
    corr = num.corr(method=method)
    fig, ax = plt.subplots(figsize=(min(12, 1.2*corr.shape[1]), min(10, 1.2*corr.shape[0])))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)

# ===== Load assets =====
left, right = st.columns([1.2, 1.4])

with left:
    # ---- Model 1 ----
    st.caption(f"📦 Loading Model 1 from: `{MODEL_PATH}`")
    try:
        with st.spinner("Loading Model 1..."):
            model1, meta1 = load_model_from_path(MODEL_PATH)
        st.success("Model 1 loaded.")
    except Exception as e:
        st.error(f"Failed to load Model 1: {e}")
        st.stop()

    task1, est_name1 = detect_task(model1)
    features1 = infer_feature_names(model1)
    info1 = f"Model 1 → Task: **{task1.capitalize()}** | Estimator: **{est_name1}**"
    if meta1.get("target_name"): info1 += f" | Target: **{meta1['target_name']}**"
    st.info(info1)
    if features1:
        st.caption(f"Model 1 expects **{len(features1)}** features.")
        missing = [f for f in FIXED_FEATURES if f not in features1]
        if missing:
            st.warning("App 1 inputs not in model schema will be ignored: " + ", ".join(missing))
    else:
        st.warning("Couldn't infer expected columns from Model 1. Ensure your pipeline can handle the provided 8 features.")

    # ---- Model 2 (Demand) ----
    st.caption(f"📦 Loading Demand Model from: `{DEMAND_MODEL_PATH}`")
    try:
        with st.spinner("Loading Demand Model..."):
            demand_model, meta2 = load_model_from_path(DEMAND_MODEL_PATH)
        st.success("Demand Model loaded.")
    except Exception as e:
        st.error(f"Failed to load Demand Model: {e}")
        demand_model, meta2 = None, {}

    if demand_model is not None:
        task2, est_name2 = detect_task(demand_model)
        features2 = infer_feature_names(demand_model)
        info2 = f"Demand Model → Task: **{task2.capitalize()}** | Estimator: **{est_name2}**"
        if meta2.get("target_name"): info2 += f" | Target: **{meta2['target_name']}**"
        st.info(info2)
        if features2:
            st.caption(f"Demand model expects **{len(features2)}** features.")
            missing2 = [f for f in DEMAND_FEATURES if f not in features2]
            if missing2:
                st.warning("Demand inputs not in model schema will be ignored: " + ", ".join(missing2))
        else:
            st.warning("Couldn't infer expected columns from the Demand model. It will receive only the 8 demand features below.")

with right:
    st.caption(f"📄 Loading dataset from: `{CSV_PATH}`")
    try:
        with st.spinner("Loading CSV..."):
            df = load_csv(CSV_PATH, FEATURE_RENAME_MAP)
        st.success(f"Dataset loaded. Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        df = None

# ====== EDA Section ======
st.header("Exploratory Data Analysis")
if df is not None:
    with st.expander("Preview (first 20 rows)", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.subheader("Schema")
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "missing": df.isna().sum().values
        })
        st.dataframe(schema, use_container_width=True, height=260)

    with c2:
        st.subheader("Numeric Summary")
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            st.dataframe(num.describe().T, use_container_width=True, height=260)
        else:
            st.info("No numeric columns")

    with c3:
        st.subheader("Feature Snapshots")
        # Top cities table
        if "city" in df.columns:
            top_city = df["city"].astype(str).value_counts(dropna=True).head(50).rename_axis("city").reset_index(name="count")
            st.markdown("**Top cities (table)**")
            st.dataframe(top_city.head(20), use_container_width=True, height=140)

    # === Only Heatmap: Correlation ===
    st.subheader("Heatmap")
    with st.expander("Correlation heatmap", expanded=True):
        corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0, key="corr_method")
        corr_df = df.copy()
        for col in corr_df.select_dtypes(include=["boolean","bool"]).columns:
            corr_df[col] = corr_df[col].astype(float)
        corr_heatmap(corr_df, method=corr_method, include_bool=True, title=f"Correlation Heatmap ({corr_method})")

    # === New: Visuals you asked for ===
    st.subheader(" Top Cities — Bar Chart")
    if "city" in df.columns:
        top_n = st.slider("Show top N cities", min_value=5, max_value=50, value=15, step=1)
        tc = df["city"].astype(str).value_counts(dropna=True).head(top_n).rename_axis("city").reset_index(name="count")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(tc))))
        sns.barplot(data=tc, y="city", x="count", ax=ax)
        ax.set_xlabel("Listings count")
        ax.set_ylabel("City")
        ax.set_title(f"Top {top_n} Cities by Count")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("`city` column not found for top-cities chart.")

    st.subheader("Price Distributions")
    dist_cols = st.columns(2)
    with dist_cols[0]:
        if "price" in df.columns:
            bins = st.slider("Price bins", min_value=10, max_value=200, value=50, step=10, key="price_bins")
            fig, ax = plt.subplots(figsize=(7,4))
            sns.histplot(df["price"].dropna(), bins=bins, kde=True, ax=ax)
            ax.set_title("Price Distribution")
            ax.set_xlabel("price")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("`price` column not found.")
    # with dist_cols[1]:
    #     # Show demand distribution only if present in CSV
    #     demand_col_candidates = ["demand", "demand_target", "demand_label"]
    #     dcol = next((c for c in demand_col_candidates if c in df.columns), None)
    #     if dcol:
    #         bins = st.slider("Demand bins", min_value=10, max_value=200, value=50, step=10, key="demand_bins")
    #         fig, ax = plt.subplots(figsize=(7,4))
    #         if pd.api.types.is_numeric_dtype(df[dcol]):
    #             sns.histplot(df[dcol].dropna(), bins=bins, kde=True, ax=ax)
    #             ax.set_xlabel(dcol)
    #         else:
    #             # categorical demand: bar plot of counts
    #             counts = df[dcol].astype(str).value_counts()
    #             sns.barplot(x=counts.values, y=counts.index, ax=ax)
    #             ax.set_xlabel("count")
    #             ax.set_ylabel(dcol)
    #         ax.set_title("Demand Distribution")
    #         st.pyplot(fig, clear_figure=True)
    #     else:
    #         st.info("No `demand` column found. If your demand column has a different name, map it via FEATURE_RENAME_MAP.")

# ====== Inference — App 1 (Preloaded Model) ======
st.header("Price Prediction")
row_prefill1 = {f: np.nan for f in FIXED_FEATURES}
city_options = sorted(df["city"].dropna().astype(str).unique().tolist()) if (df is not None and "city" in df.columns) else None

pref1a, pref1b = st.columns([1,1])
with pref1a:
    if df is not None:
        choice1 = st.radio("Prefill inputs", ["Blank", "Use medians/modes", "Pick a row index"], horizontal=True, index=1)
        if choice1 == "Use medians/modes":
            for c in ["bathrooms","beds","review_score"]:
                if c in df.columns: row_prefill1[c] = float(df[c].median(skipna=True))
            for c in ["essentials","kitchen","host_verified","is_superhost"]:
                if c in df.columns:
                    mode_val = df[c].mode(dropna=True)
                    row_prefill1[c] = bool(mode_val.iloc[0]) if not mode_val.empty else False
            if "city" in df.columns:
                mode_city = df["city"].mode(dropna=True)
                # row_prefill1["city"] = str(mode_city.iloc[0]) if not mode_val.empty else ""
        elif choice1 == "Pick a row index":
            idx = st.number_input("Row index", min_value=0, max_value=(len(df)-1 if df is not None else 0), step=1, value=0)
            if df is not None and 0 <= idx < len(df):
                for f in FIXED_FEATURES:
                    if f in df.columns:
                        row_prefill1[f] = df.loc[idx, f]
with pref1b:
    st.caption("If CSV columns differ, adjust `FEATURE_RENAME_MAP`.")

with st.form("single_form_app1"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bathrooms = st.number_input("bathrooms", min_value=0.0, step=1.0, format="%.3f",
                                    value=float(row_prefill1["bathrooms"]) if not pd.isna(row_prefill1["bathrooms"]) else 0.0)
        essentials = st.checkbox("essentials", value=bool(row_prefill1["essentials"]) if not pd.isna(row_prefill1["essentials"]) else False)
    with c2:
        beds = st.number_input("beds", min_value=0.0, step=1.0, format="%.3f",
                               value=float(row_prefill1["beds"]) if not pd.isna(row_prefill1["beds"]) else 0.0)
        kitchen = st.checkbox("kitchen", value=bool(row_prefill1["kitchen"]) if not pd.isna(row_prefill1["kitchen"]) else False)
    with c3:
        review_score = st.number_input("review_score", step=0.1, format="%.3f",
                                       value=float(row_prefill1["review_score"]) if not pd.isna(row_prefill1["review_score"]) else 0.0)
        host_verified = st.checkbox("host_verified", value=bool(row_prefill1["host_verified"]) if not pd.isna(row_prefill1["host_verified"]) else False)
    with c4:
        is_superhost = st.checkbox("is_superhost", value=bool(row_prefill1["is_superhost"]) if not pd.isna(row_prefill1["is_superhost"]) else False)
        if city_options:
            default_city = str(row_prefill1["city"]) if isinstance(row_prefill1["city"], str) and row_prefill1["city"] in city_options else city_options[0]
            city = st.selectbox("city", options=city_options, index=city_options.index(default_city) if default_city in city_options else 0)
        else:
            city = st.text_input("city", value=str(row_prefill1["city"]) if isinstance(row_prefill1["city"], str) else "", placeholder="e.g., Mumbai")
    submitted_app1 = st.form_submit_button("Predict")

def predict_and_show(model, task, X, label="Prediction", P=1):
    with st.spinner("Predicting..."):
        try:
            y_pred = model.predict(X)
            st.success("Prediction complete.")
            # NOTE: preserving your *21 scaling from previous code
            st.metric(label, str(y_pred[0] * P))
            est = get_final_estimator(model)
            if task == "classification" and hasattr(est, "predict_proba"):
                probs = est.predict_proba(X)
                st.dataframe(pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(probs.shape[1])]), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if submitted_app1:
    row1 = {
        "bathrooms": to_number(bathrooms),
        "beds": to_number(beds),
        "review_score": to_number(review_score),
        "essentials": bool(essentials),
        "kitchen": bool(kitchen),
        "host_verified": bool(host_verified),
        "is_superhost": bool(is_superhost),
        "city": city if isinstance(city, str) else str(city),
    }
    X1 = build_X_generic(features1, FIXED_FEATURES, row1)
    predict_and_show(model1, task1, X1, label="Model 1 Prediction",P=21)

st.divider()

# ====== Inference — App 2 (Preloaded Demand Model) ======
st.header("Demand Prediction")

if demand_model is None:
    st.warning("Demand model failed to load. Check DEMAND_MODEL_PATH.")
else:
    # Prefill for demand features
    row_prefill2 = {f: np.nan for f in DEMAND_FEATURES}
    if df is not None:
        choice2 = st.radio("Prefill demand inputs", ["Blank", "Use medians/modes", "Pick a row index"], horizontal=True, index=1)
        if choice2 == "Use medians/modes":
            for c in ["reviews_per_month","number_of_reviews","availability_30","days_between_reviews","room_type_encoded","price"]:
                if c in df.columns: row_prefill2[c] = float(df[c].median(skipna=True))
            for c in ["host_is_superhost","cooking basics"]:
                if c in df.columns:
                    mode_val = df[c].mode(dropna=True)
                    row_prefill2[c] = bool(mode_val.iloc[0]) if not mode_val.empty else False
        elif choice2 == "Pick a row index":
            idx2 = st.number_input("Row index (demand)", min_value=0, max_value=(len(df)-1 if df is not None else 0), step=1, value=0)
            if df is not None and 0 <= idx2 < len(df):
                for f in DEMAND_FEATURES:
                    if f in df.columns:
                        row_prefill2[f] = df.loc[idx2, f]

    with st.form("single_form_demand"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            reviews_per_month = st.number_input("reviews_per_month", step=0.1, format="%.3f",
                                                value=float(row_prefill2["reviews_per_month"]) if not pd.isna(row_prefill2["reviews_per_month"]) else 0.0)
            number_of_reviews = st.number_input("number_of_reviews", min_value=0.0, step=1.0, format="%.3f",
                                                value=float(row_prefill2["number_of_reviews"]) if not pd.isna(row_prefill2["number_of_reviews"]) else 0.0)
        with c2:
            host_is_superhost = st.checkbox("host_is_superhost",
                                            value=bool(row_prefill2["host_is_superhost"]) if not pd.isna(row_prefill2["host_is_superhost"]) else False)
            cooking_basics = st.checkbox("cooking basics",
                                         value=bool(row_prefill2["cooking basics"]) if not pd.isna(row_prefill2["cooking basics"]) else False)
        with c3:
            availability_30 = st.number_input("availability_30", min_value=0.0, step=1.0, format="%.3f",
                                              value=float(row_prefill2["availability_30"]) if not pd.isna(row_prefill2["availability_30"]) else 0.0)
            days_between_reviews = st.number_input("days_between_reviews", min_value=0.0, step=1.0, format="%.3f",
                                                   value=float(row_prefill2["days_between_reviews"]) if not pd.isna(row_prefill2["days_between_reviews"]) else 0.0)
        with c4:
            room_type_encoded = st.number_input("room_type_encoded", min_value=0.0, step=1.0, format="%.3f",
                                                value=float(row_prefill2["room_type_encoded"]) if not pd.isna(row_prefill2["room_type_encoded"]) else 0.0)
            price = st.number_input("price", min_value=0.0, step=1.0, format="%.3f",
                                    value=float(row_prefill2["price"]) if not pd.isna(row_prefill2["price"]) else 0.0)
        submitted_dm = st.form_submit_button("Predict")

    if submitted_dm:
        row2 = {
            "reviews_per_month": to_number(reviews_per_month),
            "number_of_reviews": to_number(number_of_reviews),
            "host_is_superhost": bool(host_is_superhost),
            "cooking basics": bool(cooking_basics),
            "availability_30": to_number(availability_30),
            "days_between_reviews": to_number(days_between_reviews),
            "room_type_encoded": to_number(room_type_encoded),
            "price": to_number(price),
        }
        features2 = infer_feature_names(demand_model)  # re-check in case
        task2, _ = detect_task(demand_model)
        X2 = build_X_generic(features2, DEMAND_FEATURES, row2)
        predict_and_show(demand_model, task2, X2, label="Demand Prediction",P=1)

st.divider()
st.caption(
    "Security tip: only load pickle files you trust. "
    "If models were trained on more than 8 inputs, keep imputers/encoders inside the pickled Pipelines so NaNs on non-used columns are handled."
)

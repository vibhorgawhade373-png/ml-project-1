"""
Autoscope — Car Price Intelligence
A Streamlit dashboard for the Linear / Ridge / Lasso car-price regression project.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Autoscope · Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRANDS = ["BMW", "Mercedes-Benz", "Audi", "Volkswagen", "Toyota", "Renault", "Mitsubishi"]
BODIES = ["sedan", "hatch", "crossover", "vagon", "van", "other"]
ENGINE_TYPES = ["Petrol", "Diesel", "Gas", "Other"]
REGISTRATION = ["yes", "no"]
REQUIRED_COLS = ["Brand", "Price", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year"]

# --------------------------------------------------------------------------------------
# STYLE — dashboard / cockpit theme
# --------------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root{
  --bg:        #0E1013;
  --panel:     #161920;
  --panel-2:   #1D2129;
  --line:      #2B3038;
  --ink:       #ECE9E2;
  --ink-dim:   #9CA1AC;
  --amber:     #F2A93B;
  --amber-dim: #7A5C25;
  --teal:      #3ED6B5;
  --red:       #E2604B;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

.stApp {
  background: radial-gradient(1200px 600px at 85% -10%, #1a1f2a 0%, var(--bg) 55%) fixed;
  color: var(--ink);
}

section[data-testid="stSidebar"] {
  background: #0B0D10;
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Hero */
.hero {
  padding: 2.1rem 2.4rem;
  border-radius: 4px;
  border: 1px solid var(--line);
  background: linear-gradient(120deg, #171B22 0%, #12151B 100%);
  position: relative;
  overflow: hidden;
  margin-bottom: 1.4rem;
}
.hero::after{
  content:"";
  position:absolute; right:-40px; top:-60px;
  width:220px; height:220px; border-radius:50%;
  background: radial-gradient(circle, rgba(242,169,59,0.16) 0%, rgba(242,169,59,0) 70%);
}
.hero .eyebrow { color: var(--teal); font-size: 0.8rem; font-weight: 600; }
.hero h1 { font-size: 2.3rem; margin: 0.2rem 0 0.4rem 0; color: var(--ink); }
.hero p { color: var(--ink-dim); max-width: 640px; font-size: 0.98rem; line-height: 1.5; margin: 0; }

/* Flat instrument panels (avoid uniform card-kit: left accent bar, square corners) */
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--amber);
  border-radius: 2px;
  padding: 1.0rem 1.2rem;
  height: 100%;
}
.panel.teal { border-left-color: var(--teal); }
.panel.red  { border-left-color: var(--red); }
.panel .label { color: var(--ink-dim); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.panel .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 600; color: var(--ink); margin-top: 0.15rem; }
.panel .sub   { color: var(--ink-dim); font-size: 0.8rem; margin-top: 0.2rem; }

.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.15rem;
  color: var(--ink);
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.4rem;
  margin: 1.6rem 0 0.9rem 0;
}

.divider { height:1px; background: var(--line); margin: 1.4rem 0; border:none; }

.stButton>button {
  background: var(--amber);
  color: #14100A;
  border: none;
  border-radius: 2px;
  font-weight: 600;
  padding: 0.6rem 1.4rem;
  font-family: 'Space Grotesk', sans-serif;
}
.stButton>button:hover { background: #ffbb55; color: #14100A; }

.result-box {
  background: linear-gradient(120deg, #171B22 0%, #12151B 100%);
  border: 1px solid var(--amber-dim);
  border-radius: 3px;
  padding: 1.6rem 2rem;
  text-align: center;
  margin-top: 1rem;
}
.result-box .tag { color: var(--teal); font-size: 0.8rem; letter-spacing: 0.05em; }
.result-box .amount { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; color: var(--amber); font-weight: 700; margin: 0.3rem 0; }
.result-box .range { color: var(--ink-dim); font-size: 0.9rem; }

[data-testid="stDataFrame"] { border: 1px solid var(--line); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#161920",
        plot_bgcolor="#161920",
        font=dict(color="#ECE9E2", family="Inter"),
        xaxis=dict(gridcolor="#2B3038", zerolinecolor="#2B3038"),
        yaxis=dict(gridcolor="#2B3038", zerolinecolor="#2B3038"),
        colorway=["#F2A93B", "#3ED6B5", "#E2604B", "#7C93C9", "#B98BD6"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

# --------------------------------------------------------------------------------------
# DATA — synthetic stand-in for "1.04. Real-life example.csv" (used-car dataset)
# Mirrors its real quirks: missing Price/EngineV, an EngineV=99.99 style data-entry error,
# and a right-skewed Price distribution that motivates the log transform.
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n=4200, seed=42):
    rng = np.random.default_rng(seed)

    brand_weights = [0.16, 0.14, 0.13, 0.22, 0.15, 0.11, 0.09]
    brand_mult = {"BMW": 1.15, "Mercedes-Benz": 1.22, "Audi": 1.12, "Volkswagen": 0.85,
                  "Toyota": 0.82, "Renault": 0.64, "Mitsubishi": 0.70}
    body_mult = {"sedan": 1.0, "hatch": 0.88, "crossover": 1.12, "vagon": 0.93, "van": 0.80, "other": 0.90}
    engine_mult = {"Petrol": 1.0, "Diesel": 1.05, "Gas": 0.86, "Other": 0.92}
    reg_mult = {"yes": 1.0, "no": 0.78}

    brand = rng.choice(BRANDS, size=n, p=brand_weights)
    body = rng.choice(BODIES, size=n)
    engine_type = rng.choice(ENGINE_TYPES, size=n, p=[0.42, 0.42, 0.11, 0.05])
    registration = rng.choice(REGISTRATION, size=n, p=[0.83, 0.17])
    year = rng.integers(1970, 2017, size=n)
    engine_v = np.clip(rng.normal(2.4, 1.0, size=n), 0.6, 6.5).round(1)

    years_old = 2017 - year
    mileage = np.clip(rng.normal(years_old * 9000, 25000, size=n), 0, None).round(0)

    base = 3200
    price = (
        base
        * np.array([brand_mult[b] for b in brand])
        * np.array([body_mult[b] for b in body])
        * np.array([engine_mult[e] for e in engine_type])
        * np.array([reg_mult[r] for r in registration])
        * (1 + 0.033 * (year - 1970))
        * np.clip(1 - mileage / 480000, 0.28, None)
        * (1 + 0.09 * engine_v)
    )
    noise = rng.normal(0, 0.16, size=n)
    price = price * np.exp(noise)
    price = price.round(0)

    model_names = [f"{b.split('-')[0]} {rng.integers(1, 9)}{chr(65 + rng.integers(0, 6))}" for b in brand]

    df = pd.DataFrame({
        "Brand": brand, "Price": price, "Body": body, "Mileage": mileage.astype(int),
        "EngineV": engine_v, "Engine Type": engine_type, "Registration": registration,
        "Year": year, "Model": model_names,
    })

    # Data-entry quirks, matching the source notebook's narrative
    miss_price_idx = rng.choice(n, size=int(n * 0.025), replace=False)
    df.loc[miss_price_idx, "Price"] = np.nan
    remaining = np.setdiff1d(np.arange(n), miss_price_idx)
    miss_ev_idx = rng.choice(remaining, size=int(n * 0.03), replace=False)
    df.loc[miss_ev_idx, "EngineV"] = np.nan

    outlier_idx = rng.choice(np.setdiff1d(remaining, miss_ev_idx), size=18, replace=False)
    df.loc[outlier_idx, "EngineV"] = rng.uniform(20, 99.99, size=18).round(2)

    return df


def load_raw_data(uploaded_file):
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            missing = [c for c in REQUIRED_COLS if c not in user_df.columns]
            if missing:
                st.sidebar.error(f"Missing columns: {', '.join(missing)}. Using sample dataset instead.")
                return generate_synthetic_dataset(), False
            return user_df, True
        except Exception as e:
            st.sidebar.error(f"Couldn't read file ({e}). Using sample dataset instead.")
            return generate_synthetic_dataset(), False
    return generate_synthetic_dataset(), False


# --------------------------------------------------------------------------------------
# PIPELINE — mirrors the notebook: dropna -> outlier filter -> log target -> encode
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(df):
    d = df.copy()
    d = d.dropna(subset=["Price", "EngineV"])
    d = d[d["EngineV"] <= 10]
    d["Log_price"] = np.log(d["Price"])
    return d


@st.cache_resource(show_spinner=False)
def build_features(df_hash_key, df):
    d = df.drop(columns=[c for c in ["Model", "Price"] if c in df.columns]).copy()
    encoders = {}
    cat_cols = d.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col])
        encoders[col] = le
    return d, encoders


@st.cache_resource(show_spinner=False)
def train_models(df_hash_key, features_df):
    X = features_df.drop(columns=["Log_price"])
    y = features_df["Log_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = {
            "model": m,
            "r2": r2_score(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "y_test": y_test,
            "pred": pred,
        }
    return results, X.columns.tolist()


# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚗 Autoscope")
    st.caption("Linear · Ridge · Lasso — used-car pricing")
    page = st.radio(
        "Navigate",
        ["Overview", "Explore the Data", "Model Lab", "Predict a Price"],
        label_visibility="collapsed",
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload your own CSV", type=["csv"], help=f"Needs columns: {', '.join(REQUIRED_COLS)}")
    st.caption("No file? A realistic sample dataset is used automatically.")

raw_df, is_user_data = load_raw_data(uploaded)
clean_df = clean_data(raw_df)
features_df, encoders = build_features("v1" if not is_user_data else "user", clean_df)
results, feature_order = train_models("v1" if not is_user_data else "user", features_df)

best_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]

# --------------------------------------------------------------------------------------
# PAGE: OVERVIEW
# --------------------------------------------------------------------------------------
if page == "Overview":
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">USED-CAR PRICING MODEL</div>
      <h1>Autoscope</h1>
      <p>A regression dashboard that cleans a messy used-car dataset, compares Linear, Ridge and Lasso
      regression, and turns the winner into an instant price estimator. Log-price transform in,
      real-world dollars out.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="panel"><div class="label">Rows after cleaning</div>
        <div class="value">{len(clean_df):,}</div><div class="sub">of {len(raw_df):,} raw rows</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="panel teal"><div class="label">Best model</div>
        <div class="value" style="font-size:1.3rem;">{best_name}</div><div class="sub">by R² on test set</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="panel"><div class="label">Best R²</div>
        <div class="value">{results[best_name]['r2']:.3f}</div><div class="sub">variance explained</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="panel red"><div class="label">RMSE (log price)</div>
        <div class="value">{results[best_name]['rmse']:.3f}</div><div class="sub">lower is better</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model performance at a glance</div>', unsafe_allow_html=True)
    g1, g2 = st.columns([1, 1.4])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results[best_name]["r2"] * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#F2A93B"}},
            title={"text": f"{best_name} — R² Score", "font": {"size": 14, "color": "#9CA1AC"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#2B3038"},
                "bar": {"color": "#F2A93B"},
                "bgcolor": "#1D2129",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#20242C"},
                    {"range": [50, 80], "color": "#262B34"},
                ],
            },
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=280)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        comp = pd.DataFrame({
            "Model": list(results.keys()),
            "R²": [results[k]["r2"] for k in results],
        })
        fig2 = px.bar(comp, x="Model", y="R²", text_auto=".3f", template=PLOTLY_TEMPLATE, title="R² by model")
        fig2.update_traces(marker_color=["#F2A93B" if k == best_name else "#3ED6B5" for k in results])
        fig2.update_layout(height=280, yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Raw sample</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(12), use_container_width=True, height=280)
    if not is_user_data:
        st.caption("This is a synthetic dataset generated to resemble the used-car pricing data (same schema, quirks and relationships as the source notebook). Upload your own CSV in the sidebar to use real data.")

# --------------------------------------------------------------------------------------
# PAGE: EXPLORE THE DATA
# --------------------------------------------------------------------------------------
elif page == "Explore the Data":
    st.markdown('<div class="section-title">Why Price gets log-transformed</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(clean_df, x="Price", nbins=50, template=PLOTLY_TEMPLATE, title="Price — right-skewed")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(clean_df, x="Log_price", nbins=50, template=PLOTLY_TEMPLATE, title="Log(Price) — closer to normal", color_discrete_sequence=["#3ED6B5"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Cleaning EngineV outliers</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(raw_df, y="EngineV", template=PLOTLY_TEMPLATE, title="Before — up to ~99.99L 'engines'", color_discrete_sequence=["#E2604B"])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(clean_df, y="EngineV", template=PLOTLY_TEMPLATE, title="After — capped at 10L", color_discrete_sequence=["#3ED6B5"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Feature relationships</div>', unsafe_allow_html=True)
    corr = features_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", template=PLOTLY_TEMPLATE, color_continuous_scale=["#0E1013", "#1D2129", "#F2A93B"], title="Correlation heatmap (encoded features)")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.bar(clean_df["Brand"].value_counts().reset_index(), x="Brand", y="count", template=PLOTLY_TEMPLATE, title="Cars by brand")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.scatter(clean_df, x="Mileage", y="Price", color="Brand", template=PLOTLY_TEMPLATE, title="Mileage vs. price", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# PAGE: MODEL LAB
# --------------------------------------------------------------------------------------
elif page == "Model Lab":
    st.markdown('<div class="section-title">Linear vs. Ridge vs. Lasso</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for col, name in zip(cols, results):
        marker = " 🏆" if name == best_name else ""
        with col:
            st.markdown(f"""<div class="panel {'teal' if name==best_name else ''}">
            <div class="label">{name}{marker}</div>
            <div class="value">{results[name]['r2']:.3f}</div>
            <div class="sub">RMSE (log): {results[name]['rmse']:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Actual vs. predicted (log price)</div>', unsafe_allow_html=True)
    pick = st.selectbox("Model to inspect", list(results.keys()), index=list(results.keys()).index(best_name))
    r = results[pick]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["y_test"], y=r["pred"], mode="markers", marker=dict(color="#3ED6B5", opacity=0.6), name="Predictions"))
    lo, hi = r["y_test"].min(), r["y_test"].max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#F2A93B", dash="dash"), name="Perfect prediction"))
    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Actual log price", yaxis_title="Predicted log price", height=420, title=f"{pick} — test set")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Feature coefficients</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({name: results[name]["model"].coef_ for name in results}, index=feature_order)
    fig = px.bar(coef_df, barmode="group", template=PLOTLY_TEMPLATE, title="How each feature moves log price")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lasso can shrink coefficients to exactly zero — that's automatic feature selection. Ridge shrinks them but rarely to zero.")

# --------------------------------------------------------------------------------------
# PAGE: PREDICT A PRICE
# --------------------------------------------------------------------------------------
elif page == "Predict a Price":
    st.markdown('<div class="section-title">Estimate a car\'s price</div>', unsafe_allow_html=True)
    st.caption(f"Powered by **{best_name}** (highest R² on held-out data: {results[best_name]['r2']:.3f})")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            brand_opts = list(encoders["Brand"].classes_) if "Brand" in encoders else BRANDS
            in_brand = st.selectbox("Brand", brand_opts)
            body_opts = list(encoders["Body"].classes_) if "Body" in encoders else BODIES
            in_body = st.selectbox("Body type", body_opts)
        with c2:
            engine_opts = list(encoders["Engine Type"].classes_) if "Engine Type" in encoders else ENGINE_TYPES
            in_engine_type = st.selectbox("Engine type", engine_opts)
            reg_opts = list(encoders["Registration"].classes_) if "Registration" in encoders else REGISTRATION
            in_registration = st.selectbox("Registered", reg_opts)
        with c3:
            in_year = st.slider("Year", 1970, 2020, 2012)
            in_engine_v = st.slider("Engine volume (L)", 0.6, 6.5, 2.0, 0.1)

        in_mileage = st.slider("Mileage (km)", 0, 400000, 90000, 1000)
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        row = {}
        for col in feature_order:
            if col == "Brand":
                row[col] = in_brand
            elif col == "Body":
                row[col] = in_body
            elif col == "Engine Type":
                row[col] = in_engine_type
            elif col == "Registration":
                row[col] = in_registration
            elif col == "Year":
                row[col] = in_year
            elif col == "EngineV":
                row[col] = in_engine_v
            elif col == "Mileage":
                row[col] = in_mileage
            else:
                row[col] = clean_df[col].median() if col in clean_df.columns else 0

        input_df = pd.DataFrame([row])
        for col, le in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    input_df[col] = 0
        input_df = input_df[feature_order]

        log_pred = best_model.predict(input_df)[0]
        price_pred = float(np.exp(log_pred))
        rmse_log = results[best_name]["rmse"]
        low = np.exp(log_pred - rmse_log)
        high = np.exp(log_pred + rmse_log)

        st.markdown(f"""
        <div class="result-box">
          <div class="tag">ESTIMATED MARKET PRICE</div>
          <div class="amount">${price_pred:,.0f}</div>
          <div class="range">Likely range: ${low:,.0f} – ${high:,.0f} (±1 RMSE)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Where this car sits</div>', unsafe_allow_html=True)
        fig = px.histogram(clean_df, x="Price", nbins=50, template=PLOTLY_TEMPLATE, title="Price distribution — your estimate marked")
        fig.add_vline(x=price_pred, line_color="#F2A93B", line_width=3, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

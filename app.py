"""
app.py  -  EstateIQ: Smart House Price Prediction
Streamlit website with 4 pages: Dashboard, Predict Price, Model Analytics, About Project.
The model is trained by train_model.py (saved in artifacts/). If the saved file is
missing, the website trains the model automatically on first start.

Run:  streamlit run app.py
"""

import base64
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import ARTIFACT_PATH, DOLLARS_PER_UNIT, build_bundle, save_bundle

st.set_page_config(page_title="EstateIQ | House Price Prediction", page_icon="🏠", layout="wide")

ASSETS = Path(__file__).resolve().parent / "assets"
# Optional online hero photo. If it fails to load, the local illustration is shown instead.
HERO_URL = "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=1600&q=70"
PAGES = {
    "Dashboard": ":material/space_dashboard:",
    "Predict Price": ":material/real_estate_agent:",
    "Model Analytics": ":material/insights:",
    "About Project": ":material/info:",
}
# Approximate rupee rate (RBI-intervention close on 16 Oct 2026). The user can edit it on the Predict page.
DEFAULT_USD_INR = 94.92
RATE_DATE = "16 Oct 2026"
NAVY, TEAL, EMERALD = "#0B1F3A", "#0F9D8A", "#10B981"

# Text for each model input: label, helper text, widget type, step, display format
FIELDS = {
    "Latitude": ("Latitude", "North-south position (California: about 32.5 to 42.0)", "number", 0.01, "%.2f"),
    "Longitude": ("Longitude", "East-west position (California: about -124.3 to -114.3)", "number", 0.01, "%.2f"),
    "HouseAge": ("Median house age (years)", "How old the houses in the area are", "slider", 1, "%d"),
    "AveRooms": ("Average rooms per household", "Total rooms divided by households", "number", 0.1, "%.2f"),
    "AveBedrms": ("Average bedrooms per household", "Must not be more than the average rooms", "number", 0.05, "%.2f"),
    "MedInc": ("Median income (x $10,000)", "Example: 3.5 means $35,000 per year", "slider", 0.1, "%.1f"),
    "Population": ("Block population", "Number of people living in the area", "number", 10.0, "%.0f"),
    "AveOccup": ("Average occupancy", "Average number of people per household", "number", 0.1, "%.2f"),
}
SECTIONS = [
    ("Location Details", ["Latitude", "Longitude"]),
    ("Property Characteristics", ["HouseAge", "AveRooms", "AveBedrms"]),
    ("Household Information", ["MedInc", "Population", "AveOccup"]),
]


# ------------------------------------------------------------------ styling
def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def hero_background():
    """CSS layers: dark overlay, local photo, online photo, local illustration, plain gradient.
    If an image fails to load, the next layer simply shows through."""
    layers = ["linear-gradient(90deg, rgba(11,31,58,.90) 0%, rgba(11,31,58,.60) 55%, rgba(11,31,58,.25) 100%)"]
    for name, mime in [("hero.jpg", "image/jpeg"), ("hero.jpeg", "image/jpeg"), ("hero.png", "image/png"), ("hero.webp", "image/webp")]:
        if (ASSETS / name).exists():  # your own photo: put it in assets/ named hero.jpg
            layers.append(f'url("{data_uri(ASSETS / name, mime)}")')
            break
    layers.append(f'url("{HERO_URL}")')
    if (ASSETS / "hero.svg").exists():
        layers.append(f'url("{data_uri(ASSETS / "hero.svg", "image/svg+xml")}")')
    layers.append("linear-gradient(135deg, #0B1F3A, #0F9D8A)")
    return ", ".join(layers)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Serif+Display&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif; }
.stApp { background: #F7F9FB; }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { color: #0B1F3A !important; font-weight: 700 !important; letter-spacing: -0.01em; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0B1F3A; }
[data-testid="stSidebar"] * { color: #E6EDF5; }
[data-testid="stSidebar"] .brand { display:flex; align-items:center; gap:12px; margin: 6px 0 4px 0; }
[data-testid="stSidebar"] .brand-icon { width:44px; height:44px; border-radius:12px; background:#0F9D8A; display:flex; align-items:center; justify-content:center; }
[data-testid="stSidebar"] .brand-name { font-size:1.35rem; font-weight:800; color:#fff; line-height:1.1; }
[data-testid="stSidebar"] .brand-sub { font-size:0.78rem; color:#9FB3C8; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
[data-testid="stSidebar"] .stButton button { width:100%; justify-content:flex-start; border-radius:10px; border:none;
  background:transparent; color:#C9D6E4; font-weight:600; padding:0.6rem 0.9rem; transition: all .2s ease; }
[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.08); color:#fff; }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] { background:#0F9D8A; color:#fff; }
.status { display:flex; align-items:center; gap:8px; font-size:0.82rem; background:rgba(255,255,255,0.06); padding:10px 12px; border-radius:10px; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.dot.ok { background:#34D399; box-shadow:0 0 0 3px rgba(52,211,153,.25); }
.dot.bad { background:#F87171; box-shadow:0 0 0 3px rgba(248,113,113,.25); }

/* Hero */
.hero { background-size: cover; background-position: center; border-radius: 22px; padding: 64px 48px; min-height: 300px;
  box-shadow: 0 12px 32px rgba(11,31,58,.18); margin-bottom: 1.4rem; }
.hero h1 { color:#fff !important; font-family:'DM Serif Display', Georgia, serif; font-weight:400 !important; font-size:2.7rem; line-height:1.15; max-width:640px; margin:0 0 14px 0; }
.hero p { color:#D6E2EE; font-size:1.05rem; max-width:560px; margin:0; }
.hero .tag { display:inline-block; background:rgba(15,157,138,.25); color:#7CE8D3; border:1px solid rgba(124,232,211,.4);
  border-radius:999px; padding:4px 12px; font-size:0.78rem; font-weight:600; margin-bottom:16px; }

/* Cards */
.grid { display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin: 0 0 1.6rem 0; }
.card, .stat, .step { background:#fff; border:1px solid #E5EBF2; border-radius:18px; padding:20px 22px;
  box-shadow: 0 2px 10px rgba(11,31,58,.05); transition: transform .2s ease, box-shadow .2s ease; }
.card:hover, .stat:hover, .step:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(11,31,58,.10); }
.stat-label { color:#6B7C93; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.stat-value { color:#0B1F3A; font-size:1.8rem; font-weight:800; margin:6px 0 2px 0; }
.stat-note { color:#0F9D8A; font-size:0.8rem; font-weight:600; }
.step-num { width:34px; height:34px; border-radius:50%; background:#E6F6F3; color:#0F9D8A; font-weight:800; display:flex; align-items:center; justify-content:center; margin-bottom:12px; }
.step h4, .card h4 { margin: 0 0 6px 0; font-size:1.05rem; }
.step p, .card p, .card li { color:#51627A; font-size:0.93rem; line-height:1.55; margin:0; }
.card ul { padding-left: 18px; margin: 6px 0 0 0; }
.section-title { font-size:1.35rem; font-weight:700; color:#0B1F3A; margin: 1.2rem 0 0.8rem 0; }
.group-title { font-size:0.95rem; font-weight:700; color:#0B1F3A; margin: 14px 0 2px 0; padding-bottom:6px; border-bottom:2px solid #E6F6F3; }
.helper { color:#7A8BA1; font-size:0.78rem; margin: -10px 0 10px 0; }

/* Result card */
.result { background: linear-gradient(160deg, #0B1F3A 0%, #12506B 100%); color:#fff; border-radius:22px; padding:30px 28px;
  box-shadow: 0 14px 34px rgba(11,31,58,.22); }
.result .label { color:#9FE9DA; font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.result .price { font-family:'DM Serif Display', Georgia, serif; font-size:3.2rem; margin:8px 0 4px 0; line-height:1.1; }
.result .inr { color:#9FE9DA; font-size:1.25rem; font-weight:600; margin-bottom:6px; }
.result .inr small { color:#C7D5E3; font-weight:500; font-size:0.85rem; }
.result .note { color:#C7D5E3; font-size:0.85rem; line-height:1.5; }
.result .chips { display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:18px; }
.chip { background:rgba(255,255,255,0.08); border-radius:10px; padding:8px 12px; }
.chip span { display:block; color:#9FB3C8; font-size:0.7rem; }
.chip b { color:#fff; font-size:0.95rem; }
.placeholder { background:#fff; border:2px dashed #C9D6E4; border-radius:22px; padding:48px 28px; text-align:center; color:#6B7C93; }
.placeholder b { color:#0B1F3A; font-size:1.1rem; }
.disclaimer { background:#FFF8E6; border:1px solid #F6E2A8; color:#7A5B00; border-radius:12px; padding:10px 14px; font-size:0.84rem; margin-top:14px; }

/* Timeline (About page) */
.timeline { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin-bottom:1.4rem; }
.tl-item { background:#fff; border:1px solid #E5EBF2; border-top:4px solid #0F9D8A; border-radius:14px; padding:14px 16px; }
.tl-item b { color:#0B1F3A; display:block; font-size:0.92rem; }
.tl-item span { color:#6B7C93; font-size:0.8rem; }
table.info { width:100%; border-collapse:collapse; font-size:0.9rem; }
table.info th { text-align:left; color:#6B7C93; font-size:0.74rem; text-transform:uppercase; letter-spacing:.05em; padding:8px 10px; border-bottom:2px solid #E6F6F3; }
table.info td { padding:8px 10px; border-bottom:1px solid #EEF2F6; color:#32445C; }

/* Inputs, buttons, forms */
[data-testid="stForm"] { background:#fff; border:1px solid #E5EBF2; border-radius:20px; padding:10px 22px 18px 22px; box-shadow: 0 2px 10px rgba(11,31,58,.05); }
[data-baseweb="input"], [data-baseweb="base-input"] { border-radius:10px !important; }
.stNumberInput input { font-weight:600; color:#0B1F3A; }
.stButton button, .stFormSubmitButton button { border-radius:12px; font-weight:700; transition: all .2s ease; }
.stFormSubmitButton button { width:100%; padding:0.7rem 1rem; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #0F9D8A, #10B981); border:none; color:#fff; box-shadow: 0 6px 16px rgba(15,157,138,.30); }
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  transform: translateY(-2px); box-shadow: 0 10px 22px rgba(15,157,138,.38); color:#fff; }
.st-key-cta button { padding: 0.85rem 2.2rem; font-size: 1.05rem; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #E5EBF2; border-radius:18px; padding:8px; box-shadow: 0 2px 10px rgba(11,31,58,.05); }

@media (max-width: 768px) {
  .hero { padding: 36px 24px; min-height: 240px; }
  .hero h1 { font-size: 1.9rem; }
  .result .price { font-size: 2.4rem; }
}
"""


def html(markup):
    """Render HTML (lines joined so Markdown does not treat indentation as code)."""
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def show_chart(fig):
    """Display a Plotly chart full-width (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def money(value):
    return f"${value:,.0f}"


def inr(value):
    """Format a number in rupees with Indian digit grouping, e.g. Rs 1,67,00,000."""
    digits = f"{int(round(value))}"
    if len(digits) > 3:
        head, tail, groups = digits[:-3], digits[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return "\u20b9" + digits


def inr_words(value):
    """Short form: Crore / Lakh."""
    if value >= 1e7:
        return f"\u20b9{value / 1e7:.2f} Crore"
    if value >= 1e5:
        return f"\u20b9{value / 1e5:.2f} Lakh"
    return inr(value)


# ------------------------------------------------------------------ model loading
@st.cache_resource(show_spinner="Loading the trained model...")
def get_bundle():
    """Load artifacts/house_model.joblib; if missing or unreadable, train and save it."""
    if ARTIFACT_PATH.exists():
        try:
            return joblib.load(ARTIFACT_PATH)
        except Exception:
            pass  # damaged or incompatible file -> train again below
    bundle = build_bundle()
    save_bundle(bundle)
    return bundle


# ------------------------------------------------------------------ navigation
def goto(page):
    st.session_state.page = page


def sidebar(loaded, bundle):
    with st.sidebar:
        html("""
        <div class="brand">
          <div class="brand-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/><path d="M10 20v-6h4v6"/>
            </svg>
          </div>
          <div><div class="brand-name">EstateIQ</div><div class="brand-sub">Smart House Price Prediction</div></div>
        </div>
        """)
        st.divider()
        for name, icon in PAGES.items():
            active = st.session_state.page == name
            st.button(name, icon=icon, key=f"nav_{name}", on_click=goto, args=(name,),
                      type="primary" if active else "secondary")
        st.divider()
        if loaded:
            html(f'<div class="status"><span class="dot ok"></span>Model loaded<br>{bundle["model_name"]}</div>')
        else:
            html('<div class="status"><span class="dot bad"></span>Model not loaded</div>')


# ------------------------------------------------------------------ pages
def page_dashboard(b):
    m = b["metrics"]
    html(f"""
    <div class="hero" style="background-image: {hero_background()};">
      <span class="tag">Machine Learning Powered</span>
      <h1>Find the Estimated Value of Your Next Home</h1>
      <p>Explore property characteristics and generate a machine-learning-based house price estimate.</p>
    </div>
    """)
    html(f"""
    <div class="grid">
      <div class="stat"><div class="stat-label">Model Type</div><div class="stat-value" style="font-size:1.35rem">{b["model_name"]}</div><div class="stat-note">Scikit-learn</div></div>
      <div class="stat"><div class="stat-label">Dataset Records</div><div class="stat-value">{b["n_records"]:,}</div><div class="stat-note">California Housing</div></div>
      <div class="stat"><div class="stat-label">Test-set R&sup2; Score</div><div class="stat-value">{m["r2"]:.3f}</div><div class="stat-note">on {b["n_test"]:,} unseen records</div></div>
      <div class="stat"><div class="stat-label">Mean Absolute Error</div><div class="stat-value">{money(m["mae"] * DOLLARS_PER_UNIT)}</div><div class="stat-note">average prediction error</div></div>
    </div>
    <div class="section-title">How It Works</div>
    <div class="grid">
      <div class="step"><div class="step-num">1</div><h4>Enter property details</h4><p>Fill in location, property and household information for the area.</p></div>
      <div class="step"><div class="step-num">2</div><h4>Let the ML model analyze</h4><p>A trained Linear Regression model processes the eight inputs.</p></div>
      <div class="step"><div class="step-num">3</div><h4>View the estimated value</h4><p>See the predicted price in US dollars on a clear result card.</p></div>
    </div>
    """)
    with st.container(key="cta"):
        st.button("Predict House Price", type="primary", on_click=goto, args=("Predict Price",), key="cta_btn")
    html('<div class="disclaimer">Trained on a <b>US (California) housing dataset</b>. Prices are in US dollars and do not represent Indian market prices.</div>')


def default_value(name, ranges):
    """Default for each input = the dataset median (as the right type for its widget)."""
    r = ranges[name]
    if FIELDS[name][2] == "slider" and FIELDS[name][3] == 1:
        return int(round(r["median"]))
    return float(round(r["median"], 2))


def reset_inputs(ranges):
    """Callback for 'Make another prediction': clear the result and restore default inputs."""
    st.session_state.result = None
    for name in FIELDS:
        st.session_state[f"in_{name}"] = default_value(name, ranges)


def input_field(name, ranges):
    """Draw one input (slider or number box) with a short helper text; return its value."""
    label, helper, kind, step, fmt = FIELDS[name]
    r = ranges[name]
    key = f"in_{name}"
    if kind == "slider" and step == 1:
        value = st.slider(label, int(r["min"]), int(r["max"]), step=1, key=key)
    elif kind == "slider":
        value = st.slider(label, float(r["min"]), float(r["max"]), step=float(step), format=fmt, key=key)
    else:
        value = st.number_input(label, min_value=float(r["min"]), max_value=float(r["max"]),
                                step=float(step), format=fmt, key=key)
    html(f'<div class="helper">{helper}</div>')
    return value


def validate(values, ranges):
    """Return a list of error messages for invalid inputs."""
    errors = []
    for name, v in values.items():
        r = ranges[name]
        if not r["min"] <= v <= r["max"]:
            errors.append(f"{FIELDS[name][0]} must be between {r['min']:g} and {r['max']:g}.")
    if values["AveBedrms"] > values["AveRooms"]:
        errors.append("Average bedrooms cannot be greater than average rooms.")
    return errors


def page_predict(b):
    model, features, ranges = b["model"], b["feature_names"], b["ranges"]
    st.markdown("## Predict House Price")
    st.caption("Enter the details of a California neighbourhood and click Estimate Price.")
    for name in features:  # first visit: put the default values in session state
        st.session_state.setdefault(f"in_{name}", default_value(name, ranges))
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("usd_inr", DEFAULT_USD_INR)

    left, right = st.columns([1.15, 1], gap="large")
    error_box = []

    with left:
        with st.form("predict_form"):
            values = {}
            for title, names in SECTIONS:
                html(f'<div class="group-title">{title}</div>')
                cols = st.columns(2) if len(names) == 2 else [st.container()] * len(names)
                for col, name in zip(cols, names):
                    with col:
                        values[name] = input_field(name, ranges)
            submitted = st.form_submit_button("Estimate Price", type="primary")

    if submitted:
        error_box = validate(values, ranges)
        if not error_box:
            try:
                with st.spinner("Estimating property value..."):
                    row = pd.DataFrame([[values[f] for f in features]], columns=features)  # exact feature order
                    price = float(model.predict(row)[0]) * DOLLARS_PER_UNIT
                if price <= 0:
                    error_box = ["These inputs look unrealistic, so the model could not give a sensible price. Please adjust them."]
                    st.session_state.result = None
                else:
                    unusual = [FIELDS[f][0] for f in features if not ranges[f]["p1"] <= values[f] <= ranges[f]["p99"]]
                    st.session_state.result = {"price": price, "values": values, "unusual": unusual}
            except Exception as error:
                error_box = [f"Prediction failed: {error}"]
                st.session_state.result = None
        else:
            st.session_state.result = None

    with right:
        for message in error_box:
            st.error(message)
        result = st.session_state.result
        if result is None:
            html("""
            <div class="placeholder"><b>Your estimate will appear here</b><br><br>
            Fill in the property details and click <b>Estimate Price</b>.</div>
            """)
        else:
            chips = "".join(
                f'<div class="chip"><span>{FIELDS[f][0]}</span><b>{result["values"][f]:,.2f}</b></div>' for f in features
            )
            mae = b["metrics"]["mae"] * DOLLARS_PER_UNIT
            rate = float(st.session_state.usd_inr)
            rupees = result["price"] * rate
            html(f"""
            <div class="result">
              <div class="label">Estimated Property Value</div>
              <div class="price">{money(result["price"])}</div>
              <div class="inr">&asymp; {inr(rupees)} <small>({inr_words(rupees)} at &#8377;{rate:,.2f} per US$1)</small></div>
              <div class="note">Machine-learning estimate in US dollars; the rupee figure is only a currency conversion. On test data, predictions were off by
              about {money(mae)} on average, so this is not a guaranteed selling price.</div>
              <div class="chips">{chips}</div>
            </div>
            """)
            if result["unusual"]:
                st.info("Unusual values compared with the training data: " + ", ".join(result["unusual"])
                        + ". The estimate may be less reliable.")
            html('<div class="disclaimer">Based on a <b>US (California) dataset</b>. The rupee value is the US price converted at the rate below, '
                 '<b>not an estimate of Indian market prices</b>.</div>')
            st.number_input("Exchange rate (\u20b9 per US$1)", min_value=50.0, max_value=200.0, step=0.1, format="%.2f",
                            key="usd_inr", help=f"Default: rupee close of {RATE_DATE}. Exchange rates change daily, so enter today's rate.")
            st.button("Make another prediction", on_click=reset_inputs, args=(ranges,), key="reset_btn")


def chart_layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY)), height=height, template="plotly_white",
        font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", color="#32445C"),
        margin=dict(l=50, r=20, t=60, b=50), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def page_analytics(b):
    m = b["metrics"]
    st.markdown("## Model Analytics")
    st.caption("All values come from the trained model and its unseen test data.")
    html(f"""
    <div class="grid">
      <div class="stat"><div class="stat-label">MAE</div><div class="stat-value">{money(m["mae"] * DOLLARS_PER_UNIT)}</div><div class="stat-note">average error</div></div>
      <div class="stat"><div class="stat-label">RMSE</div><div class="stat-value">{money(m["rmse"] * DOLLARS_PER_UNIT)}</div><div class="stat-note">punishes big errors more</div></div>
      <div class="stat"><div class="stat-label">R&sup2; Score</div><div class="stat-value">{m["r2"]:.3f}</div><div class="stat-note">1.0 = perfect</div></div>
    </div>
    """)

    c1, c2 = st.columns(2, gap="medium")
    with c1:  # Actual vs predicted (max 1,500 points to keep the page fast)
        n = min(1500, len(b["y_test"]))
        sample = pd.DataFrame({"Actual": b["y_test"][:n] * DOLLARS_PER_UNIT, "Predicted": b["y_pred"][:n] * DOLLARS_PER_UNIT})
        fig = px.scatter(sample, x="Actual", y="Predicted", opacity=0.45, color_discrete_sequence=[TEAL])
        top = float(max(sample.max()))
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect prediction",
                                 line=dict(color=NAVY, dash="dash")))
        fig.update_traces(hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>", selector=dict(mode="markers"))
        fig.update_xaxes(title="Actual price (USD)", tickprefix="$")
        fig.update_yaxes(title="Predicted price (USD)", tickprefix="$")
        fig.data[0].name = "Test houses"
        fig.data[0].showlegend = True
        show_chart(chart_layout(fig, f"Actual vs Predicted Prices ({n:,} test records)"))
    with c2:
        fig = px.histogram(x=b["y_all"] * DOLLARS_PER_UNIT, nbins=40, color_discrete_sequence=[TEAL])
        fig.update_traces(hovertemplate="Price range: %{x}<br>Houses: %{y}<extra></extra>", name="Houses", showlegend=True)
        fig.update_xaxes(title="Median house value (USD)", tickprefix="$")
        fig.update_yaxes(title="Number of areas")
        show_chart(chart_layout(fig, "Distribution of House Values"))
    st.caption("The tall bar at the far right appears because the original data caps prices at about $500,000.")

    corr = b["corr"].rename(index={"MedHouseVal": "House Value"}, columns={"MedHouseVal": "House Value"})
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale=[[0, "#1D4E89"], [0.5, "#FFFFFF"], [1, TEAL]])
    fig.update_traces(hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>")
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    show_chart(chart_layout(fig, "Feature Correlation Heatmap", height=520))
    st.caption("Values near +1 or -1 mean a strong relationship. Median income is the feature most linked to house value.")


def page_about(b):
    m = b["metrics"]
    st.markdown("## About Project")
    html(f"""
    <div class="grid">
      <div class="card"><h4>Project Overview</h4><p>EstateIQ is a small web application that estimates house prices using machine learning. The user enters property details and the trained model returns a predicted value.</p></div>
      <div class="card"><h4>Problem Statement</h4><p>Pricing a house by guessing is unreliable. We need a data-driven way to estimate the value of a property from its location and basic characteristics.</p></div>
      <div class="card"><h4>Objectives</h4><ul><li>Train a Linear Regression model on real data</li><li>Measure accuracy with MAE, RMSE and R&sup2;</li><li>Build a simple, attractive prediction website</li></ul></div>
      <div class="card"><h4>Technologies Used</h4><ul><li>Python, Pandas, NumPy</li><li>Scikit-learn (model) and Joblib (saving)</li><li>Streamlit (website), Plotly (charts), CSS (design)</li></ul></div>
    </div>
    <div class="section-title">Machine Learning Workflow</div>
    <div class="timeline">
      <div class="tl-item"><b>1. Load data</b><span>California Housing</span></div>
      <div class="tl-item"><b>2. Split</b><span>80% train, 20% test</span></div>
      <div class="tl-item"><b>3. Scale</b><span>StandardScaler</span></div>
      <div class="tl-item"><b>4. Train</b><span>Linear Regression</span></div>
      <div class="tl-item"><b>5. Evaluate</b><span>MAE, RMSE, R&sup2;</span></div>
      <div class="tl-item"><b>6. Predict</b><span>Website uses saved model</span></div>
    </div>
    """)
    rows = "".join(
        f"<tr><td>{n}</td><td>{FIELDS[n][0]}</td><td>{FIELDS[n][1]}</td></tr>" for n in b["feature_names"]
    )
    html(f"""
    <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));">
      <div class="card"><h4>Dataset Description</h4>
        <p>California Housing dataset from Scikit-learn: {b["n_records"]:,} records from the 1990 US census. Each row describes one neighbourhood. The target is the median house value in units of $100,000 (converted to dollars on the website).</p>
        <br><table class="info"><tr><th>Column</th><th>Meaning</th><th>Description</th></tr>{rows}</table></div>
      <div class="card"><h4>Linear Regression Explained</h4>
        <p>Linear Regression finds the best straight-line relationship between the inputs and the price. It learns a weight for each feature and adds them up:</p>
        <p style="margin:10px 0"><b>price = w1 x income + w2 x age + ... + bias</b></p>
        <p>Training means choosing the weights that make the prediction errors as small as possible. It is simple, fast and easy to explain.</p></div>
      <div class="card"><h4>Model Evaluation Metrics</h4>
        <table class="info"><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr>
        <tr><td>MAE</td><td>{money(m["mae"] * DOLLARS_PER_UNIT)}</td><td>Average size of the error</td></tr>
        <tr><td>RMSE</td><td>{money(m["rmse"] * DOLLARS_PER_UNIT)}</td><td>Like MAE but punishes large errors more</td></tr>
        <tr><td>R&sup2;</td><td>{m["r2"]:.3f}</td><td>Share of price variation explained</td></tr></table>
        <p style="margin-top:10px">Measured on {b["n_test"]:,} test records the model never saw during training.</p></div>
      <div class="card"><h4>Limitations and Future Improvements</h4>
        <ul><li>Linear Regression cannot capture complex patterns, so accuracy is moderate</li>
        <li>Data is from 1990 California, so prices are not current and not Indian prices</li>
        <li>Prices in the data are capped at about $500,000</li>
        <li>Future: try Random Forest or Gradient Boosting, add more features, use newer data</li></ul></div>
    </div>
    """)


# ------------------------------------------------------------------ main
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.session_state.setdefault("page", "Dashboard")

bundle, load_error = None, None
try:
    bundle = get_bundle()
except Exception as error:  # e.g. dataset could not be downloaded
    load_error = str(error)

sidebar(bundle is not None, bundle)

if bundle is None:
    st.error("The model is not available. " + (load_error or "") +
             " Run `python train_model.py` once (internet needed for the first dataset download), then refresh this page.")
    st.stop()

{"Dashboard": page_dashboard, "Predict Price": page_predict,
 "Model Analytics": page_analytics, "About Project": page_about}[st.session_state.page](bundle)

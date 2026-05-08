import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy: #0D1B2A;
    --blue: #1E6FD9;
    --cyan: #00C2D1;
    --green: #00C896;
    --red: #FF4D6D;
    --card: #131F2E;
    --border: #1E3A5F;
    --text: #E2EAF4;
    --muted: #7A9BBF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy);
    color: var(--text);
}

.main { background-color: var(--navy); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A2A4A 50%, #0D1B2A 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00C2D1, #1E6FD9, #00C896);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--cyan);
}
.metric-label {
    color: var(--muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

.result-healthy {
    background: linear-gradient(135deg, #0D2A1F, #0A3D2A);
    border: 2px solid var(--green);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-diabetic {
    background: linear-gradient(135deg, #2A0D1A, #3D0A1F);
    border: 2px solid var(--red);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

section[data-testid="stSidebar"] {
    background-color: #0A1520;
    border-right: 1px solid var(--border);
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--cyan);
    border-left: 3px solid var(--cyan);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

.stButton > button {
    background: linear-gradient(90deg, #1E6FD9, #00C2D1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    width: 100%;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
}
.stTabs [aria-selected="true"] { color: var(--cyan) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Models & Data ───────────────────────────────────────────────────────
import joblib

@st.cache_resource
def load_models():
    models = joblib.load('models.pkl')
    scaler = joblib.load('scaler.pkl')
    return models, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_clean.csv')
    return df

models, scaler = load_models()
df = load_data()

# ── الأرقام الحقيقية الثابتة ──────────────────────────────────────────────────
results = {
    'Logistic Regression': {
        'acc': 0.7450,
        'report': {
            'weighted avg': {'precision': 0.75, 'recall': 0.74, 'f1-score': 0.74}
        }
    },
    'Decision Tree': {
        'acc': 0.7750,
        'report': {
            'weighted avg': {'precision': 0.78, 'recall': 0.78, 'f1-score': 0.77}
        }
    },
    'KNN': {
        'acc': 0.7950,
        'report': {
            'weighted avg': {'precision': 0.81, 'recall': 0.80, 'f1-score': 0.79}
        }
    },
    'Naive Bayes': {
        'acc': 0.7500,
        'report': {
            'weighted avg': {'precision': 0.75, 'recall': 0.75, 'f1-score': 0.75}
        }
    },
    'SVM': {
        'acc': 0.7950,
        'report': {
            'weighted avg': {'precision': 0.80, 'recall': 0.80, 'f1-score': 0.80}
        }
    },
}

ICONS = {
    'Logistic Regression': '📈',
    'Decision Tree':        '🌳',
    'KNN':                  '🔵',
    'Naive Bayes':          '🎲',
    'SVM':                  '⚡',
}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🩺 Diabetes Risk Predictor</p>
    <p class="hero-sub">Pima Indians Diabetes Dataset · 5 ML Algorithms · SMOTE Balanced</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮  Predict", "📊  Model Comparison", "🗄️  Dataset Info"])

# ══════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════
with tab1:
    col_inputs, col_result = st.columns([1.1, 1], gap="large")

    with col_inputs:
        st.markdown('<p class="section-title">Patient Information</p>', unsafe_allow_html=True)

        model_name = st.selectbox(
            "Choose Algorithm",
            list(models.keys()),
            format_func=lambda x: f"{ICONS[x]}  {x}"
        )

        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.slider("Pregnancies", 0, 17, 3)
            glucose     = st.slider("Glucose (mg/dL)", 44, 199, 120)
            bp          = st.slider("Blood Pressure (mmHg)", 24, 122, 72)
            skin        = st.slider("Skin Thickness (mm)", 7, 99, 29)
        with c2:
            insulin = st.slider("Insulin (μU/mL)", 14, 846, 125)
            bmi     = st.slider("BMI", 18.2, 67.1, 32.0, step=0.1)
            dpf     = st.slider("Diabetes Pedigree Function", 0.078, 2.420, 0.471, step=0.001)
            age     = st.slider("Age", 21, 81, 33)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  RUN PREDICTION")

    with col_result:
        st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            inp = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            inp_s = scaler.transform(inp)
            model = models[model_name]
            pred  = model.predict(inp_s)[0]
            prob  = model.predict_proba(inp_s)[0]

            if pred == 0:
                st.markdown(f"""
                <div class="result-healthy">
                    <div style="font-size:3rem">✅</div>
                    <div class="result-title" style="color:#00C896">HEALTHY</div>
                    <div style="color:#7A9BBF">No diabetes detected</div>
                    <div style="margin-top:1rem;font-family:'Space Mono',monospace;font-size:1.8rem;color:#00C896">
                        {prob[0]:.1%}
                    </div>
                    <div style="color:#7A9BBF;font-size:0.85rem">Confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-diabetic">
                    <div style="font-size:3rem">⚠️</div>
                    <div class="result-title" style="color:#FF4D6D">DIABETIC</div>
                    <div style="color:#7A9BBF">Diabetes risk detected</div>
                    <div style="margin-top:1rem;font-family:'Space Mono',monospace;font-size:1.8rem;color:#FF4D6D">
                        {prob[1]:.1%}
                    </div>
                    <div style="color:#7A9BBF;font-size:0.85rem">Confidence</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">Input Summary</p>', unsafe_allow_html=True)
            summary = pd.DataFrame({
                'Feature': ['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DPF','Age'],
                'Value':   [pregnancies, glucose, bp, skin, insulin, f"{bmi:.1f}", f"{dpf:.3f}", age]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div style="background:#131F2E;border:1px dashed #1E3A5F;border-radius:16px;
                        padding:3rem;text-align:center;margin-top:1rem;">
                <div style="font-size:3rem;margin-bottom:1rem">🔮</div>
                <div style="color:#7A9BBF;font-size:1rem">
                    Fill in patient details and click<br>
                    <strong style="color:#00C2D1">RUN PREDICTION</strong> to see results
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Accuracy Comparison</p>', unsafe_allow_html=True)

    accs = {n: r['acc'] for n, r in results.items()}
    best = max(accs, key=accs.get)

    cols = st.columns(5)
    for i, (name, acc) in enumerate(accs.items()):
        with cols[i]:
            is_best = name == best
            color = "#00C896" if is_best else "#00C2D1"
            border = "#00C896" if is_best else "#1E3A5F"
            crown = "👑 " if is_best else ""
            st.markdown(f"""
            <div style="background:#131F2E;border:2px solid {border};border-radius:14px;
                        padding:1rem;text-align:center;margin-bottom:1rem;">
                <div style="font-size:1.8rem">{ICONS[name]}</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.3rem;
                            color:{color};font-weight:700">{crown}{acc:.1%}</div>
                <div style="color:#7A9BBF;font-size:0.75rem;margin-top:0.3rem">{name}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Performance Bars</p>', unsafe_allow_html=True)
    for name, acc in sorted(accs.items(), key=lambda x: -x[1]):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"<div style='padding-top:0.5rem'>{ICONS[name]} {name}</div>", unsafe_allow_html=True)
        with c2:
            st.progress(acc)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Detailed Metrics Table</p>', unsafe_allow_html=True)
    rows = []
    for name, r in results.items():
        rep = r['report']
        rows.append({
            'Algorithm':  f"{ICONS[name]} {name}",
            'Accuracy':   f"{r['acc']:.2%}",
            'Precision':  f"{rep['weighted avg']['precision']:.2%}",
            'Recall':     f"{rep['weighted avg']['recall']:.2%}",
            'F1-Score':   f"{rep['weighted avg']['f1-score']:.2%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 3 — DATASET INFO
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [("768", "Total Samples"), ("8", "Features"), ("500", "Healthy (0)"), ("268", "Diabetic (1)")]
    for col, (val, lbl) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Sample Data (First 10 Rows)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Statistical Summary</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Preprocessing Steps</p>', unsafe_allow_html=True)
    steps = [
        ("1️⃣ Missing Values", "Zeros in 5 columns replaced with column median"),
        ("2️⃣ Class Imbalance", "SMOTE: 500 Healthy vs 268 Diabetic → 500 vs 500"),
        ("3️⃣ Train/Test Split", "80% training / 20% testing"),
        ("4️⃣ Feature Scaling",  "StandardScaler applied to all features"),
    ]
    for step, desc in steps:
        st.markdown(f"""
        <div style="background:#131F2E;border:1px solid #1E3A5F;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-bottom:0.6rem;display:flex;gap:1rem;align-items:center;">
            <span style="font-weight:700;color:#00C2D1;min-width:180px">{step}</span>
            <span style="color:#7A9BBF;font-size:0.9rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

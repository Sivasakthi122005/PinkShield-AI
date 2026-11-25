import streamlit as st
import sys
import os
import json
import numpy as np
import hmac
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="OncAI — Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL CSS — Dark Medical Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080C14 !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D1219; }
::-webkit-scrollbar-thumb { background: #2563EB; border-radius: 2px; }

/* ── Top Navigation Bar ── */
.nav-bar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(8,12,20,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(37,99,235,0.2);
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60px;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
}
.nav-logo span { color: #2563EB; }
.nav-links {
    display: flex;
    gap: 0.25rem;
}
.nav-link {
    position: relative;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    color: #94A3B8;
    cursor: pointer;
    border: none;
    background: transparent;
    transition: all 0.15s;
    text-decoration: none;
}
.nav-link:hover { color: #fff; background: rgba(255,255,255,0.06); }
.nav-link.active { color: #fff; background: rgba(37,99,235,0.2); border: 1px solid rgba(37,99,235,0.4); }
.nav-link::after {
    content: '';
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: -6px;
    height: 2px;
    border-radius: 99px;
    background: transparent;
    transition: background 0.2s, box-shadow 0.2s;
}
.nav-link.active::after {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    box-shadow: 0 0 10px rgba(37,99,235,0.6);
}
.nav-icon {
    display: inline-block;
    margin-right: 0.45rem;
    font-size: 0.9rem;
    opacity: 0.9;
}
.nav-badge {
    font-size: 0.7rem;
    background: #EF4444;
    color: #fff;
    padding: 2px 6px;
    border-radius: 99px;
    margin-left: 0.5rem;
    font-weight: 600;
}

/* ── Hero Section ── */
.hero {
    padding: 5rem 2.5rem 4rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -100px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.3);
    color: #60A5FA;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 700;
    line-height: 1.1;
    color: #fff;
    margin-bottom: 1.2rem;
    max-width: 700px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94A3B8;
    line-height: 1.7;
    max-width: 580px;
    margin-bottom: 2.5rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #fff;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Section Container ── */
.section {
    padding: 3rem 2.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 0.4rem;
}
.section-sub {
    font-size: 0.9rem;
    color: #64748B;
    margin-bottom: 2rem;
}

/* ── Cards ── */
.card {
    background: #0F1623;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
    height: 100%;
}
.card:hover {
    border-color: rgba(37,99,235,0.4);
    transform: translateY(-2px);
}
.card-icon {
    width: 40px; height: 40px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 1rem;
}
.card-title {
    font-weight: 600;
    font-size: 1rem;
    color: #fff;
    margin-bottom: 0.4rem;
}
.card-text {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.6;
}

/* ── Assessment Panel ── */
.panel {
    background: #0F1623;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.75rem;
    height: 100%;
}
.panel-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.panel-num {
    width: 26px; height: 26px;
    background: #2563EB;
    color: #fff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.panel-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #fff;
}

/* ── Streamlit Widget Overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(37,99,235,0.04) !important;
    border: 1px dashed rgba(37,99,235,0.35) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label { color: #94A3B8 !important; }
[data-testid="stFileUploader"] button {
    background: rgba(37,99,235,0.2) !important;
    border: 1px solid rgba(37,99,235,0.4) !important;
    color: #60A5FA !important;
    border-radius: 6px !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #2563EB !important;
}
[data-testid="stSlider"] > div > div > div {
    background: rgba(37,99,235,0.2) !important;
}
.stSlider label { color: #94A3B8 !important; font-size: 0.82rem !important; }

/* Text Area */
textarea {
    background: #080C14 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
textarea:focus {
    border-color: rgba(37,99,235,0.5) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
textarea::placeholder { color: #475569 !important; }

/* Buttons */
.stButton > button {
    background: #2563EB !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #94A3B8 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #fff !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #0F1623 !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0 !important;
    padding: 0.4rem 0.4rem 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 7px 7px 0 0 !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(37,99,235,0.15) !important;
    color: #60A5FA !important;
    border-bottom: 2px solid #2563EB !important;
}
[data-testid="stTabContent"] {
    background: #0F1623 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1.5rem !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Info / Warning / Success */
[data-testid="stAlert"] {
    background: rgba(37,99,235,0.08) !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #2563EB !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0F1623 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 0 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #0F1623 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #94A3B8 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0D1219 !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Custom Result Box */
.result-box {
    background: linear-gradient(135deg, #0F1A2E 0%, #0F1623 100%);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 14px;
    padding: 2rem;
    margin: 1.5rem 0;
}
.result-score {
    font-family: 'DM Mono', monospace;
    font-size: 4rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-category {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.result-action {
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.6;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #2563EB;
    border-radius: 0 6px 6px 0;
}

/* Module breakdown row */
.mod-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.mod-row:last-child { border-bottom: none; }
.mod-label {
    font-size: 0.8rem;
    color: #64748B;
    width: 140px;
    flex-shrink: 0;
    font-family: 'DM Mono', monospace;
}
.mod-bar-wrap {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
}
.mod-bar {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.mod-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #fff;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}
.mod-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 99px;
    font-weight: 600;
    flex-shrink: 0;
}

/* Disclaimer banner */
.disclaimer {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: #A16207;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1rem 2.5rem;
}

/* FL Timeline */
.fl-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.fl-step:last-child { border-bottom: none; }
.fl-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #2563EB;
    margin-top: 5px;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(37,99,235,0.5);
}
.fl-step-title { font-size: 0.9rem; font-weight: 600; color: #fff; }
.fl-step-desc { font-size: 0.82rem; color: #64748B; margin-top: 0.2rem; line-height: 1.5; }

/* About comparison table override */
[data-testid="stDataFrame"] table th {
    background: rgba(37,99,235,0.15) !important;
    color: #60A5FA !important;
}
[data-testid="stDataFrame"] table td {
    color: #E8EDF5 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FEATURE NAMES & DEFAULTS
# ============================================================
FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean',
    'concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se',
    'smoothness_se','compactness_se','concavity_se',
    'concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst',
    'smoothness_worst','compactness_worst','concavity_worst',
    'concave_points_worst','symmetry_worst','fractal_dimension_worst'
]
DEFAULTS = [
    14.13,18.25,92.0,654.9,0.096,0.104,0.089,0.049,0.181,0.062,
    0.405,1.216,2.866,40.34,0.007,0.025,0.032,0.012,0.020,0.004,
    16.27,25.68,107.3,880.6,0.132,0.253,0.272,0.115,0.290,0.084
]

# ============================================================
# SESSION STATE INIT
# ============================================================
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'prefill_symptom' not in st.session_state:
    st.session_state['prefill_symptom'] = ''
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state:
    st.session_state['auth_error'] = ''


def get_query_param(name):
    try:
        value = st.query_params.get(name)
        if isinstance(value, list):
            return value[0] if value else None
        return value
    except Exception:
        value = st.experimental_get_query_params().get(name, [None])
        return value[0]

def set_page(page_name):
    st.session_state['page'] = page_name
    try:
        if st.session_state.get('authenticated'):
            st.query_params['page'] = page_name
            st.query_params['auth'] = '1'
        else:
            st.query_params['page'] = page_name
    except Exception:
        if st.session_state.get('authenticated'):
            st.experimental_set_query_params(page=page_name, auth='1')
        else:
            st.experimental_set_query_params(page=page_name)
    st.rerun()


def get_auth_credentials():
    username = os.getenv('ONCAI_USERNAME', 'admin')
    password = os.getenv('ONCAI_PASSWORD', 'admin123')

    try:
        username = st.secrets.get('auth_username', username)
        password = st.secrets.get('auth_password', password)
    except Exception:
        pass

    return str(username), str(password)


def is_valid_login(username, password):
    expected_username, expected_password = get_auth_credentials()
    return (
        hmac.compare_digest(username.strip(), expected_username)
        and hmac.compare_digest(password, expected_password)
    )


def logout():
    st.session_state['authenticated'] = False
    st.session_state['results'] = None
    st.session_state['auth_error'] = ''
    st.session_state['page'] = 'Home'
    try:
        st.query_params['page'] = 'Home'
        if 'auth' in st.query_params:
            del st.query_params['auth']
    except Exception:
        st.experimental_set_query_params(page='Home')
    st.rerun()


def show_login_page():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Secure Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Sign in to access OncAI dashboard and risk assessment tools</div>', unsafe_allow_html=True)

    left, center, right = st.columns([1.2, 1.6, 1.2])
    with center:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">🔐</div>
          <div class="panel-title">Account Access</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form('login_form', clear_on_submit=False):
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            submitted = st.form_submit_button('Sign In', use_container_width=True)

        if submitted:
            if is_valid_login(username, password):
                st.session_state['authenticated'] = True
                st.session_state['auth_error'] = ''
                try:
                    st.query_params['page'] = st.session_state.get('page', 'Home')
                    st.query_params['auth'] = '1'
                except Exception:
                    st.experimental_set_query_params(
                        page=st.session_state.get('page', 'Home'),
                        auth='1'
                    )
                st.rerun()
            else:
                st.session_state['auth_error'] = 'Invalid username or password.'

        if st.session_state.get('auth_error'):
            st.error(st.session_state['auth_error'])

        st.caption('Set credentials with ONCAI_USERNAME / ONCAI_PASSWORD or Streamlit secrets: auth_username, auth_password.')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


query_auth = get_query_param('auth')
if query_auth == '1':
    st.session_state['authenticated'] = True

if not st.session_state['authenticated']:
    show_login_page()
    st.stop()

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        from module4_fusion.pipeline import load_all_models
        tabular_model, scaler, nlp_pipeline = load_all_models()
        return tabular_model, scaler, nlp_pipeline, True
    except Exception as e:
        return None, None, None, False

tabular_model, scaler, nlp_pipeline, models_loaded = load_models()


def get_missing_required_model_files(models_dir='models'):
    required_files = [
        'xgboost_model.pkl',
        'scaler.pkl',
        'nlp_pipeline.pkl'
    ]
    missing_files = []
    for file_name in required_files:
        if not os.path.exists(os.path.join(models_dir, file_name)):
            missing_files.append(file_name)
    return missing_files


def is_git_lfs_pointer(file_path):
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            return first_line == 'version https://git-lfs.github.com/spec/v1'
    except Exception:
        return False


def run_image_model_prediction(uploaded_file, model_path='models/efficientnet_best.pt'):
    default_result = {
        'probability': 0.50,
        'label': 'Benign',
        'confidence': 50.0,
        'heatmap_path': None,
        'available': False
    }

    if uploaded_file is None:
        return default_result, None

    if not os.path.exists(model_path) or is_git_lfs_pointer(model_path):
        warning = (
            "Image model file is missing or invalid (`models/efficientnet_best.pt`). "
            "Image prediction used neutral fallback value."
        )
        return default_result, warning

    try:
        import torch
        from module1_image.predict import load_image_model, predict_image

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ext = os.path.splitext(getattr(uploaded_file, 'name', 'scan.png'))[1] or '.png'

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_image_path = temp_file.name

        model = load_image_model(model_path, device)
        prediction = predict_image(temp_image_path, model, device)

        confidence = float(prediction.get('confidence', 0.0))
        if confidence <= 1.0:
            confidence *= 100.0

        result = {
            'probability': float(prediction['probability']),
            'label': prediction['label'],
            'confidence': round(confidence, 1),
            'heatmap_path': prediction.get('heatmap_path'),
            'available': True
        }
        return result, None
    except Exception as error:
        warning = f"Image inference failed: {error}. Neutral fallback was used."
        return default_result, warning

# ============================================================
# TOP NAV BAR
# ============================================================
pages = ["Home", "Assessment", "Performance", "Resources", "Donate", "About"]
page_icons = {"Home":"⬡", "Assessment":"◈", "Performance":"◆", "Resources":"❖", "Donate":"♥", "About":"◇"}

query_page = get_query_param('page')

if query_page in pages:
    st.session_state['page'] = query_page

# Navigation uses real buttons (HTML links open a new browser tab in Streamlit)
nav_cols = st.columns(len(pages))
for _col, _p in zip(nav_cols, pages):
    with _col:
        _is_active = st.session_state['page'] == _p
        if st.button(f"{page_icons.get(_p, '')} {_p}", key=f"nav_{_p}", use_container_width=True,
                     type="primary" if _is_active else "secondary"):
            if not _is_active:
                set_page(_p)

logout_col_left, logout_col_right = st.columns([8, 1])
with logout_col_right:
    if st.button('Logout', use_container_width=True, type='secondary'):
        logout()

page = st.session_state['page']

# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    from module6_dashboard.awareness import render_awareness_home
    render_awareness_home(set_page)
    # Hero (numbers come from the saved metrics files, not hard-coded)
    from module6_dashboard.performance import headline_stats
    _hs = headline_stats()
    st.markdown(f"""
    <div class="hero">
      <div class="hero-tag">🔬 Multi-Modal AI · Explainable · Privacy-Preserving</div>
      <div class="hero-title">
        Next-Generation<br><em>Breast Cancer</em><br>Detection System
      </div>
      <div class="hero-sub">
        Three AI models — imaging, clinical data, and natural language —
        fused into a single explainable risk score. Built with federated
        learning to simulate real-world hospital privacy constraints.
      </div>
      <div class="hero-stats">
        <div>
          <div class="hero-stat-val">{_hs['image']}<span style="font-size:1rem;color:#475569">%</span></div>
          <div class="hero-stat-label">Image CNN Accuracy</div>
        </div>
        <div>
          <div class="hero-stat-val">{_hs['tabular']}<span style="font-size:1rem;color:#475569">%</span></div>
          <div class="hero-stat-label">Clinical ML Accuracy</div>
        </div>
        <div>
          <div class="hero-stat-val">{_hs['federated']}<span style="font-size:1rem;color:#475569">%</span></div>
          <div class="hero-stat-label">Federated Accuracy</div>
        </div>
        <div>
          <div class="hero-stat-val">3<span style="font-size:1rem;color:#475569">×</span></div>
          <div class="hero-stat-label">Input Modalities</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="text-align:center;">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center;">Five independent modules working in concert</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "🖼️", "#1E3A5F", "Module 1", "Image CNN",
         "EfficientNet-B0 trained on 1,995 BreakHis histopathology images with Grad-CAM heatmaps."),
        (c2, "📊", "#1A3A2A", "Module 2", "Clinical ML",
         "XGBoost on 30 Wisconsin features. SHAP waterfall charts for per-prediction explanations."),
        (c3, "💬", "#3A2A1A", "Module 3", "NLP Analysis",
         "TF-IDF + Logistic Regression on symptom text. Converts clinical notes to risk signal."),
        (c4, "⚡", "#2A1A3A", "Module 4", "Fusion Layer",
         "Weighted ensemble of all 3 outputs into a unified 0–100 risk score with recommendations."),
        (c5, "🔒", "#1A2A3A", "Module 5", "Federated Learning",
         "FedAvg simulation across 3 virtual hospitals. Model weights shared, never patient data."),
    ]
    for col, icon, bg, mod, title, desc in cards:
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-icon" style="background:{bg}22; border:1px solid {bg}66;">{icon}</div>
              <div style="font-size:0.7rem; color:#475569; font-family:'DM Mono',monospace;
                          margin-bottom:0.3rem; text-transform:uppercase;">{mod}</div>
              <div class="card-title">{title}</div>
              <div class="card-text">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # How it Works
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="text-align:center;">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center;">Four steps from input to clinical recommendation</div>', unsafe_allow_html=True)

    steps_col = st.columns(4)
    steps = [
        ("01", "Upload Scan", "Provide a histopathology or mammogram image for CNN analysis"),
        ("02", "Enter Measurements", "Input clinical feature values using the 10 key sliders"),
        ("03", "Describe Symptoms", "Type symptoms in plain English for NLP risk extraction"),
        ("04", "Receive Risk Score", "Get a 0–100 unified score with explanation and report"),
    ]
    for col, (num, title, desc) in zip(steps_col, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-family:'DM Mono',monospace; font-size:2rem; color:rgba(37,99,235,0.3);
                          font-weight:500; margin-bottom:0.8rem;">{num}</div>
              <div class="card-title">{title}</div>
              <div class="card-text" style="margin-top:0.4rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # CTA
    st.markdown('<div style="padding: 2rem 2.5rem 4rem;">', unsafe_allow_html=True)
    cta_c1, cta_c2, cta_c3 = st.columns([1, 2, 1])
    with cta_c2:
        if st.button("Start Risk Assessment →", use_container_width=True):
            set_page('Assessment')
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: ASSESSMENT
# ============================================================
elif page == "Assessment":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="text-align:center; font-size:1.9rem;">Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center; font-size:1rem;">Complete all three input panels, then run the AI analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ---- Panel 1: Image ----
    with col1:
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">1</div>
          <div class="panel-title">Histopathology Image</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload scan", type=['png','jpg','jpeg'],
            label_visibility="collapsed"
        )
        if uploaded_file:
            from PIL import Image as PILImage
            img = PILImage.open(uploaded_file)
            st.image(img, use_container_width=True)
            st.markdown("""
            <div style="display:flex;align-items:center;gap:0.4rem;margin-top:0.5rem;">
              <div style="width:6px;height:6px;border-radius:50%;background:#22C55E;"></div>
              <span style="font-size:0.8rem;color:#22C55E;">Image ready for analysis</span>
            </div>
            """, unsafe_allow_html=True)
        else:
                        st.info("No image uploaded. Demo mode will use a neutral probability.")

    # ---- Panel 2: Sliders ----
    with col2:
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">2</div>
          <div class="panel-title">Clinical Measurements</div>
        </div>
        """, unsafe_allow_html=True)

        important_features = [
            ('radius_mean',             6.0,   30.0),
            ('texture_mean',            9.0,   40.0),
            ('perimeter_mean',         40.0,  190.0),
            ('area_mean',             140.0, 2500.0),
            ('smoothness_mean',        0.05,   0.17),
            ('compactness_mean',       0.02,   0.35),
            ('concavity_mean',         0.00,   0.43),
            ('concave_points_mean',    0.00,   0.20),
            ('symmetry_mean',          0.10,   0.30),
            ('fractal_dimension_mean', 0.05,   0.10),
        ]
        slider_values = []
        for name, min_v, max_v in important_features:
            idx = FEATURE_NAMES.index(name)
            val = st.slider(
                name.replace('_',' ').title(),
                min_value=float(min_v), max_value=float(max_v),
                value=float(DEFAULTS[idx]),
                step=float((max_v - min_v) / 200),
                label_visibility="visible"
            )
            slider_values.append(val)

        all_features = list(DEFAULTS)
        for i, (name, _, _) in enumerate(important_features):
            all_features[FEATURE_NAMES.index(name)] = slider_values[i]

    # ---- Panel 3: Symptoms ----
    with col3:
        st.markdown("""
        <div class="panel-header">
          <div class="panel-num">3</div>
          <div class="panel-title">Symptom Description</div>
        </div>
        """, unsafe_allow_html=True)

        default_text = st.session_state.get('prefill_symptom', '')
        symptom_text = st.text_area(
            "Symptom text",
            value=default_text,
            height=180,
            placeholder="Describe symptoms — e.g., painless lump in left breast for 3 months with skin dimpling and nipple retraction...",
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top:0.8rem;'>", unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            if st.button("⚡ High Risk", use_container_width=True, type="secondary"):
                st.session_state['prefill_symptom'] = \
                    "painless hard lump in left breast for 3 months with skin dimpling and nipple retraction"
                st.rerun()
        with q2:
            if st.button("✓ Low Risk", use_container_width=True, type="secondary"):
                st.session_state['prefill_symptom'] = \
                    "bilateral breast tenderness before menstruation, soft movable lump that changes with cycle"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Run Button ----
    st.markdown("<div style='padding:0 2.5rem 1rem;'>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([2, 1, 2])
    with b2:
        run_clicked = st.button(
            "Run AI Analysis →",
            use_container_width=True,
            disabled=not models_loaded
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if not models_loaded:
        missing_files = get_missing_required_model_files()
        if missing_files:
            st.warning(
                "Required model files are missing: " + ", ".join(missing_files) +
                ". Upload these files to the 'models/' folder before deployment."
            )
        else:
            st.warning("Models could not be loaded. Verify dependencies and model compatibility on the deployment server.")

    # ============================================================
    # RESULTS (inline after button)
    # ============================================================
    if run_clicked or st.session_state.get('results'):
        if run_clicked:
            with st.spinner(""):
                try:
                    from module2_tabular.predict_tabular import predict_tabular
                    from module3_nlp.predict_nlp import predict_text
                    from module4_fusion.fusion import weighted_fusion
                    from module4_fusion.risk_score import compute_risk_score
                    from module4_fusion.report_generator import generate_report

                    tabular_result = predict_tabular(all_features, tabular_model, scaler)
                    text_input = symptom_text.strip() if symptom_text.strip() else "no symptoms provided"
                    nlp_result = predict_text(text_input, nlp_pipeline)
                    nlp_result['available'] = bool(symptom_text.strip())
                    image_result, image_warning = run_image_model_prediction(uploaded_file)
                    from module4_fusion.fusion import modality_agreement
                    img_p = image_result['probability'] if image_result.get('available') else None
                    nlp_p = nlp_result['probability'] if nlp_result['available'] else None
                    # missing modalities are dropped and weights re-normalised (no fake 0.5)
                    fused_prob = weighted_fusion(img_p, tabular_result['probability'], nlp_p)
                    agreement = modality_agreement(img_p, tabular_result['probability'], nlp_p)
                    risk = compute_risk_score(fused_prob)
                    report = generate_report(image_result, tabular_result, nlp_result, risk)
                    st.session_state['results'] = {
                        'image': image_result,
                        'tabular': tabular_result,
                        'nlp': nlp_result,
                        'risk': risk,
                        'agreement': agreement,
                        'report': report,
                        'uploaded': uploaded_file is not None,
                        'image_warning': image_warning
                    }
                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    st.stop()

        res = st.session_state.get('results')
        if not res:
            st.stop()

        risk = res['risk']
        image_result = res['image']
        tabular_result = res['tabular']
        nlp_result = res['nlp']
        report = res['report']
        image_warning = res.get('image_warning')

        # ---- Score Banner ----
        color_map = {
            'green': '#22C55E', 'lightgreen': '#86EFAC',
            'orange': '#F97316', 'darkorange': '#EA580C', 'red': '#EF4444'
        }
        score_color = color_map.get(risk['color'], '#EF4444')

        st.markdown(f"""
        <div style="padding:0 2.5rem; margin-top:1rem;">
        <div class="result-box">
          <div style="display:flex; gap:3rem; align-items:flex-start; flex-wrap:wrap;">
            <div>
              <div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                          letter-spacing:0.08em; margin-bottom:0.5rem;">Unified Risk Score</div>
              <div class="result-score" style="color:{score_color};">{risk['score']}</div>
              <div style="font-size:0.75rem; color:#475569; margin-top:0.2rem;">out of 100</div>
            </div>
            <div style="flex:1; min-width:200px;">
              <div class="result-category" style="color:{score_color};">{risk['category']}</div>
              <div class="result-action">{risk['action']}</div>
              <div style="margin-top:1.2rem;">
                <div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                            letter-spacing:0.06em; margin-bottom:0.8rem;">Module Breakdown</div>
        """, unsafe_allow_html=True)

        # Module bars
        modules = [
            ("IMAGE CNN", image_result['probability'], image_result['label'], "#2563EB"),
            ("CLINICAL ML", tabular_result['probability'], tabular_result['label'], "#06B6D4"),
            ("SYMPTOM NLP", nlp_result['probability'], nlp_result['label'], "#8B5CF6"),
        ]
        bars_html = ""
        for label, prob, lbl, color in modules:
            pct = round(prob * 100, 1)
            badge_bg = "#EF444420" if lbl in ['Malignant', 'High Risk'] else "#22C55E20"
            badge_color = "#EF4444" if lbl in ['Malignant', 'High Risk'] else "#22C55E"
            bars_html += (
                f'<div class="mod-row">\n'
                f'  <div class="mod-label">{label}</div>\n'
                f'  <div class="mod-bar-wrap">\n'
                f'    <div class="mod-bar" style="width:{pct}%; background:{color};"></div>\n'
                f'  </div>\n'
                f'  <div class="mod-val">{pct}%</div>\n'
                f'  <div class="mod-badge" style="background:{badge_bg}; color:{badge_color};">{lbl}</div>\n'
                f'</div>\n'
            )

        st.markdown(
            bars_html + "</div>\n</div>\n</div>\n</div>\n</div>\n",
            unsafe_allow_html=True
        )

        # ---- Detail Tabs ----
        st.markdown("<div style='padding:0 2.5rem 3rem; margin-top:1.5rem;'>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs([
            "🖼️  Imaging Analysis",
            "📊  Clinical SHAP",
            "💬  Symptom NLP",
            "📋  Full Report"
        ])

        with tab1:
            i1, i2 = st.columns(2)
            with i1:
                if res['uploaded']:
                    st.image(uploaded_file, caption="Uploaded Scan", use_container_width=True)
                    heatmap_path = image_result.get('heatmap_path')
                    if heatmap_path and os.path.exists(heatmap_path):
                        st.image(heatmap_path, caption="Grad-CAM Heatmap", use_container_width=True)
                else:
                    st.info("Upload an image in the assessment panel to view Grad-CAM output.")
            with i2:
                st.metric("Malignant Probability", f"{image_result['probability']*100:.1f}%")
                st.metric("Prediction", image_result['label'])
                st.metric("Model Confidence", f"{image_result['confidence']}%")
                if image_warning:
                    st.warning(image_warning)
                st.markdown("""
                <div style="margin-top:1rem; padding:0.8rem; background:rgba(37,99,235,0.05);
                            border-radius:8px; font-size:0.82rem; color:#64748B; line-height:1.6;">
                  <strong style="color:#60A5FA;">EfficientNet-B0</strong> trained on 1,995 BreakHis
                  histopathology images at 40× magnification.
                  Grad-CAM heatmap highlights suspicious regions.
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Malignant Probability", f"{tabular_result['probability']*100:.1f}%")
                st.metric("Prediction", tabular_result['label'])
                st.metric("Confidence", f"{tabular_result['confidence']}%")
            with s2:
                shap_path = 'models/shap_waterfall.png'
                from module2_tabular.explain_live import explain_single
                live_fig = explain_single(tabular_model, scaler, all_features)
                if live_fig is not None:
                    st.pyplot(live_fig, use_container_width=True)
                    st.caption("SHAP waterfall computed live for the values you entered.")
                elif os.path.exists(shap_path):
                    st.image(shap_path, caption="Example SHAP chart (static - live SHAP unavailable)", use_container_width=True)
                else:
                    st.markdown("""
                    <div style="padding:2rem; text-align:center; border:1px dashed rgba(255,255,255,0.1); border-radius:8px;">
                      <div style="font-size:2rem;">📊</div>
                      <div style="color:#475569; margin-top:0.5rem; font-size:0.85rem;">
                        SHAP chart available after running Module 2 training
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            t1, t2 = st.columns(2)
            with t1:
                st.metric("Risk Signal", f"{nlp_result['risk_signal']}%")
                st.metric("Classification", nlp_result['label'])
                st.metric("Confidence", nlp_result['confidence'])
            with t2:
                st.markdown(f"""
                <div style="background:rgba(139,92,246,0.07); border:1px solid rgba(139,92,246,0.2);
                            border-radius:10px; padding:1.2rem;">
                  <div style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                              letter-spacing:0.06em; margin-bottom:0.6rem;">Input Text</div>
                  <div style="font-size:0.9rem; color:#E8EDF5; line-height:1.6; font-style:italic;">
                    "{text_input if 'text_input' in dir() else 'N/A'}"
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            st.markdown("""
            <div style="background:#080C14; border-radius:8px; padding:1rem;">
            """, unsafe_allow_html=True)
            st.code(report['detailed'], language=None)
            st.markdown("</div>", unsafe_allow_html=True)
            st.download_button(
                "⬇  Download Report",
                data=report['detailed'],
                file_name="oncai_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "Resources":
    from module6_dashboard.awareness import render_resources
    render_resources()

elif page == "Donate":
    from module6_dashboard.awareness import render_donate
    render_donate()

elif page == "Performance":
    from module6_dashboard.performance import render_performance
    render_performance()

elif page == "About":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="text-align:center;">About</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center;">Project Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:900px; margin:1rem auto 0 auto; background:#0F1623;
                border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1.5rem;">
      <div style="font-size:0.98rem; color:#C9D3E0; line-height:1.9; text-align:center;">
        This project is a multi-modal breast cancer risk assessment system that combines
        image-based prediction, clinical feature analysis, and symptom-text understanding
        into a single risk score. The dashboard is designed for demonstration and project
        presentation, with explainable outputs and a streamlined user workflow from input
        collection to final report generation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

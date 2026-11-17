
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Emergency Dispatch Intelligence",
    page_icon="🚨",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sentiment_model.pkl"


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    st.warning("No trained model was found. Training a new model now...")

    try:
        from train import train_and_save_model

        train_and_save_model()
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - runtime guard for missing setup
        st.error(f"Unable to train the model automatically: {exc}")
        st.stop()


def detect_incident_topic(text):
    text = text.lower()

    topic_rules = [
        ("Fire Incident", ["fire", "burning", "smoke", "blaze"]),
        ("Medical Emergency", ["unconscious", "not breathing", "difficulty breathing", "bleeding", "injured", "collapsed", "heart attack", "hurt"]),
        ("Threat / Security Issue", ["shooting", "threatening", "attacking", "stolen", "break-in", "following me", "fighting"]),
        ("Rescue / Trapped Person", ["trapped", "missing", "in danger", "danger", "stuck"]),
        ("Traffic / Infrastructure Problem", ["traffic signal", "street light", "fallen tree", "road", "noise complaint", "lost wallet"]),
        ("General Service Request", ["information", "service", "report", "help", "complaint"]),
    ]

    for topic, keywords in topic_rules:
        if any(keyword in text for keyword in keywords):
            return topic

    return "Unclear / Other Situation"


def detect_help_needed(text):
    text = text.lower()

    help_rules = [
        ("Firefighters / Rescue Team", ["fire", "burning", "smoke", "building is on fire", "trapped"]),
        ("Ambulance / Medical Team", ["unconscious", "breathing", "bleeding", "injured", "collapsed", "heart attack", "hurt"]),
        ("Police / Security Support", ["shooting", "threatening", "attacking", "stolen", "break-in", "following me", "fighting"]),
        ("Search / Welfare Team", ["missing", "in danger", "danger", "worried", "child is missing"]),
        ("Road / Public Works Team", ["traffic signal", "street light", "fallen tree", "road issue"]),
        ("General Information / Non-urgent Assistance", ["information", "service", "complaint", "lost wallet"]),
    ]

    for help_text, keywords in help_rules:
        if any(keyword in text for keyword in keywords):
            return help_text

    return "Dispatcher review required"


def determine_severity(text, prediction, confidence):
    text = text.lower()
    critical_keywords = [
        "fire",
        "unconscious",
        "not breathing",
        "difficulty breathing",
        "bleeding",
        "trapped",
        "shooting",
        "heart attack",
        "severe injury"
    ]

    if any(keyword in text for keyword in critical_keywords):
        return "Critical"
    if prediction in ["Panic", "Distress"] and confidence >= 60:
        return "Critical"
    if prediction in ["Fear", "Anger"] or confidence >= 45:
        return "Medium"
    return "Low"


# Load trained model
model = load_model()

st.markdown(
    """
    <style>
    :root {
        --navy: #101d32;
        --navy-light: #1a2c46;
        --blue: #2878c8;
        --gold: #e3b341;
        --ink: #17263b;
        --muted: #66758a;
        --panel: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(40, 120, 200, 0.13), transparent 30%),
            linear-gradient(180deg, #f7f9fc 0%, #edf2f8 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(241, 244, 248, 0.9);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
    }

    .dispatch-banner {
        background:
            radial-gradient(circle at 88% 20%, rgba(59, 130, 196, 0.35), transparent 22%),
            linear-gradient(115deg, #0d192b, #1d3554 68%, #24466a);
        border: 1px solid rgba(227, 179, 65, 0.55);
        border-left: 6px solid var(--gold);
        border-radius: 18px;
        color: #fff;
        min-height: 190px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 42px rgba(16, 29, 50, 0.18);
        position: relative;
        overflow: hidden;
    }

    .dispatch-kicker {
        color: #f4ce65;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .dispatch-banner h1 {
        color: #fff;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.15;
        margin: 0;
    }

    .dispatch-banner p {
        color: #d6e2f0;
        font-size: 1.02rem;
        max-width: 670px;
        margin: 0.6rem 0 0;
    }

    .banner-status {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        color: #e8f1fb;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1.15rem;
        padding: 0.38rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .status-dot {
        background: #61d394;
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(97, 211, 148, 0.14);
        height: 8px;
        width: 8px;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 0 0 1.3rem;
    }

    .status-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #dfe6ef;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 4px 14px rgba(16, 29, 50, 0.04);
    }

    .status-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .status-value {
        color: var(--navy);
        font-size: 0.95rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    .section-heading {
        color: var(--navy);
        font-size: 1.2rem;
        font-weight: 750;
        margin: 0.25rem 0 0.25rem;
    }

    .section-caption {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stTextArea"] label,
    [data-testid="stTextArea"] label p {
        color: var(--navy);
        font-weight: 700;
    }

    [data-testid="stTextArea"] textarea {
        background: #fff;
        border: 1px solid #c6d2e0;
        border-radius: 12px;
        color: var(--ink);
        font-size: 0.98rem;
        line-height: 1.55;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--blue);
        box-shadow: 0 0 0 2px rgba(40, 120, 200, 0.16);
    }

    .stButton > button {
        background: linear-gradient(100deg, #246eb5, #3286d0);
        border: 0;
        border-radius: 10px;
        color: #fff;
        font-weight: 750;
        min-height: 3rem;
        box-shadow: 0 7px 16px rgba(40, 120, 200, 0.2);
        transition: filter 120ms ease, transform 120ms ease;
    }

    .stButton > button:hover {
        filter: brightness(0.94);
        border: 0;
        color: #fff;
        transform: translateY(-2px);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #dce4ee;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(16, 29, 50, 0.055);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid #dce3ec;
        border-top: 3px solid var(--blue);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(16, 29, 50, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-size: 1.2rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #dce3ec;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #2878c8, #54a5e8);
    }

    .severity-badge {
        display: inline-block;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        padding: 0.35rem 0.8rem;
        text-transform: uppercase;
    }

    .severity-critical {
        background: #fee2e2;
        color: #991b1b;
    }

    .severity-medium {
        background: #fef3c7;
        color: #854d0e;
    }

    .severity-low {
        background: #dcfce7;
        color: #166534;
    }

    .dispatch-note {
        background: #e9eff7;
        border-left: 4px solid var(--gold);
        border-radius: 8px;
        color: #36465c;
        font-size: 0.9rem;
        padding: 0.8rem 1rem;
    }

    .protocol-title {
        color: var(--navy);
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .protocol-item {
        border-bottom: 1px solid #e8edf3;
        color: #53647a;
        font-size: 0.88rem;
        padding: 0.65rem 0;
    }

    .protocol-item:last-child {
        border-bottom: 0;
    }

    .protocol-item strong {
        color: var(--navy);
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .dispatch-banner {
            min-height: 0;
            padding: 1.4rem;
        }

        .status-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dispatch-banner">
        <div class="dispatch-kicker">Emergency Response • Dispatch Console</div>
        <h1>🚨 Incident Analysis</h1>
        <p>Review caller information and prepare a clear, human-reviewed response.</p>
        <div class="banner-status"><span class="status-dot"></span> ANALYSIS SYSTEM READY</div>
    </div>
    <div class="status-grid">
        <div class="status-card">
            <div class="status-label">Workspace</div>
            <div class="status-value">Emergency Communications</div>
        </div>
        <div class="status-card">
            <div class="status-label">Analysis mode</div>
            <div class="status-value">Transcript triage</div>
        </div>
        <div class="status-card">
            <div class="status-label">Review status</div>
            <div class="status-value">Human confirmation required</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

intake_col, protocol_col = st.columns([1.7, 1], gap="large")
with intake_col:
    with st.container(border=True):
        st.markdown('<div class="section-heading">📞 Incoming Call</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Enter the caller transcript for a preliminary incident assessment.</div>',
            unsafe_allow_html=True,
        )
        user_input = st.text_area(
            "Call transcript",
            placeholder="Example: My family is in danger, please send help...",
            height=175,
            label_visibility="collapsed",
        )
        analyze_call = st.button("Analyze Incident", type="primary", use_container_width=True)

with protocol_col:
    with st.container(border=True):
        st.markdown('<div class="protocol-title">🛡️ Dispatch quick guide</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="protocol-item"><strong>1. Describe</strong><br>Record what the caller says in their own words.</div>
            <div class="protocol-item"><strong>2. Assess</strong><br>Review suggested topic, sentiment, and severity.</div>
            <div class="protocol-item"><strong>3. Confirm</strong><br>Verify details and response with a trained dispatcher.</div>
            """,
            unsafe_allow_html=True,
        )

# Critical risk keyword screening
critical_keywords = [
    "fire",
    "unconscious",
    "not breathing",
    "difficulty breathing",
    "bleeding",
    "trapped",
    "shooting",
    "heart attack",
    "severe injury"
]

if analyze_call:

    if user_input.strip() == "":
        st.warning("Please enter an emergency message.")

    else:
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        confidence = max(probabilities) * 100

        incident_topic = detect_incident_topic(user_input)
        requested_help = detect_help_needed(user_input)
        severity = determine_severity(user_input, prediction, confidence)

        st.markdown('<div class="section-heading">Situation Overview</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Incident Topic", incident_topic)
        col2.metric("Caller Sentiment", prediction)
        col3.metric("Model Confidence", f"{confidence:.2f}%")

        severity_class = f"severity-{severity.lower()}"
        st.markdown(
            f'<p><strong>Severity:</strong> <span class="severity-badge {severity_class}">{severity}</span></p>',
            unsafe_allow_html=True,
        )
        st.write("**Recommended assistance:**", requested_help)
        st.progress(confidence / 100, text=f"Model confidence: {confidence:.2f}%")

        # Check emergency keywords
        text = user_input.lower()
        detected = [word for word in critical_keywords if word in text]

        if detected:
            st.error(
                "Potentially critical emergency indicators detected. "
                "Immediate human dispatcher review required."
            )
            st.write("**Detected keywords:**", ", ".join(detected))
        else:
            st.warning(
                "Automated urgency assessment is not available. "
                "Human dispatcher review is required."
            )

        st.markdown('<div class="section-heading">Incident Report Summary</div>', unsafe_allow_html=True)
        summary = {
            "Incident Topic": incident_topic,
            "Predicted Sentiment": prediction,
            "Confidence (%)": round(confidence, 2),
            "Severity": severity,
            "Recommended Help": requested_help,
        }
        st.dataframe(pd.DataFrame([summary]))

        st.markdown('<div class="section-heading">Dispatch Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Suggested routing only; confirm all actions with a dispatcher.</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric("Severity Level", severity)
        col2.metric("Suggested Response", requested_help)

        st.markdown(
            '<div class="dispatch-note">Decision-support prototype only. This analysis may be inaccurate and must not replace emergency services or a trained dispatcher. For an active emergency, contact your local emergency number.</div>',
            unsafe_allow_html=True,
        )

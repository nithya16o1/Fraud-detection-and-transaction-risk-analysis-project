import os
import sys
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from sklearn.metrics import confusion_matrix


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)


from sentinel.src.features.timeline import get_card_timeline

from sentinel.src.evidence.synthesizer import (
    synthesize_evidence,
    compute_empirical_cutoffs
)

from sentinel.src.audit.logger import (
    log_audit_entry,
    get_audit_logs
)


# =======================================================
# PAGE CONFIGURATION
# =======================================================

st.set_page_config(
    page_title="Sentinel — AI Risk Investigation Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =======================================================
# RECRUITER-IMPRESSIVE DESIGN SYSTEM (CSS)
# =======================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #0f172a 45%, #020617 100%);
        color: #f8fafc;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .tech-tag {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 10px;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    .info-card-modern {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .info-card-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .info-card-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .risk-high-modern {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(159, 18, 57, 0.25) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-left: 5px solid #f43f5e;
        border-radius: 14px;
        padding: 16px;
        margin: 12px 0 18px 0;
        box-shadow: 0 10px 25px -5px rgba(225, 29, 72, 0.2);
    }

    .risk-high-title {
        color: #fda4af;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .risk-high-desc {
        color: #fecdd3;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .risk-low-modern {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 5px solid #10b981;
        border-radius: 14px;
        padding: 16px;
        margin: 12px 0 18px 0;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.2);
    }

    .risk-low-title {
        color: #6ee7b7;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .risk-low-desc {
        color: #a7f3d0;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .evidence-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .evidence-tag {
        font-size: 0.72rem;
        font-weight: 700;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }

    .evidence-content {
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .recommendation-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(67, 56, 202, 0.25) 100%);
        border: 1px solid rgba(129, 140, 248, 0.4);
        border-radius: 14px;
        padding: 16px;
        margin: 14px 0 18px 0;
    }

    .rec-title {
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .rec-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 4px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    }

    .footer-modern {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 30px 0 15px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 30px;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 700 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

</style>
""", unsafe_allow_html=True)


# =======================================================
# HEADER BANNER
# =======================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>🛡️ SENTINEL AI v2.4</span> • <span>BEHAVIORAL DISPUTE INVESTIGATION ENGINE</span>
    </div>
    <h1 class="hero-title">Sentinel Risk Manager</h1>
    <div class="hero-subtitle">
        Rewind. Investigate. Defend.
        <span style="opacity: 0.5;">|</span>
        <span class="tech-tag">XGBoost ML</span>
        <span class="tech-tag">LLM Evidence Synthesis</span>
        <span class="tech-tag">Append-Only Audit</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =======================================================
# LOAD ARTIFACTS & DATASETS
# =======================================================

@st.cache_resource
def load_artifacts():
    model_path = os.path.join("sentinel", "src", "model", "xgb_fraud_model.joblib")
    meta_path = os.path.join("sentinel", "src", "model", "model_meta.json")

    model = joblib.load(model_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta


@st.cache_data
def load_datasets():
    clean_path = os.path.join(
        "sentinel",
        "data",
        "processed",
        "cleaned_transactions.parquet"
    )

    test_path = os.path.join(
        "sentinel",
        "data",
        "processed",
        "test_set.parquet"
    )

    df_full = pd.read_parquet(clean_path)
    df_test = pd.read_parquet(test_path)

    cutoffs = compute_empirical_cutoffs(df_full)

    return df_full, df_test, cutoffs


model, meta = load_artifacts()
df_full, df_test, cutoffs = load_datasets()

threshold = meta["chosen_threshold"]


# =======================================================
# SIDEBAR — CASE SELECTOR
# =======================================================

st.sidebar.markdown("### 🔍 Investigation Console")

st.sidebar.caption(
    "Select a flagged transaction for forensic analysis:"
)


fraud_demo_ids = [
    3569920,
    3447082,
    3207828,
    3259269
]


sample_txs = df_test[
    df_test["TransactionID"].isin(fraud_demo_ids)
]


selected_tx_id = st.sidebar.selectbox(
    "Disputed Transaction ID",
    sample_txs["TransactionID"].tolist()
)


st.sidebar.markdown("---")


st.sidebar.markdown("""
<div style="background: rgba(30, 41, 59, 0.5); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);">
    <div style="font-size: 0.72rem; font-weight: 700; color: #818cf8; text-transform: uppercase; margin-bottom: 4px;">
        ⚙️ ARCHITECTURE NOTICE
    </div>
    <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.3;">
        Combines GBDT fraud probability, empirical behavioral velocity cutoffs, dynamic LLM evidence synthesis, and human-in-the-loop audit logging.
    </div>
</div>
""", unsafe_allow_html=True)


# =======================================================
# TARGET TRANSACTION CALCULATION
# =======================================================

target_row = df_test[
    df_test["TransactionID"] == selected_tx_id
].iloc[0]


feature_cols = meta["feature_cols"]


X_single = pd.DataFrame(
    [target_row[feature_cols]]
)


ml_score = float(
    model.predict_proba(
        X_single
    )[:, 1][0]
)


# =======================================================
# TOP SUMMARY METRICS
# =======================================================

st.markdown("""
<div class="section-header">
    <span>📊</span> Key Investigation Metrics
</div>
""", unsafe_allow_html=True)


top1, top2, top3, top4 = st.columns(4)


with top1:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Transaction ID</div>
        <div class="kpi-value">{selected_tx_id}</div>
        <div class="kpi-sub">Target Case</div>
    </div>
    """, unsafe_allow_html=True)


with top2:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Disputed Amount</div>
        <div class="kpi-value">${target_row['TransactionAmt']:.2f}</div>
        <div class="kpi-sub">Gross Exposure</div>
    </div>
    """, unsafe_allow_html=True)


with top3:

    score_color = (
        "#f43f5e"
        if ml_score >= threshold
        else "#10b981"
    )

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">ML Fraud Score</div>
        <div class="kpi-value" style="color: {score_color};">{ml_score:.4f}</div>
        <div class="kpi-sub">Probability Output</div>
    </div>
    """, unsafe_allow_html=True)


with top4:

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Decision Cutoff</div>
        <div class="kpi-value">{threshold:.2f}</div>
        <div class="kpi-sub">Threshold Boundary</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# =======================================================
# MAIN INVESTIGATION CONSOLE
# =======================================================

col1, col2 = st.columns(
    [1.1, 1.6],
    gap="large"
)


# =======================================================
# LEFT COLUMN — DETAILS & EVIDENCE & DECISION
# =======================================================

with col1:

    st.markdown("""
    <div class="section-header">
        <span>📋</span> Case Attributes
    </div>
    """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="info-card-modern">
        <span class="info-card-label">Transaction ID</span>
        <span class="info-card-val">{selected_tx_id}</span>
    </div>

    <div class="info-card-modern">
        <span class="info-card-label">Amount</span>
        <span class="info-card-val">${target_row['TransactionAmt']:.2f}</span>
    </div>

    <div class="info-card-modern">
        <span class="info-card-label">Entity Proxy</span>
        <span class="info-card-val">{target_row['entity_id']}</span>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class="section-header">
        <span>🎯</span> Risk Status
    </div>
    """, unsafe_allow_html=True)


    if ml_score >= threshold:

        st.markdown(f"""
        <div class="risk-high-modern">
            <div class="risk-high-title">
                🚨 HIGH FRAUD RISK FLAGGED
            </div>

            <div class="risk-high-desc">
                Probability <strong>{ml_score:.4f}</strong>
                exceeds threshold (<strong>{threshold:.2f}</strong>).
                Hold transaction for review.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="risk-low-modern">
            <div class="risk-low-title">
                ✅ LOW FRAUD RISK CONFIRMED
            </div>

            <div class="risk-low-desc">
                Probability <strong>{ml_score:.4f}</strong>
                is within standard limits below cutoff
                (<strong>{threshold:.2f}</strong>).
            </div>
        </div>
        """, unsafe_allow_html=True)


    # Evidence synthesis call

    evidence_res = synthesize_evidence(

        transaction_id=selected_tx_id,

        amount=target_row["TransactionAmt"],

        ml_score=ml_score,

        threshold=threshold,

        amt_vs_avg=target_row["amt_vs_rolling_avg"],

        tx_24h=target_row["tx_count_last_24h"],

        time_since_last_tx=target_row["time_since_last_tx"],

        spike_cutoff=cutoffs["spike_cutoff"],

        tx24_cutoff=cutoffs["tx24_cutoff"]

    )


    st.markdown("""
    <div class="section-header">
        <span>🤖</span> Evidence Synthesis
    </div>
    """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="evidence-box">
        <div class="evidence-tag">Primary Concern</div>
        <div class="evidence-content">
            {evidence_res['primary_concern']}
        </div>
    </div>

    <div class="evidence-box">
        <div class="evidence-tag">Secondary Concern</div>
        <div class="evidence-content">
            {evidence_res['secondary_concern']}
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="evidence-tag" style="margin-left: 2px;">Fact Audit Summary</div>',
        unsafe_allow_html=True
    )


    for ev in evidence_res["evidence"]:

        st.markdown(f"""
        <div style="
            background: rgba(15, 23, 42, 0.4);
            border-left: 3px solid #6366f1;
            padding: 6px 10px;
            margin-bottom: 5px;
            border-radius: 6px;
            font-size: 0.82rem;
            color: #cbd5e1;
        ">
            • {ev}
        </div>
        """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="recommendation-box">
        <div class="rec-title">
            💡 Recommended Action
        </div>

        <div class="rec-value">
            {evidence_res['recommendation']}
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Merchant decision

    st.markdown("""
    <div class="section-header">
        <span>✍️</span> Merchant Decision
    </div>
    """, unsafe_allow_html=True)


    decision = st.radio(
        "Choose final action:",
        [
            "Contest Dispute",
            "Send to Review",
            "Accept Loss"
        ]
    )


    if st.button(
        "Submit Decision & Log to Audit",
        type="primary"
    ):

        log_audit_entry(

            transaction_id=selected_tx_id,

            amount=target_row["TransactionAmt"],

            ml_score=ml_score,

            threshold=threshold,

            llm_recommendation=evidence_res[
                "recommendation"
            ],

            rule_recommendation=evidence_res[
                "rule_recommendation"
            ],

            merchant_decision=decision,

            disagreement_flag=evidence_res[
                "disagreement_flag"
            ],

            evidence_summary=evidence_res[
                "evidence"
            ]

        )


        st.success(
            f"Decision '{decision}' recorded to audit log."
        )


# =======================================================
# RIGHT COLUMN — TIME MACHINE & INDICATORS
# =======================================================

with col2:

    st.markdown("""
    <div class="section-header">
        <span>⏳</span> Fraud Time Machine
    </div>
    """, unsafe_allow_html=True)


    st.caption(
        f"Historical behavior timeline for entity "
        f"`{target_row['entity_id']}`"
    )


    history_df, stats = get_card_timeline(
        df_full,
        selected_tx_id
    )


    fig = px.scatter(

        history_df,

        x="TransactionDT",

        y="TransactionAmt",

        size="TransactionAmt",

        color="isFraud",

        color_discrete_map={
            0: "#10b981",
            1: "#ef4444"
        },

        hover_data=[
            "TransactionID",
            "TransactionAmt"
        ],

        title=(
            f"Entity `{target_row['entity_id']}` "
            f"Behavior History"
        )

    )


    fig.add_trace(

        go.Scatter(

            x=[
                target_row["TransactionDT"]
            ],

            y=[
                target_row["TransactionAmt"]
            ],

            mode="markers",

            marker=dict(
                size=20,
                color="#f59e0b",
                symbol="star",
                line=dict(
                    width=2,
                    color="#ffffff"
                )
            ),

            name="Disputed Transaction"

        )

    )


    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(15, 23, 42, 0)",

        plot_bgcolor="rgba(15, 23, 42, 0.5)",

        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            color="#94a3b8"
        ),

        title_font=dict(
            size=13,
            color="#f8fafc"
        ),

        height=410,

        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="Class"
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            title="Transaction DT"
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            title="Amount ($)"
        )

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Behavioral indicators

    st.markdown("""
    <div class="section-header">
        <span>📊</span> Behavioral Indicators
    </div>
    """, unsafe_allow_html=True)


    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "24h Transactions",
            f"{target_row['tx_count_last_24h']}"
        )


    with m2:

        st.metric(
            "Amount / Avg",
            f"{target_row['amt_vs_rolling_avg']:.2f}x"
        )


    with m3:

        st.metric(

            "Time Since Prev",

            (
                f"{target_row['time_since_last_tx']:.0f}s"
                if target_row["time_since_last_tx"] > 0
                else "N/A"
            )

        )


    st.markdown(f"""
    <div style="
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 14px;
        border-radius: 8px;
        margin-top: 8px;
        font-size: 0.78rem;
        color: #94a3b8;
    ">
        ⚡ <strong>Velocity Cutoff:</strong>
        {cutoffs['tx24_cutoff']} tx / 24h
        &nbsp;•&nbsp;

        📈 <strong>Spending Spike Cutoff:</strong>
        {cutoffs['spike_cutoff']:.2f}x
    </div>
    """, unsafe_allow_html=True)


# =======================================================
# MODEL PERFORMANCE
# =======================================================

st.markdown(
    "<br><hr style='border-color: rgba(255,255,255,0.08);'><br>",
    unsafe_allow_html=True
)


st.markdown("""
<div class="section-header">
    <span>📈</span> Model Performance Metrics (Internal Test Set)
</div>
""", unsafe_allow_html=True)


perf1, perf2, perf3, perf4 = st.columns(4)


with perf1:

    st.metric(
        "ROC-AUC",
        f"{meta['test_auc']:.4f}"
    )


with perf2:

    st.metric(
        "Precision",
        f"{meta['test_precision']:.4f}"
    )


with perf3:

    st.metric(
        "Recall",
        f"{meta['test_recall']:.4f}"
    )


with perf4:

    st.metric(
        "F1-Score",
        f"{meta['test_f1']:.4f}"
    )


# =======================================================
# CONFUSION MATRIX
# =======================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


st.markdown("""
<div class="section-header">
    <span>📊</span> Confusion Matrix
</div>
""", unsafe_allow_html=True)


X_test_full = df_test[feature_cols]

y_test_full = df_test["isFraud"]


test_probabilities = model.predict_proba(
    X_test_full
)[:, 1]


test_predictions = (
    test_probabilities >= threshold
).astype(int)


cm = confusion_matrix(
    y_test_full,
    test_predictions
)


tn, fp, fn, tp = cm.ravel()


cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual Legitimate",
        "Actual Fraud"
    ],

    columns=[
        "Predicted Legitimate",
        "Predicted Fraud"
    ]

)


fig_cm = px.imshow(

    cm_df,

    text_auto=True,

    aspect="auto",

    color_continuous_scale="Viridis",

    labels={
        "x": "Model Prediction",
        "y": "Actual Class",
        "color": "Transactions"
    },

    title="Fraud Detection Confusion Matrix"

)


fig_cm.update_layout(

    template="plotly_dark",

    paper_bgcolor="rgba(15, 23, 42, 0)",

    plot_bgcolor="rgba(15, 23, 42, 0.5)",

    font=dict(
        family="Plus Jakarta Sans, sans-serif",
        color="#94a3b8"
    ),

    title_font=dict(
        size=13,
        color="#f8fafc"
    ),

    height=380,

    margin=dict(
        l=15,
        r=15,
        t=45,
        b=15
    )

)


st.plotly_chart(
    fig_cm,
    use_container_width=True
)


cm1, cm2, cm3, cm4 = st.columns(4)


with cm1:

    st.metric(
        "True Negatives",
        f"{tn:,}"
    )


with cm2:

    st.metric(
        "False Positives",
        f"{fp:,}"
    )


with cm3:

    st.metric(
        "False Negatives",
        f"{fn:,}"
    )


with cm4:

    st.metric(
        "True Positives",
        f"{tp:,}"
    )


# =======================================================
# AUDIT LOG
# =======================================================

st.markdown(
    "<br><hr style='border-color: rgba(255,255,255,0.08);'><br>",
    unsafe_allow_html=True
)


st.markdown("""
<div class="section-header">
    <span>📜</span> Audit Trail
</div>
""", unsafe_allow_html=True)


logs = get_audit_logs()


if logs:

    st.dataframe(
        pd.DataFrame(logs),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No investigation decisions have been recorded yet."
    )


# =======================================================
# FOOTER
# =======================================================

st.markdown("""
<div class="footer-modern">
    🛡️ Sentinel Risk Manager
    &nbsp;•&nbsp;
    Rewind. Investigate. Defend.
    &nbsp;•&nbsp;
    AI-assisted fraud investigation console
</div>
""", unsafe_allow_html=True)
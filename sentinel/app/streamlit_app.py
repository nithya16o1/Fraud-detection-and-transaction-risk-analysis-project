import os

import sys

import json

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st

import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sentinel.src.features.timeline import get_card_timeline

from sentinel.src.evidence.synthesizer import synthesize_evidence, compute_empirical_cutoffs

from sentinel.src.audit.logger import log_audit_entry, get_audit_logs

st.set_page_config(page_title="Sentinel — Dispute Investigation", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 800; color: #0F172A; }
    .sub-header { font-size: 1.1rem; font-weight: 600; color: #0EA5E9; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Sentinel Risk Manager</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Rewind. Investigate. Defend.</div>', unsafe_allow_html=True)

st.divider()

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

    # Load canonical clean dataset and saved test set (FIX #4)

    clean_path = os.path.join("sentinel", "data", "processed", "cleaned_transactions.parquet")

    test_path = os.path.join("sentinel", "data", "processed", "test_set.parquet")

    df_full = pd.read_parquet(clean_path)

    df_test = pd.read_parquet(test_path)

    cutoffs = compute_empirical_cutoffs(df_full)

    return df_full, df_test, cutoffs

model, meta = load_artifacts()

df_full, df_test, cutoffs = load_datasets()

threshold = meta["chosen_threshold"]

st.sidebar.header("🔍 Dispute Case Selector")

sample_txs = df_test.sample(min(100, len(df_test)), random_state=42)

selected_tx_id = st.sidebar.selectbox("Select Disputed Transaction ID:", sample_txs['TransactionID'].tolist())

target_row = df_test[df_test['TransactionID'] == selected_tx_id].iloc[0]

feature_cols = meta["feature_cols"]

X_single = pd.DataFrame([target_row[feature_cols]])

ml_score = float(model.predict_proba(X_single)[:, 1][0])

col1, col2 = st.columns([1, 2])

with col1:

    st.subheader("📋 Case Summary")

    st.write(f"**Transaction ID:** `{selected_tx_id}`")

    st.write(f"**Amount:** `${target_row['TransactionAmt']:.2f}`")

    st.write(f"**Entity Proxy:** `{target_row['entity_id']}`")

    st.divider()

    st.subheader("🎯 ML Risk Assessment")

    st.metric(label="ML Fraud Risk Score", value=f"{ml_score:.4f}", delta=f"Cutoff: {threshold:.2f}")

    if ml_score >= threshold:

        st.error("⚠️ HIGH FRAUD RISK FLAGGED")

    else:

        st.success("✅ LOW FRAUD RISK SCORE")

    evidence_res = synthesize_evidence(

        transaction_id=selected_tx_id,

        amount=target_row['TransactionAmt'],

        ml_score=ml_score,

        threshold=threshold,

        amt_vs_avg=target_row['amt_vs_rolling_avg'],

        tx_24h=target_row['tx_count_last_24h'],

        spike_cutoff=cutoffs['spike_cutoff'],

        tx24_cutoff=cutoffs['tx24_cutoff']

    )

    st.subheader("🤖 AI Evidence Synthesis")

    st.write(f"**Primary Concern:** {evidence_res['primary_concern']}")

    st.write(f"**Secondary Concern:** {evidence_res['secondary_concern']}")

    st.write("**Evidence Facts:**")

    for ev in evidence_res['evidence']:

        st.write(f"- {ev}")

    st.info(f"💡 **AI Recommended Action:** `{evidence_res['recommendation']}`")

    st.divider()

    st.subheader("✍️ Merchant Final Decision")

    decision = st.radio("Select Action:", ["Contest Dispute", "Send to Review", "Accept Loss"])

    if st.button("Submit Decision & Log to Audit"):

        log_audit_entry(

            transaction_id=selected_tx_id,

            amount=target_row['TransactionAmt'],

            ml_score=ml_score,

            threshold=threshold,

            llm_recommendation=evidence_res['recommendation'],

            rule_recommendation=evidence_res['rule_recommendation'],

            merchant_decision=decision,

            disagreement_flag=evidence_res['disagreement_flag'],

            evidence_summary=evidence_res['evidence']

        )

        st.success(f"Decision '{decision}' recorded to audit log!")

with col2:

    st.subheader("⏳ Behavioral Timeline (Fraud Time Machine)")

    # Renders timeline from FULL canonical dataset for complete history display (FIX #4)

    history_df, stats = get_card_timeline(df_full, selected_tx_id)

    fig = px.scatter(

        history_df,

        x='TransactionDT',

        y='TransactionAmt',

        size='TransactionAmt',

        color='isFraud',

        hover_data=['TransactionID', 'TransactionAmt'],

        title=f"Entity `{target_row['entity_id']}` History Timeline"

    )

    fig.add_trace(go.Scatter(

        x=[target_row['TransactionDT']],

        y=[target_row['TransactionAmt']],

        mode='markers',

        marker=dict(size=18, color='gold', symbol='star'),

        name='Disputed Target Tx'

    ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Key Velocity Deviation Indicators")

    m1, m2, m3 = st.columns(3)

    m1.metric("24h Transaction Count", f"{target_row['tx_count_last_24h']}")

    m2.metric("Spike Ratio (vs Rolling Avg)", f"{target_row['amt_vs_rolling_avg']:.2f}x")

    m3.metric("Time Since Last Tx", f"{target_row['time_since_last_tx']:.0f}s" if target_row['time_since_last_tx'] > 0 else "N/A")

st.divider()

st.subheader("📜 Running Audit Log Trail")

logs = get_audit_logs()

if logs:

    st.dataframe(pd.DataFrame(logs))

else:

    st.write("No audit entries recorded yet.")
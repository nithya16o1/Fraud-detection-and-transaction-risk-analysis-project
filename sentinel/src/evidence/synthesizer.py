import os

import json

import numpy as np

from dotenv import load_dotenv

load_dotenv()


def compute_empirical_cutoffs(df):
    """
    Computes empirical behavioral cutoffs from the dataset.
    """

    fraud_df = df[df["isFraud"] == 1]

    legit_df = df[df["isFraud"] == 0]

    spike_fraud = fraud_df["amt_vs_rolling_avg"].median()

    spike_legit = legit_df["amt_vs_rolling_avg"].median()

    spike_cutoff = max(
        1.0,
        round(
            float((spike_fraud + spike_legit) / 2.0),
            2
        )
    )

    tx24_fraud = fraud_df["tx_count_last_24h"].median()

    tx24_legit = legit_df["tx_count_last_24h"].median()

    tx24_cutoff = max(
        1,
        int(
            round(
                float((tx24_fraud + tx24_legit) / 2.0)
            )
        )
    )

    return {
        "spike_cutoff": spike_cutoff,
        "tx24_cutoff": tx24_cutoff
    }


def synthesize_evidence(
    transaction_id,
    amount,
    ml_score,
    threshold,
    amt_vs_avg,
    tx_24h,
    time_since_last_tx,
    spike_cutoff,
    tx24_cutoff
):
    """
    Synthesizes evidence using deterministic rules or an LLM.
    """

    high_ml_risk = ml_score >= threshold

    high_velocity = tx_24h >= tx24_cutoff

    spending_spike = amt_vs_avg > spike_cutoff

    if high_ml_risk:

        primary_concern = (
            f"High ML fraud risk detected "
            f"(score: {ml_score:.4f}; "
            f"threshold: {threshold:.2f})"
        )

    else:

        primary_concern = (
            f"Low ML fraud risk score "
            f"({ml_score:.4f}; "
            f"threshold: {threshold:.2f})"
        )

    if tx_24h == 0 and time_since_last_tx <= 0:

        secondary_concern = (
            "Insufficient transaction history "
            "to establish a baseline"
        )

    elif high_velocity:

        secondary_concern = (
            f"High transaction velocity detected "
            f"({tx_24h} transactions in the last 24h; "
            f"cutoff: {tx24_cutoff})"
        )

    elif spending_spike:

        secondary_concern = (
            f"Recent spending spike detected "
            f"({amt_vs_avg:.2f}x average)"
        )

    else:

        secondary_concern = (
            "Normal historical spending velocity"
        )

    evidence = [
        primary_concern,
        secondary_concern,
        f"Transaction amount: ${amount:.2f}",
        f"24h transaction count: {tx_24h}",
        f"Amount vs historical average: {amt_vs_avg:.2f}x"
    ]

    if high_ml_risk and (high_velocity or spending_spike):

        rule_recommendation = "Send to Review"

    elif high_ml_risk:

        rule_recommendation = "Send to Review"

    elif not high_ml_risk and not high_velocity and not spending_spike:

        rule_recommendation = "Contest Dispute"

    else:

        rule_recommendation = "Accept Loss"

    recommendation = rule_recommendation

    disagreement_flag = False

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        return {
            "primary_concern": primary_concern,
            "secondary_concern": secondary_concern,
            "evidence": evidence,
            "recommendation": recommendation,
            "rule_recommendation": rule_recommendation,
            "disagreement_flag": disagreement_flag
        }

    try:

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        model_name = os.getenv(
            "LLM_MODEL",
            "gpt-4o-mini"
        )

        prompt = f"""
You are a fraud investigation assistant.

Analyze the transaction using ONLY the provided evidence.

Transaction ID: {transaction_id}
Transaction Amount: ${amount:.2f}
ML Fraud Score: {ml_score:.4f}
Decision Threshold: {threshold:.2f}
Amount vs Historical Average: {amt_vs_avg:.2f}x
Transactions in Previous 24h: {tx_24h}
Time Since Last Transaction: {time_since_last_tx}
Velocity Cutoff: {tx24_cutoff}
Spending Spike Cutoff: {spike_cutoff}

Return JSON with:
primary_concern
secondary_concern
evidence
recommendation

Possible recommendations:
- Contest Dispute
- Send to Review
- Accept Loss

Do not invent evidence.
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful fraud investigation "
                        "assistant. Use only supplied evidence."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        llm_result = json.loads(content)

        llm_recommendation = llm_result.get(
            "recommendation",
            rule_recommendation
        )

        disagreement_flag = (
            llm_recommendation != rule_recommendation
        )

        return {
            "primary_concern": llm_result.get(
                "primary_concern",
                primary_concern
            ),
            "secondary_concern": llm_result.get(
                "secondary_concern",
                secondary_concern
            ),
            "evidence": llm_result.get(
                "evidence",
                evidence
            ),
            "recommendation": llm_recommendation,
            "rule_recommendation": rule_recommendation,
            "disagreement_flag": disagreement_flag
        }

    except Exception:

        return {
            "primary_concern": primary_concern,
            "secondary_concern": secondary_concern,
            "evidence": evidence,
            "recommendation": recommendation,
            "rule_recommendation": rule_recommendation,
            "disagreement_flag": disagreement_flag
        }
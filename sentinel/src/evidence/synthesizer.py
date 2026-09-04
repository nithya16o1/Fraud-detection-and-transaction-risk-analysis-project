import os

import json

import pandas as pd

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """

You are Sentinel AI Risk Manager — an evidence synthesizer for payment dispute investigations.

Input: Numerical risk score from an ML model and historical timeline deviation stats.

YOUR STRICT MANDATE: Describe ONLY what the numerical input data explicitly shows.

Do NOT invent backstory, fake user behavior, or unprovided facts.

Return a JSON object with this EXACT structure:

{

  "primary_concern": "string",

  "secondary_concern": "string",

  "evidence": ["list of numerical evidence facts"],

  "confidence": "High" | "Medium" | "Low",

  "recommendation": "Contest" | "Review" | "Accept"

}

"""

def compute_empirical_cutoffs(df: pd.DataFrame) -> dict:

    """

    Computes dynamic feature cutoffs based on the midpoints between

    fraud and legitimate transaction medians in the dataset (FIX #3).

    """

    fraud_mask = (df['isFraud'] == 1)

    legit_mask = (df['isFraud'] == 0)

    spike_fraud = df.loc[fraud_mask, 'amt_vs_rolling_avg'].median()

    spike_legit = df.loc[legit_mask, 'amt_vs_rolling_avg'].median()

    tx24_fraud = df.loc[fraud_mask, 'tx_count_last_24h'].median()

    tx24_legit = df.loc[legit_mask, 'tx_count_last_24h'].median()

    spike_cutoff = round(float((spike_fraud + spike_legit) / 2.0), 2)

    tx24_cutoff = max(1, int(round((tx24_fraud + tx24_legit) / 2.0)))

    print("\n[EMPIRICAL CUTOFF DERIVATION]")

    print(f"  - Amount Spike Ratio -> Fraud Median: {spike_fraud:.2f}x | Legit Median: {spike_legit:.2f}x => Derived Cutoff: {spike_cutoff:.2f}x")

    print(f"  - 24h Tx Velocity   -> Fraud Median: {tx24_fraud:.0f} | Legit Median: {tx24_legit:.0f} => Derived Cutoff: {tx24_cutoff}")

    return {"spike_cutoff": spike_cutoff, "tx24_cutoff": tx24_cutoff}

def deterministic_policy_recommendation(

    ml_score: float,

    threshold: float,

    amt_vs_avg: float,

    tx_24h: int,

    spike_cutoff: float = 2.0,

    tx24_cutoff: int = 2

) -> str:

    """

    Deterministic rule engine serving as fallback and sanity check.

    Cutoffs are dynamically derived from empirical dataset medians (FIX #3).

    """

    if ml_score >= threshold:

        if amt_vs_avg >= spike_cutoff or tx_24h >= tx24_cutoff:

            return "Review"

        else:

            return "Accept"

    else:

        return "Contest"

def synthesize_evidence(

    transaction_id: int,

    amount: float,

    ml_score: float,

    threshold: float,

    amt_vs_avg: float,

    tx_24h: int,

    spike_cutoff: float = 2.0,

    tx24_cutoff: int = 2

):

    rule_rec = deterministic_policy_recommendation(ml_score, threshold, amt_vs_avg, tx_24h, spike_cutoff, tx24_cutoff)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_openai_api_key_here":

        evidence_list = [

            f"Transaction #{transaction_id} Amount: ${amount:.2f}",

            f"ML Fraud Score: {ml_score:.4f} (Cutoff Threshold: {threshold:.2f})",

            f"Spending Velocity: {amt_vs_avg:.2f}x card's historical average (Empirical Cutoff: {spike_cutoff:.2f}x)",

            f"24-Hour Velocity: {tx_24h} prior transactions in last 24h (Empirical Cutoff: {tx24_cutoff})"

        ]

        return {

            "primary_concern": f"ML fraud score ({ml_score:.2f}) exceeds threshold ({threshold:.2f})" if ml_score >= threshold else "Low ML risk probability score",

            "secondary_concern": f"Recent spending spike detected ({amt_vs_avg:.2f}x average)" if amt_vs_avg >= spike_cutoff else "Normal historical spending velocity",

            "evidence": evidence_list,

            "confidence": "High",

            "recommendation": rule_rec,

            "rule_recommendation": rule_rec,

            "disagreement_flag": False

        }

    try:

        client = OpenAI(api_key=api_key)

        prompt = f"""

        Transaction ID: {transaction_id}

        Disputed Amount: ${amount:.2f}

        ML Fraud Risk Score: {ml_score:.4f} (Threshold: {threshold:.2f})

        Amount vs Rolling Avg Ratio: {amt_vs_avg:.2f}x (Cutoff: {spike_cutoff:.2f}x)

        Transactions in Last 24h: {tx_24h} (Cutoff: {tx24_cutoff})

        """

        res = client.chat.completions.create(

            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),

            response_format={"type": "json_object"},

            messages=[

                {"role": "system", "content": SYSTEM_PROMPT},

                {"role": "user", "content": prompt}

            ],

            temperature=0.1

        )

        result = json.loads(res.choices[0].message.content)

        disagreement = (result.get("recommendation") != rule_rec)

        result["rule_recommendation"] = rule_rec

        result["disagreement_flag"] = disagreement

        return result

    except Exception as e:

        print(f"[WARN] OpenAI call failed ({e}). Using deterministic fallback rule.")

        return synthesize_evidence(transaction_id, amount, ml_score, threshold, amt_vs_avg, tx_24h, spike_cutoff, tx24_cutoff)
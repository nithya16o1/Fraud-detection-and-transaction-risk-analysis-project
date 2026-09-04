import os

import json

from datetime import datetime

AUDIT_LOG_PATH = os.path.join("sentinel", "data", "audit_log.jsonl")

def log_audit_entry(

    transaction_id: int,

    amount: float,

    ml_score: float,

    threshold: float,

    llm_recommendation: str,

    rule_recommendation: str,

    merchant_decision: str,

    disagreement_flag: bool,

    evidence_summary: list

):

    entry = {

        "timestamp": datetime.utcnow().isoformat() + "Z",

        "transaction_id": int(transaction_id),

        "amount": float(amount),

        "ml_score": round(float(ml_score), 4),

        "threshold": round(float(threshold), 2),

        "llm_recommendation": llm_recommendation,

        "rule_recommendation": rule_recommendation,

        "merchant_decision": merchant_decision,

        "disagreement_flag": disagreement_flag,

        "evidence_summary": evidence_summary

    }

    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)

    with open(AUDIT_LOG_PATH, "a") as f:

        f.write(json.dumps(entry) + "\n")

    print(f"[AUDIT LOGGED] Tx #{transaction_id} -> Merchant Decision: {merchant_decision}")

def get_audit_logs():

    if not os.path.exists(AUDIT_LOG_PATH):

        return []

    logs = []

    with open(AUDIT_LOG_PATH, "r") as f:

        for line in f:

            if line.strip():

                logs.append(json.loads(line.strip()))

    return logs
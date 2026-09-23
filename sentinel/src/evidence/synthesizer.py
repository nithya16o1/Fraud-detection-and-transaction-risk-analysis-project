from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
import os
import json
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# STAGE 1: VALIDATION METRICS
# ============================================================

VALIDATION_METRICS: Dict[str, int] = {
    "first_try_success": 0,
    "retry_success": 0,
    "fallback_used": 0,
}


def get_validation_metrics() -> Dict[str, int]:
    """Return current structured-output validation metrics."""
    return VALIDATION_METRICS.copy()


def reset_validation_metrics() -> None:
    """Reset structured-output validation metrics."""
    global VALIDATION_METRICS

    VALIDATION_METRICS = {
        "first_try_success": 0,
        "retry_success": 0,
        "fallback_used": 0,
    }


# ============================================================
# STAGE 1: PYDANTIC SCHEMA
# ============================================================

class EvidenceOutput(BaseModel):
    """Pydantic model enforcing evidence JSON shape and value constraints."""

    primary_concern: str = Field(
        ...,
        description="Main reason for risk classification",
    )

    secondary_concern: str = Field(
        ...,
        description="Secondary anomaly or context factor",
    )

    evidence: List[str] = Field(
        ...,
        description="List of numerical evidence facts",
    )

    confidence: Literal["High", "Medium", "Low"] = Field(
        ...,
        description="Confidence tier of explanation",
    )

    recommendation: Literal["Contest", "Review", "Accept"] = Field(
        ...,
        description="Recommended merchant action",
    )

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, value: Any) -> str:
        """
        Normalize legacy recommendation strings into the
        canonical Sentinel values.
        """

        if not isinstance(value, str):
            return value

        value = value.strip()

        if "contest" in value.lower():
            return "Contest"

        if "review" in value.lower():
            return "Review"

        if "accept" in value.lower():
            return "Accept"

        return value


# ============================================================
# LLM SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Sentinel AI Risk Manager — an evidence synthesizer for
payment dispute investigations.

Input:
- Numerical risk score from an ML model
- Historical timeline deviation statistics

STRICT RULE:
Describe ONLY what the numerical input data explicitly shows.

Do NOT invent:
- customer backstory
- user behavior
- transaction history that was not provided
- unsupported facts

Return a JSON object with EXACTLY this structure:

{
    "primary_concern": "string",
    "secondary_concern": "string",
    "evidence": ["list of string facts"],
    "confidence": "High" | "Medium" | "Low",
    "recommendation": "Contest" | "Review" | "Accept"
}
"""


# ============================================================
# BEHAVIORAL CUTOFF CALCULATION
# ============================================================

def compute_empirical_cutoffs(df):
    """
    Computes empirical behavioral cutoffs from the dataset.
    """

    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]

    spike_fraud = (
        fraud_df["amt_vs_rolling_avg"].median()
        if not fraud_df.empty
        else 2.5
    )

    spike_legit = (
        legit_df["amt_vs_rolling_avg"].median()
        if not legit_df.empty
        else 1.0
    )

    spike_cutoff = max(
        1.0,
        round(
            float((spike_fraud + spike_legit) / 2.0),
            2,
        ),
    )

    tx24_fraud = (
        fraud_df["tx_count_last_24h"].median()
        if not fraud_df.empty
        else 3.0
    )

    tx24_legit = (
        legit_df["tx_count_last_24h"].median()
        if not legit_df.empty
        else 0.0
    )

    tx24_cutoff = max(
        1,
        int(
            round(
                float((tx24_fraud + tx24_legit) / 2.0)
            )
        ),
    )

    return {
        "spike_cutoff": spike_cutoff,
        "tx24_cutoff": tx24_cutoff,
    }


# ============================================================
# DETERMINISTIC RULE ENGINE
# ============================================================

def deterministic_policy_recommendation(
    ml_score: float,
    threshold: float,
    amt_vs_avg: float,
    tx_24h: int,
    spike_cutoff: float = 2.0,
    tx24_cutoff: int = 2,
) -> str:
    """
    Deterministic rule engine used as:
    1. fallback
    2. sanity check against LLM recommendation
    """

    high_ml_risk = ml_score >= threshold
    high_velocity = tx_24h >= tx24_cutoff
    spending_spike = amt_vs_avg > spike_cutoff

    if high_ml_risk:
        return "Review"

    elif not high_ml_risk and not high_velocity and not spending_spike:
        return "Contest"

    else:
        return "Accept"


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def build_deterministic_fallback(
    transaction_id: int,
    amount: float,
    ml_score: float,
    threshold: float,
    amt_vs_avg: float,
    tx_24h: int,
    time_since_last_tx: float,
    spike_cutoff: float,
    tx24_cutoff: int,
) -> Dict[str, Any]:
    """
    Generates a deterministic response matching EvidenceOutput.
    """

    high_ml_risk = ml_score >= threshold
    high_velocity = tx_24h >= tx24_cutoff
    spending_spike = amt_vs_avg > spike_cutoff

    # Primary concern
    if high_ml_risk:
        primary = (
            f"High ML fraud risk score "
            f"({ml_score:.4f}; threshold: {threshold:.2f})"
        )
    else:
        primary = (
            f"Low ML fraud risk score "
            f"({ml_score:.4f}; threshold: {threshold:.2f})"
        )

    # Secondary concern
    if tx_24h == 0 and time_since_last_tx <= 0:
        secondary = (
            "Insufficient transaction history "
            "to establish baseline"
        )

    elif high_velocity:
        secondary = (
            f"High transaction velocity "
            f"({tx_24h} tx in last 24h; cutoff: {tx24_cutoff})"
        )

    elif spending_spike:
        secondary = (
            f"Recent spending spike detected "
            f"({amt_vs_avg:.2f}x average)"
        )

    else:
        secondary = "Normal historical spending velocity"

    # Evidence
    evidence_facts = [
        primary,
        secondary,
        f"Transaction amount: ${amount:.2f}",
        f"24h transaction count: {tx_24h}",
        f"Amount vs historical average: {amt_vs_avg:.2f}x",
    ]

    # Deterministic recommendation
    rule_rec = deterministic_policy_recommendation(
        ml_score=ml_score,
        threshold=threshold,
        amt_vs_avg=amt_vs_avg,
        tx_24h=tx_24h,
        spike_cutoff=spike_cutoff,
        tx24_cutoff=tx24_cutoff,
    )

    # Confidence based on distance from threshold
    confidence = (
        "High"
        if (
            ml_score >= threshold + 0.2
            or ml_score <= threshold - 0.2
        )
        else "Medium"
    )

    return {
        "primary_concern": primary,
        "secondary_concern": secondary,
        "evidence": evidence_facts,
        "confidence": confidence,
        "recommendation": rule_rec,
        "rule_recommendation": rule_rec,
        "disagreement_flag": False,
    }


# ============================================================
# MAIN EVIDENCE SYNTHESIS
# ============================================================

def synthesize_evidence(
    transaction_id: int,
    amount: float,
    ml_score: float,
    threshold: float,
    amt_vs_avg: float,
    tx_24h: int,
    time_since_last_tx: float = 0.0,
    spike_cutoff: float = 2.0,
    tx24_cutoff: int = 2,
) -> Dict[str, Any]:
    """
    Evidence synthesis pipeline:

    1. Calculate deterministic rule recommendation.
    2. Call LLM.
    3. Validate LLM output using Pydantic.
    4. Retry once if validation fails.
    5. Fall back to deterministic output if retry fails.
    """

    # --------------------------------------------------------
    # Deterministic rule recommendation
    # --------------------------------------------------------

    rule_rec = deterministic_policy_recommendation(
        ml_score=ml_score,
        threshold=threshold,
        amt_vs_avg=amt_vs_avg,
        tx_24h=tx_24h,
        spike_cutoff=spike_cutoff,
        tx24_cutoff=tx24_cutoff,
    )

    # --------------------------------------------------------
    # API key check
    # --------------------------------------------------------

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_openai_api_key_here":
        logger.warning(
            "OPENAI_API_KEY missing. Using deterministic fallback."
        )

        VALIDATION_METRICS["fallback_used"] += 1

        return build_deterministic_fallback(
            transaction_id=transaction_id,
            amount=amount,
            ml_score=ml_score,
            threshold=threshold,
            amt_vs_avg=amt_vs_avg,
            tx_24h=tx_24h,
            time_since_last_tx=time_since_last_tx,
            spike_cutoff=spike_cutoff,
            tx24_cutoff=tx24_cutoff,
        )

    # --------------------------------------------------------
    # LLM call
    # --------------------------------------------------------

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        model_name = os.getenv(
            "LLM_MODEL",
            "gpt-4o-mini",
        )

        user_prompt = f"""
Transaction ID: {transaction_id}

Disputed Amount: ${amount:.2f}

ML Fraud Risk Score: {ml_score:.4f}
Threshold: {threshold:.2f}

Amount vs Rolling Average:
{amt_vs_avg:.2f}x

Transactions in Last 24h:
{tx_24h}

Velocity Cutoff:
{tx24_cutoff}

Spending Spike Cutoff:
{spike_cutoff:.2f}x

Time Since Last Transaction:
{time_since_last_tx}s
"""

        # ====================================================
        # FIRST TRY
        # ====================================================

        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content

        try:
            raw_json = json.loads(content)

            parsed = EvidenceOutput.model_validate(raw_json)

            VALIDATION_METRICS["first_try_success"] += 1

            result = parsed.model_dump()

            disagreement = (
                result["recommendation"] != rule_rec
            )

            result["rule_recommendation"] = rule_rec
            result["disagreement_flag"] = disagreement

            return result

        except (
            ValidationError,
            json.JSONDecodeError,
        ) as validation_error:

            logger.warning(
                "Schema validation failed on first try: "
                f"{validation_error}. Retrying once."
            )

            # =================================================
            # RETRY
            # =================================================

            retry_prompt = f"""
{user_prompt}

CRITICAL FIX REQUIRED:

Your previous response failed Pydantic schema validation.

Validation error:
{str(validation_error)}

Return a corrected JSON object.

Required fields:

- primary_concern: string
- secondary_concern: string
- evidence: list of strings
- confidence: "High", "Medium", or "Low"
- recommendation: "Contest", "Review", or "Accept"

Do not add unsupported facts.
"""

            retry_response = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": retry_prompt,
                    },
                ],
                temperature=0.0,
            )

            retry_content = (
                retry_response.choices[0].message.content
            )

            raw_retry_json = json.loads(retry_content)

            parsed_retry = EvidenceOutput.model_validate(
                raw_retry_json
            )

            VALIDATION_METRICS["retry_success"] += 1

            result = parsed_retry.model_dump()

            disagreement = (
                result["recommendation"] != rule_rec
            )

            result["rule_recommendation"] = rule_rec
            result["disagreement_flag"] = disagreement

            return result

    # --------------------------------------------------------
    # FINAL FALLBACK
    # --------------------------------------------------------

    except Exception as error:

        logger.error(
            "LLM validation/call failed after retry. "
            f"Using deterministic fallback. Error: {error}"
        )

        VALIDATION_METRICS["fallback_used"] += 1

        return build_deterministic_fallback(
            transaction_id=transaction_id,
            amount=amount,
            ml_score=ml_score,
            threshold=threshold,
            amt_vs_avg=amt_vs_avg,
            tx_24h=tx_24h,
            time_since_last_tx=time_since_last_tx,
            spike_cutoff=spike_cutoff,
            tx24_cutoff=tx24_cutoff,
        )
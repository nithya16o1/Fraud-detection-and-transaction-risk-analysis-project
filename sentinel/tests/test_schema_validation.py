import pytest
import json
from pydantic import ValidationError
from sentinel.src.evidence.synthesizer import (
    EvidenceOutput,
    synthesize_evidence,
    get_validation_metrics,
    reset_validation_metrics,
    build_deterministic_fallback
)

def test_evidence_output_valid_model():
    """Verify that EvidenceOutput validates valid dictionary structures correctly."""
    valid_data = {
        "primary_concern": "High ML score 0.8500",
        "secondary_concern": "Velocity spike of 4 tx/24h",
        "evidence": ["High score", "Velocity spike"],
        "confidence": "High",
        "recommendation": "Review"
    }
    model = EvidenceOutput.model_validate(valid_data)
    assert model.recommendation == "Review"
    assert model.confidence == "High"
    assert len(model.evidence) == 2

def test_evidence_output_recommendation_normalization():
    """Verify that legacy string recommendations map cleanly to canonical enums."""
    data_contest = {
        "primary_concern": "Low risk",
        "secondary_concern": "Normal velocity",
        "evidence": ["Fact 1"],
        "confidence": "High",
        "recommendation": "Contest Dispute"
    }
    model = EvidenceOutput.model_validate(data_contest)
    assert model.recommendation == "Contest"

    data_review = {
        "primary_concern": "Risk",
        "secondary_concern": "Spike",
        "evidence": ["Fact 1"],
        "confidence": "Medium",
        "recommendation": "Send to Review"
    }
    model2 = EvidenceOutput.model_validate(data_review)
    assert model2.recommendation == "Review"

def test_evidence_output_schema_validation_error():
    """Verify that missing required fields raise Pydantic ValidationError."""
    invalid_data = {
        "primary_concern": "High ML score",
        # missing secondary_concern, evidence, confidence, recommendation
    }
    with pytest.raises(ValidationError):
        EvidenceOutput.model_validate(invalid_data)

def test_synthesizer_fallback_on_missing_api_key(monkeypatch):
    """Verify that without OPENAI_API_KEY, synthesizer uses deterministic fallback and logs fallback_used."""
    reset_validation_metrics()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    res = synthesize_evidence(
        transaction_id=101,
        amount=150.0,
        ml_score=0.85,
        threshold=0.70,
        amt_vs_avg=3.5,
        tx_24h=4,
        time_since_last_tx=120.0,
        spike_cutoff=2.0,
        tx24_cutoff=2
    )

    metrics = get_validation_metrics()
    assert metrics["fallback_used"] >= 1
    assert res["recommendation"] == "Review"
    assert "primary_concern" in res
    assert "evidence" in res
    assert isinstance(res["evidence"], list)

def test_deterministic_fallback_schema_compliance():
    """Verify that build_deterministic_fallback produces a schema-valid response."""
    fallback = build_deterministic_fallback(
        transaction_id=202,
        amount=50.0,
        ml_score=0.20,
        threshold=0.70,
        amt_vs_avg=1.0,
        tx_24h=0,
        time_since_last_tx=3600.0,
        spike_cutoff=2.0,
        tx24_cutoff=2
    )
    validated = EvidenceOutput.model_validate(fallback)
    assert validated.recommendation == "Contest"
    assert validated.confidence in ["High", "Medium", "Low"]

import json
import pathlib
import sys

import pytest

# Import diagnosis as a top-level module, bypassing standissect/__init__.py
# (which imports scanpy via .cluster). Requires diagnosis.py's dual-import.
_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]   # .../standissect
sys.path.insert(0, str(_PKG_DIR))

import diagnosis  # noqa: E402
from diagnosis import (parse_llm_result, _diagnosis_from_dict,  # noqa: E402
                       CallableChatClient, ALLOWED_CAUSES,
                       make_chat_client, make_diagnosis_engine, ArkChatClient)
from llm_client import call_structured, LLMUnavailable  # noqa: E402


def test_parse_llm_result_happy_path():
    raw = json.dumps({"likely_cause": ALLOWED_CAUSES[0], "confidence": 0.8,
                      "rationale": "because evidence"})
    r = parse_llm_result(raw, rule_baseline=None, mode="llm", model="m")
    assert r.likely_cause == ALLOWED_CAUSES[0]
    assert r.diagnosis_source == "llm"
    assert r.confidence == 0.8


def test_parse_llm_result_strips_fences():
    raw = "```json\n" + json.dumps({"likely_cause": ALLOWED_CAUSES[0],
                                    "rationale": "r"}) + "\n```"
    r = parse_llm_result(raw, rule_baseline=None, mode="llm", model=None)
    assert r.likely_cause == ALLOWED_CAUSES[0]


def test_call_structured_with_callable_client_builds_result():
    payload = json.dumps({"likely_cause": ALLOWED_CAUSES[0],
                          "confidence": 0.6, "rationale": "r"})
    client = CallableChatClient(lambda s, u: payload)
    r = call_structured(client, "sys", "usr",
                        lambda d: _diagnosis_from_dict(d, rule_baseline=None,
                                                       mode="llm", model=None))
    assert r.likely_cause == ALLOWED_CAUSES[0]


def test_bad_enum_becomes_llm_unavailable():
    client = CallableChatClient(lambda s, u: json.dumps({"likely_cause": "NOT_A_CAUSE"}))
    with pytest.raises(LLMUnavailable):
        call_structured(client, "s", "u",
                        lambda d: _diagnosis_from_dict(d, rule_baseline=None,
                                                       mode="llm", model=None))


def test_make_chat_client_returns_none_for_rule():
    assert make_chat_client(mode="rule") is None


def test_make_chat_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    assert make_chat_client(mode="llm", ark_api_key_env="ARK_API_KEY") is None


def test_make_chat_client_wraps_callable():
    c = make_chat_client(mode="llm", llm_client=lambda s, u: "{}")
    assert isinstance(c, CallableChatClient)
    assert hasattr(c, "complete")


def test_make_chat_client_builds_ark_with_key():
    c = make_chat_client(mode="llm", ark_api_key="secret", ark_model="mymodel")
    assert isinstance(c, ArkChatClient)
    assert c.model == "mymodel"


def test_make_diagnosis_engine_still_raises_on_llm_without_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        make_diagnosis_engine(mode="llm")


def test_diagnosis_retries_then_succeeds():
    calls = {"n": 0}
    def flaky(s, u):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("read timeout")
        return json.dumps({"likely_cause": ALLOWED_CAUSES[0], "confidence": 0.7,
                           "rationale": "r"})
    from diagnosis import LLMDiagnosisEngine, MinorEvidence
    eng = LLMDiagnosisEngine(CallableChatClient(flaky, model="m"), mode="llm",
                             llm_retries=3)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    assert r.diagnosis_source == "llm"
    assert calls["n"] == 3


def test_diagnosis_retry_exhaustion_falls_back_to_rule():
    def always_timeout(s, u):
        raise TimeoutError("read timeout")
    from diagnosis import LLMDiagnosisEngine, MinorEvidence
    eng = LLMDiagnosisEngine(CallableChatClient(always_timeout, model="m"), mode="llm",
                             llm_retries=2)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    assert r.diagnosis_source == "rule-fallback"
    assert r.error is not None


def test_prompt_advertises_disposition_proposed_and_eleven_causes():
    from diagnosis import build_llm_prompt, MinorEvidence, ALLOWED_CAUSES
    ev = MinorEvidence(parent_cluster="0", subcluster="c0_1",
                       reference_subcluster="c0_0", minor_umap_label="u1",
                       main_umap_label="u0", n_cells=10, frac_of_parent=0.1)
    system, user = build_llm_prompt(ev, mode="llm")
    assert len(ALLOWED_CAUSES) == 11
    for key in ("recommended_disposition", "disposition_reason",
                "proposed_cell_type", "cause_signatures"):
        assert key in user
    assert "escalate" in (system + user).lower()


def test_rule_engine_result_carries_disposition_baseline():
    from diagnosis import RuleDiagnosisEngine, MinorEvidence
    eng = RuleDiagnosisEngine(discard_confidence_threshold=0.5)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    assert r.recommended_disposition == r.disposition_baseline
    assert r.disposition_overridden is False
    assert r.proposed_cell_type is None


def test_llm_disposition_parsed_and_clamped():
    payload = json.dumps({"likely_cause": "cell-cycle", "confidence": 0.9,
                          "rationale": "cycling", "recommended_disposition": "DISCARD",
                          "disposition_reason": "looks junky"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.recommended_disposition == "KEEP"
    assert r.disposition_overridden is False


def test_llm_missing_disposition_falls_back_to_baseline():
    payload = json.dumps({"likely_cause": "doublet-driven", "confidence": 0.9,
                          "rationale": "doublets"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.recommended_disposition == "DISCARD"
    assert r.proposed_cell_type is None


def test_llm_proposed_cell_type_parsed():
    payload = json.dumps({"likely_cause": "biology-candidate", "confidence": 0.8,
                          "rationale": "distinct program",
                          "proposed_cell_type": "pDC"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.proposed_cell_type == "pDC"
    assert r.recommended_disposition == "KEEP"


def test_new_cause_dissociation_is_accepted_by_llm():
    payload = json.dumps({"likely_cause": "dissociation-effect", "confidence": 0.8,
                          "rationale": "HSP + IEG", "recommended_disposition": "DISCARD"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.likely_cause == "dissociation-effect"
    assert r.recommended_disposition == "DISCARD"


def test_llm_relaxation_through_parse_layer_beats_post_init_default():
    # doublet-driven baseline = DISCARD; LLM relaxes to UNCERTAIN (allowed).
    # Without the _diagnosis_from_dict finalize wiring, __post_init__ would leave DISCARD.
    payload = json.dumps({"likely_cause": "doublet-driven", "confidence": 0.9,
                          "rationale": "actually fine",
                          "recommended_disposition": "UNCERTAIN",
                          "disposition_reason": "looks like real cells"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.recommended_disposition == "UNCERTAIN"
    assert r.disposition_overridden is True

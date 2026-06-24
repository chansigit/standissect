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
                       CallableChatClient, ALLOWED_CAUSES)
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

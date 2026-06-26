import pathlib
import sys

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_DIR))

import diagnosis  # noqa: E402
from diagnosis import (ALLOWED_CAUSES, DISPOSITION_MAP, derive_disposition,  # noqa: E402
                       DiagnosisResult)

NEW_CAUSES = {'dissociation-effect', 'cell-cycle', 'ambient-contamination',
              'sex-driven', 'interferon-response'}


def test_taxonomy_has_eleven_causes_including_new_ones():
    assert len(ALLOWED_CAUSES) == 11
    assert NEW_CAUSES.issubset(set(ALLOWED_CAUSES))


def test_disposition_map_covers_exactly_allowed_causes():
    assert set(DISPOSITION_MAP) == set(ALLOWED_CAUSES)
    assert set(DISPOSITION_MAP.values()) <= {'DISCARD', 'KEEP', 'UNCERTAIN'}


def test_baseline_mapping_per_cause():
    for c in ('doublet-driven', 'low-quality (high mt)', 'shallow-depth',
              'dissociation-effect', 'ambient-contamination'):
        assert DISPOSITION_MAP[c] == 'DISCARD'
    for c in ('cell-cycle', 'sex-driven', 'interferon-response', 'biology-candidate'):
        assert DISPOSITION_MAP[c] == 'KEEP'
    for c in ('sample-driven', 'unclear'):
        assert DISPOSITION_MAP[c] == 'UNCERTAIN'


def test_gate_downgrades_low_confidence_discard():
    assert derive_disposition('doublet-driven', 0.3, threshold=0.5)[:3] == (
        'UNCERTAIN', 'DISCARD', True)


def test_gate_keeps_high_confidence_discard():
    assert derive_disposition('doublet-driven', 0.9, threshold=0.5)[:3] == (
        'DISCARD', 'DISCARD', False)


def test_override_relax_toward_keep_is_accepted():
    assert derive_disposition('doublet-driven', 0.9, threshold=0.5,
                              llm_disposition='KEEP', llm_reason='real')[:3] == (
        'KEEP', 'DISCARD', True)


def test_override_escalate_toward_discard_is_rejected():
    final, baseline, overridden, reason = derive_disposition(
        'cell-cycle', 0.9, threshold=0.5, llm_disposition='DISCARD',
        llm_reason='looks junky')
    assert (final, baseline, overridden) == ('KEEP', 'KEEP', False)
    assert 'rejected' in reason.lower()


def test_uncertain_baseline_cannot_be_escalated_to_discard():
    assert derive_disposition('unclear', 0.9, threshold=0.5,
                              llm_disposition='DISCARD')[0] == 'UNCERTAIN'


def test_result_disposition_and_proposed_fields_round_trip():
    r = DiagnosisResult(likely_cause='cell-cycle', confidence=0.9,
                        proposed_cell_type='pDC')
    assert r.disposition_baseline == 'KEEP'
    assert r.recommended_disposition == 'KEEP'
    assert r.proposed_cell_type == 'pDC'
    pf = r.to_panel_fields()
    for k in ('recommended_disposition', 'disposition_baseline',
              'disposition_overridden', 'disposition_reason', 'proposed_cell_type'):
        assert k in pf
    assert pf['proposed_cell_type'] == 'pDC'


def test_finalize_applies_gate():
    r = DiagnosisResult(likely_cause='doublet-driven', confidence=0.2)
    r.finalize_disposition(0.5)
    assert r.recommended_disposition == 'UNCERTAIN'
    assert r.disposition_overridden is True
    assert r.proposed_cell_type is None

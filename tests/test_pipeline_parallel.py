import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))
from standissect import pipeline  # noqa: E402   (scanpy import -> runs on this compute node)


def test_dissect_helper_signature_is_global_free():
    # the refactored helper must accept explicit subset+ctx (no _DISSECT_CTX)
    import inspect
    params = inspect.signature(pipeline._dissect_one_subset).parameters
    assert 'subset' in params and 'ctx' in params

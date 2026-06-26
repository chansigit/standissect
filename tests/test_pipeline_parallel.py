import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))
from standissect import pipeline  # noqa: E402   (scanpy import -> runs on this compute node)


def test_dissect_helper_signature_is_global_free():
    # the refactored helper must accept explicit subset+ctx (no _DISSECT_CTX)
    import inspect
    params = inspect.signature(pipeline._dissect_one_subset).parameters
    assert 'subset' in params and 'ctx' in params


def test_subset_categorical_cast_pickles_for_spawn():
    # Regression: a pandas Categorical column (in obs OR var) raises
    # NotImplementedError in NDArrayBacked.__setstate__ when unpickled in a
    # spawn worker, which would silently force the dissect stage onto its serial
    # fallback (so cross-cluster DEG parallelism never actually runs). Casting
    # obs *and* var Categorical columns to object — exactly what the dissect
    # subset marshaling does — must make the subset round-trip through pickle.
    import pickle
    import numpy as np
    import pandas as pd
    import anndata as ad

    obs = pd.DataFrame({
        'cluster': pd.Categorical(['a', 'a', 'b', 'b']),
        'batch': pd.Categorical(['10X_P7_2', '10X_P7_3', '10X_P7_2', '10X_P7_3']),
    })
    # var carries a Categorical gene-annotation column (the marrow data's
    # gene_id_harmonized / gene_symbol_harmonized are stored this way).
    var = pd.DataFrame(
        {'gene_id_harmonized': pd.Categorical(['ENSMUSG1', 'ENSMUSG2', 'ENSMUSG3'])},
        index=['g0', 'g1', 'g2'])
    subset = ad.AnnData(X=np.zeros((4, 3), dtype=np.float32), obs=obs, var=var)

    # raw categorical (obs+var) fails to unpickle (the bug we guard against)
    try:
        pickle.loads(pickle.dumps(subset))
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "expected NotImplementedError unpickling raw Categorical obs/var"

    # apply the same cast the pipeline applies before handing subsets to spawn
    def _decat(frame):
        for _col in frame.columns:
            if isinstance(frame[_col].dtype, pd.CategoricalDtype):
                frame[_col] = frame[_col].astype(object)
    _decat(subset.obs)
    _decat(subset.var)

    # now it round-trips, and the categorical columns are plain object dtype
    rt = pickle.loads(pickle.dumps(subset))
    assert rt.obs['cluster'].dtype == object
    assert rt.obs['batch'].dtype == object
    assert rt.var['gene_id_harmonized'].dtype == object
    assert list(rt.obs['cluster']) == ['a', 'a', 'b', 'b']
    assert list(rt.var['gene_id_harmonized']) == ['ENSMUSG1', 'ENSMUSG2', 'ENSMUSG3']

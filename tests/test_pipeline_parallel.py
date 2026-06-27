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
    pipeline._decat(subset.obs)
    pipeline._decat(subset.var)

    # now it round-trips, and the categorical columns are plain object dtype
    rt = pickle.loads(pickle.dumps(subset))
    assert rt.obs['cluster'].dtype == object
    assert rt.obs['batch'].dtype == object
    assert rt.var['gene_id_harmonized'].dtype == object
    assert list(rt.obs['cluster']) == ['a', 'a', 'b', 'b']
    assert list(rt.obs['batch']) == ['10X_P7_2', '10X_P7_3', '10X_P7_2', '10X_P7_3']
    assert list(rt.var['gene_id_harmonized']) == ['ENSMUSG1', 'ENSMUSG2', 'ENSMUSG3']


def test_annotation_composition_for_subcluster():
    import numpy as np
    import pandas as pd
    import anndata as ad
    obs = pd.DataFrame({
        'original_cluster_split': ['c3_0', 'c3_0', 'c3_1', 'c3_1', 'c3_1', 'c3_1'],
        'cell_ontology_class': ['macrophage', 'macrophage', 'B cell', 'B cell',
                                'B cell', 'T cell'],
    })
    adata = ad.AnnData(X=np.zeros((6, 2), dtype=np.float32), obs=obs)

    minor = pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c3_1',
        annotation_col='cell_ontology_class')
    assert minor[0] == {'annotation': 'B cell', 'n_cells': 3, 'frac': 0.75}
    assert minor[1] == {'annotation': 'T cell', 'n_cells': 1, 'frac': 0.25}

    main = pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c3_0',
        annotation_col='cell_ontology_class')
    assert main == [{'annotation': 'macrophage', 'n_cells': 2, 'frac': 1.0}]


def test_annotation_composition_graceful_when_unavailable():
    import numpy as np
    import pandas as pd
    import anndata as ad
    obs = pd.DataFrame({'original_cluster_split': ['c0_0', 'c0_1']})
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32), obs=obs)
    # no annotation_col requested
    assert pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c0_1',
        annotation_col=None) == []
    # requested column missing from obs
    assert pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c0_1',
        annotation_col='nope') == []
    # adata is None (diagnosis reused off persisted artifacts)
    assert pipeline._annotation_composition_for_subcluster(
        None, subcluster_col='original_cluster_split', label='c0_1',
        annotation_col='x') == []
    # label with no matching cells
    assert pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c9_9',
        annotation_col='original_cluster_split') == []


def test_run_dissect_pipeline_rejects_missing_annotation_col(tmp_path):
    import numpy as np
    import pandas as pd
    import anndata as ad
    import pytest
    obs = pd.DataFrame({'leiden': ['0', '0', '1', '1']})
    adata = ad.AnnData(X=np.zeros((4, 3), dtype=np.float32), obs=obs)
    adata.obsm['X_umap'] = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(KeyError, match="annotation_col"):
        pipeline.run_dissect_pipeline(
            adata, cluster_col='leiden', output_dir=str(tmp_path),
            annotation_col='does_not_exist')


def test_annotation_composition_skips_blank_and_nan():
    import numpy as np
    import pandas as pd
    import anndata as ad
    obs = pd.DataFrame({
        'original_cluster_split': ['c1_1'] * 5,
        # 2 real "B cell", then blank, whitespace-only, and NaN (all noise)
        'free_annotation': ['B cell', 'B cell', '', '  ', np.nan],
    })
    adata = ad.AnnData(X=np.zeros((5, 2), dtype=np.float32), obs=obs)
    comp = pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c1_1',
        annotation_col='free_annotation')
    # only the real label survives; frac is over the FULL fragment (5 cells),
    # so it honestly reflects that 2/5 are annotated.
    assert comp == [{'annotation': 'B cell', 'n_cells': 2, 'frac': 0.4}]


def test_annotation_composition_all_blank_is_empty():
    import numpy as np
    import pandas as pd
    import anndata as ad
    obs = pd.DataFrame({
        'original_cluster_split': ['c1_0'] * 3,
        'free_annotation': ['', '   ', np.nan],
    })
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32), obs=obs)
    # all blank/NaN -> [] so the LLM "ignore when empty" policy path fires
    # (regression for the real Marrow e2e: free_annotation was blank).
    assert pipeline._annotation_composition_for_subcluster(
        adata, subcluster_col='original_cluster_split', label='c1_0',
        annotation_col='free_annotation') == []

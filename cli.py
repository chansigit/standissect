"""Command line interface for standissect."""
from __future__ import annotations

import argparse
import sys

from .diagnosis import DEFAULT_ARK_ENDPOINT, DEFAULT_ARK_MODEL
from .pipeline import run_dissect_pipeline
from .report import build_report


def _none_if_empty(value):
    return value if value else None


def _add_common_run_args(parser):
    parser.add_argument('h5ad', help='Input AnnData .h5ad file.')
    parser.add_argument('--cluster-col', required=True,
                        help='Existing cluster column in adata.obs.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory where standissect outputs are written.')
    parser.add_argument('--labeled-h5ad-path',
                        help='Optional output .h5ad with standissect labels.')
    parser.add_argument('--apply-discard', metavar='PATH',
                        help='Write a cleaned .h5ad with recommended_disposition==DISCARD '
                             'cells removed (KEEP and UNCERTAIN are retained) to this exact '
                             'path. Off when omitted.')
    parser.add_argument('--umap-key', default='X_umap',
                        help='Embedding key in adata.obsm. Default: X_umap.')
    parser.add_argument('--annotation-col',
                        help='Existing cell-type annotation column in adata.obs. When '
                             'set, each fragment\'s per-cell annotation composition is '
                             'given to the LLM diagnosis as a consistency-check prior '
                             '(not blindly trusted). Must already exist in obs. Off '
                             'when omitted.')

    meta = parser.add_argument_group('metadata roles')
    meta.add_argument('--sample-col',
                      help='Sample/source column used for source-driven diagnosis.')
    meta.add_argument('--batch-col',
                      help='Batch column used for source-driven diagnosis.')
    meta.add_argument('--donor-col',
                      help='Donor column used for source-driven diagnosis.')
    meta.add_argument('--library-col',
                      help='Library/preparation column used for source-driven diagnosis.')
    meta.add_argument('--condition-col',
                      help='Condition column used as composition evidence only.')
    meta.add_argument('--doublet-score-col',
                      help='Continuous doublet/hybrid score column.')
    meta.add_argument('--mito-col',
                      help='Continuous mitochondrial fraction/percent column.')
    meta.add_argument('--feature-count-col',
                      help='Continuous detected-feature/gene-count column.')
    meta.add_argument('--umi-count-col',
                      help='Continuous UMI/read-count depth column.')
    meta.add_argument('--extra-cat-col', action='append', default=[],
                      help='Additional categorical evidence column. Repeatable.')
    meta.add_argument('--extra-qc-col', action='append', default=[],
                      help='Additional continuous QC evidence column. Repeatable.')

    tuning = parser.add_argument_group('partition and statistics')
    tuning.add_argument('--resolution', type=float, default=0.5,
                        help='Initial Leiden resolution. Default: 0.5.')
    tuning.add_argument('--target-k', type=int,
                        help='Target number of global UMAP fragments.')
    tuning.add_argument('--target-tol', type=int, default=2,
                        help='Allowed difference from target_k. Default: 2.')
    tuning.add_argument('--n-neighbors', type=int, default=30,
                        help='UMAP kNN graph neighbors. Default: 30.')
    tuning.add_argument('--min-subcluster-size', type=int, default=50,
                        help='Minimum off-core fragment size to diagnose.')
    tuning.add_argument('--top-n-deg', type=int, default=50,
                        help='Top DEG rows to keep per minor. Default: 50.')
    tuning.add_argument('--top-n-canonical', type=int, default=50,
                        help='Top canonical-core marker rows per group. Default: 50.')
    tuning.add_argument('--deg-layer',
                        help='Optional DEG expression layer, e.g. counts_recovered.')

    diag = parser.add_argument_group('diagnosis')
    diag.add_argument('--diagnosis-mode', choices=('rule', 'llm', 'hybrid'),
                      default='llm', help='Diagnosis engine. Default: llm '
                      '(falls back to rule when no ARK key is available).')
    diag.add_argument('--annotation-hint', default='',
                      help='Optional tissue/context hint for naming + narrative, '
                           'e.g. "synovial tissue, OA vs RA".')
    diag.add_argument('--naming-markers',
                      help='Optional TSV (cell_type<TAB>gene,gene,...) for the '
                           'local naming backup. Defaults to a bundled marker set.')
    diag.add_argument('--ark-model', default=DEFAULT_ARK_MODEL,
                      help=f'Ark model for LLM modes. Default: {DEFAULT_ARK_MODEL}.')
    diag.add_argument('--ark-endpoint', default=DEFAULT_ARK_ENDPOINT,
                      help='Ark chat-completions endpoint.')
    diag.add_argument('--ark-api-key-env', default='ARK_API_KEY',
                      help='Environment variable containing the Ark API key.')
    diag.add_argument('--no-diagnosis-fallback', action='store_true',
                      help='Fail instead of falling back to rule diagnosis on LLM errors.')
    diag.add_argument('--llm-concurrency', type=int, default=8,
                      help='Concurrent ARK calls for diagnosis/naming/narrative. Default: 8.')
    diag.add_argument('--llm-retries', type=int, default=3,
                      help='Retries (exp backoff + jitter) before fallback. Default: 3.')
    diag.add_argument('--ark-timeout', type=int, default=120,
                      help='Per-call ARK timeout (seconds). Default: 120.')
    diag.add_argument('--discard-confidence-threshold', type=float, default=0.5,
                      help='DISCARD calls below this diagnosis confidence are '
                           'downgraded to UNCERTAIN (kept + flagged). Default: 0.5.')

    rerun = parser.add_argument_group('rerun control')
    rerun.add_argument('--force', action='append', default=[],
                       choices=('partition', 'dissect', 'diagnosis', 'canonical',
                                'naming', 'narrative', 'profile', 'all'),
                       help='Stage to recompute. Repeatable; use all for every stage.')
    rerun.add_argument('--random-state', type=int, default=0,
                       help='Random seed for Leiden. Default: 0.')
    rerun.add_argument('--n-jobs', type=int, default=8,
                       help='DEG process-pool size (cross-cluster parallelism).')
    rerun.add_argument('--no-report', action='store_true',
                       help='Skip building report.html after the run.')


def _force_value(values):
    if not values:
        return ()
    if 'all' in values:
        return 'all'
    return tuple(values)


def run_cmd(args):
    try:
        import anndata as ad
    except ImportError as e:
        raise SystemExit("anndata is required for `standissect run`") from e

    adata = ad.read_h5ad(args.h5ad)
    result = run_dissect_pipeline(
        adata,
        cluster_col=args.cluster_col,
        output_dir=args.output_dir,
        labeled_h5ad_path=args.labeled_h5ad_path,
        apply_discard_path=args.apply_discard,
        umap_key=args.umap_key,
        annotation_col=_none_if_empty(args.annotation_col),
        sample_col=_none_if_empty(args.sample_col),
        batch_col=_none_if_empty(args.batch_col),
        donor_col=_none_if_empty(args.donor_col),
        library_col=_none_if_empty(args.library_col),
        condition_col=_none_if_empty(args.condition_col),
        doublet_score_col=_none_if_empty(args.doublet_score_col),
        mito_col=_none_if_empty(args.mito_col),
        feature_count_col=_none_if_empty(args.feature_count_col),
        umi_count_col=_none_if_empty(args.umi_count_col),
        extra_cat_cols=tuple(args.extra_cat_col or ()),
        extra_qc_cols=tuple(args.extra_qc_col or ()),
        resolution=args.resolution,
        target_k=args.target_k,
        target_tol=args.target_tol,
        n_neighbors=args.n_neighbors,
        min_subcluster_size=args.min_subcluster_size,
        top_n_deg=args.top_n_deg,
        top_n_canonical=args.top_n_canonical,
        deg_layer=args.deg_layer,
        diagnosis_mode=args.diagnosis_mode,
        diagnosis_ark_model=args.ark_model,
        diagnosis_ark_endpoint=args.ark_endpoint,
        diagnosis_ark_api_key_env=args.ark_api_key_env,
        diagnosis_fallback_to_rule=not args.no_diagnosis_fallback,
        annotation_hint=args.annotation_hint,
        naming_markers=_none_if_empty(args.naming_markers),
        force=_force_value(args.force),
        n_jobs=args.n_jobs,
        llm_concurrency=args.llm_concurrency,
        llm_retries=args.llm_retries,
        diagnosis_timeout=args.ark_timeout,
        discard_confidence_threshold=args.discard_confidence_threshold,
        random_state=args.random_state,
    )
    if not args.no_report:
        report = build_report(result['root'])
        print(f"[standissect] report: {report}")
    print(f"[standissect] root: {result['root']}")
    return 0


def report_cmd(args):
    print(build_report(args.output_root, args.output_html))
    return 0


def columns_cmd(args):
    try:
        import anndata as ad
    except ImportError as e:
        raise SystemExit("anndata is required for `standissect columns`") from e
    adata = ad.read_h5ad(args.h5ad, backed='r')
    print("obs columns:")
    for col in adata.obs.columns:
        print(f"  {col}")
    print("\nobsm keys:")
    for key in adata.obsm.keys():
        print(f"  {key}")
    adata.file.close()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='standissect',
        description='Diagnose minor fragments inside existing single-cell clusters.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='Run the full standissect pipeline.')
    _add_common_run_args(run)
    run.set_defaults(func=run_cmd)

    report = sub.add_parser('report', help='Build self-contained report.html.')
    report.add_argument('output_root', help='A standissect output root.')
    report.add_argument('--output-html', help='Optional report HTML path.')
    report.set_defaults(func=report_cmd)

    columns = sub.add_parser('columns', help='List h5ad obs columns and obsm keys.')
    columns.add_argument('h5ad', help='Input AnnData .h5ad file.')
    columns.set_defaults(func=columns_cmd)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

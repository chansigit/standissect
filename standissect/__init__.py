"""standissect — cluster cleanup-diagnosis for single-cell data.

Dissects a clustering on its UMAP-Leiden fragments: per-cluster DEG / QC /
composition drift, canonical-core markers, minor-anatomy heatmaps, and a
self-contained HTML report — orchestrated with file-existence idempotency.

    from standissect import run_dissect_pipeline, build_report
    result = run_dissect_pipeline(adata, cluster_col='leiden', output_dir='out')
    build_report(result['root'])

Submodules:
    standissect.cluster   — analysis primitives (partition, DEG, heatmaps)
    standissect.pipeline  — run_dissect_pipeline orchestrator + unified output
    standissect.report    — single-file HTML report builder
"""
from .cluster import (
    umap_leiden_partition,
    dissect_one_cluster,
    canonical_marker_deg,
    plot_minor_anatomy,
    wilcoxon_one_vs_rest,
    wilcoxon_vs_reference,
)
from .pipeline import run_dissect_pipeline
from .report import build_report

__version__ = "0.1.0"
__all__ = [
    "run_dissect_pipeline",
    "build_report",
    "umap_leiden_partition",
    "dissect_one_cluster",
    "canonical_marker_deg",
    "plot_minor_anatomy",
    "wilcoxon_one_vs_rest",
    "wilcoxon_vs_reference",
]

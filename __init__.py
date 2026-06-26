"""standissect — cluster cleanup-diagnosis for single-cell data.

Dissects a clustering on its UMAP-Leiden fragments: per-cluster DEG / QC /
composition drift, rule/LLM diagnosis, canonical-core markers, minor-profile
heatmaps, and a self-contained HTML report — orchestrated with file-existence
idempotency.

    from standissect import run_dissect_pipeline, build_report
    result = run_dissect_pipeline(adata, cluster_col='leiden', output_dir='out')
    build_report(result['root'])

Submodules:
    standissect.cluster   — analysis primitives (partition, DEG, heatmaps)
    standissect.diagnosis — rule baseline + LLM/hybrid interpretation
    standissect.pipeline  — run_dissect_pipeline orchestrator + unified output
    standissect.report    — single-file HTML report builder
"""
from .cluster import (
    umap_leiden_partition,
    dissect_one_cluster,
    canonical_marker_deg,
    plot_minor_profile,
    wilcoxon_one_vs_rest,
    wilcoxon_vs_reference,
)
from .pipeline import run_dissect_pipeline
from .report import build_report
from .diagnosis import (
    ArkChatClient,
    DiagnosisResult,
    MinorEvidence,
    RuleDiagnosisEngine,
    build_minor_evidence,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
)

__version__ = "0.1.1"
__all__ = [
    "run_dissect_pipeline",
    "build_report",
    "ArkChatClient",
    "DiagnosisResult",
    "MinorEvidence",
    "RuleDiagnosisEngine",
    "build_minor_evidence",
    "make_diagnosis_engine",
    "normalize_diagnosis_roles",
    "umap_leiden_partition",
    "dissect_one_cluster",
    "canonical_marker_deg",
    "plot_minor_profile",
    "wilcoxon_one_vs_rest",
    "wilcoxon_vs_reference",
]

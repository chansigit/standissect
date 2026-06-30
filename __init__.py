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
    make_chat_client,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
)
from .annotate import (
    ClusterNarrative,
    CoreNaming,
    LocalNamingEngine,
    NarrativeEngine,
    make_naming_engine,
)


def serve(*args, **kwargs):
    """Launch the interactive review server (lazy: imports fastapi only here)."""
    from .webreview import serve as _serve
    return _serve(*args, **kwargs)


def build_app(*args, **kwargs):
    """Build the review FastAPI app (lazy import of fastapi)."""
    from .webreview import build_app as _build_app
    return _build_app(*args, **kwargs)


def export_cell_coords(*args, **kwargs):
    """Export per-cell UMAP coords for the interactive UMAP (lazy import)."""
    from .export_coords import export_cell_coords as _ecc
    return _ecc(*args, **kwargs)


__version__ = "0.1.1"
__all__ = [
    "run_dissect_pipeline",
    "serve",
    "build_app",
    "export_cell_coords",
    "build_report",
    "ArkChatClient",
    "DiagnosisResult",
    "MinorEvidence",
    "RuleDiagnosisEngine",
    "build_minor_evidence",
    "make_chat_client",
    "make_diagnosis_engine",
    "normalize_diagnosis_roles",
    "ClusterNarrative",
    "CoreNaming",
    "LocalNamingEngine",
    "NarrativeEngine",
    "make_naming_engine",
    "umap_leiden_partition",
    "dissect_one_cluster",
    "canonical_marker_deg",
    "plot_minor_profile",
    "wilcoxon_one_vs_rest",
    "wilcoxon_vs_reference",
]

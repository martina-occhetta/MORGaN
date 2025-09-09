"""Analysis helper package.

Exports key helpers for biological analysis and plotting.
"""

from .bio import (
    load_results,
    select_gene_set,
    annotate_protein_classes,
    enrich_pathways,
    screen_immuno_targets,
    load_network_from_csv,
    network_context,
    load_druggable_symbols,
)

from .plots import (
    barplot_class_counts,
    barh_enrichment,
    dotplot_enrichment,
    network_subgraph_plot,
    scatter_score_vs_degree,
    histogram_distance_to_ref,
)

__all__ = [
    # bio
    "load_results",
    "select_gene_set",
    "annotate_protein_classes",
    "enrich_pathways",
    "screen_immuno_targets",
    "load_network_from_csv",
    "network_context",
    "load_druggable_symbols",
    # plots
    "barplot_class_counts",
    "barh_enrichment",
    "dotplot_enrichment",
    "network_subgraph_plot",
    "scatter_score_vs_degree",
    "histogram_distance_to_ref",
]

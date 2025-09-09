from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def _get_ax(ax=None, figsize: Tuple[int, int] | None = None):
    if ax is not None:
        return ax
    return plt.subplots(figsize=figsize or (6, 4))[1]



def barplot_class_counts(counts: pd.Series | dict, ax=None, title: Optional[str] = None, top_n: Optional[int] = None):
    """Bar plot of protein class counts.

    counts: Series or dict mapping class -> count
    """
    if isinstance(counts, dict):
        counts = pd.Series(counts)
    counts = counts.sort_values(ascending=False)
    if top_n is not None and len(counts) > top_n:
        counts = counts.head(int(top_n))
    ax = _get_ax(ax, (6, 3))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C78A8")
    ax.set_ylabel("Count")
    ax.set_xlabel("Protein class")
    ax.set_title(title or "Protein class counts")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return ax

def _parse_overlap_to_ratio(overlap: str) -> float:
    if not isinstance(overlap, str) or "/" not in overlap:
        return np.nan
    try:
        a, b = overlap.split("/")
        a, b = float(a), float(b)
        if b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def _ensure_enrichr_df(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "term":
            ren[c] = "Term"
        elif lc in {"adjusted p-value", "adjusted pvalue", "adj p", "padj", "fdr"}:
            ren[c] = "Adjusted P-value"
        elif lc in {"p-value", "pvalue"}:
            ren[c] = "P-value"
        elif lc.replace(" ", "") == "combinedscore":
            ren[c] = "Combined Score"
        elif lc == "overlap":
            ren[c] = "Overlap"
    out = df.rename(columns=ren).copy()
    if "Gene_set" in out.columns and "Term" not in out.columns:
        out.rename(columns={"Gene_set": "Term"}, inplace=True)
    return out


def barh_enrichment(enrichr_df: pd.DataFrame, ax=None, top_n: int = 15, use: str = "Combined Score"):
    """Horizontal bar plot for enrichment results (e.g., Enrichr).

    use: metric to plot, tries 'Combined Score', else -log10 adj p, else -log10 p
    """
    df = _ensure_enrichr_df(enrichr_df)
    if use not in df.columns:
        if "Adjusted P-value" in df.columns:
            df["-log10(FDR)"] = -np.log10(df["Adjusted P-value"].replace(0, np.nextafter(0, 1)))
            use = "-log10(FDR)"
        elif "P-value" in df.columns:
            df["-log10(p)"] = -np.log10(df["P-value"].replace(0, np.nextafter(0, 1)))
            use = "-log10(p)"
        else:
            if "Overlap" in df.columns:
                df["k"] = df["Overlap"].apply(lambda s: float(str(s).split("/")[0]) if isinstance(s, str) and "/" in s else np.nan)
                use = "k"
            else:
                raise ValueError("Could not find a numeric column to plot")
    cols_needed = [c for c in ["Term", use] if c in df.columns]
    d = df[cols_needed].dropna()
    d = d.sort_values(use, ascending=False).head(int(top_n))
    ax = _get_ax(ax, (8, 6))
    sns.barplot(data=d, y="Term", x=use, ax=ax, color="#72B7B2")
    ax.set_ylabel("")
    ax.set_xlabel(use)
    ax.set_title("Top enriched pathways")
    plt.tight_layout()
    return ax


def dotplot_enrichment(enrichr_df: pd.DataFrame, ax=None, top_n: int = 15):
    """Dot plot for enrichment results showing gene ratio vs significance.

    Uses 'Overlap' to compute Gene Ratio and plots vs -log10(FDR).
    """
    df = _ensure_enrichr_df(enrichr_df)
    if "Overlap" not in df.columns:
        raise ValueError("Expected 'Overlap' column in enrichment results")
    d = df.copy()
    d["Gene Ratio"] = d["Overlap"].apply(_parse_overlap_to_ratio)
    if "Adjusted P-value" in d.columns:
        d["-log10(FDR)"] = -np.log10(d["Adjusted P-value"].replace(0, np.nextafter(0, 1)))
        sig_col = "-log10(FDR)"
    elif "P-value" in d.columns:
        d["-log10(p)"] = -np.log10(d["P-value"].replace(0, np.nextafter(0, 1)))
        sig_col = "-log10(p)"
    else:
        raise ValueError("No p-value column found in enrichment results")
    keep = d[["Term", "Gene Ratio", sig_col]].dropna()
    keep = keep.sort_values(sig_col, ascending=False).head(int(top_n)).sort_values("Gene Ratio", ascending=True)
    ax = _get_ax(ax, (8, 6))
    sns.scatterplot(data=keep, x="Gene Ratio", y="Term", size=sig_col, hue=sig_col, palette="viridis", ax=ax, legend=True)
    ax.set_title("Enrichment dot plot")
    ax.set_xlabel("Gene Ratio")
    ax.set_ylabel("")
    plt.tight_layout()
    return ax


# ---- Network subgraph visualization ----

def network_subgraph_plot(
    G: nx.Graph,
    genes: Iterable[str],
    ref_genes: Optional[Iterable[str]] = None,
    hops: int = 1,
    max_nodes: int = 200,
    layout: str = "spring",
    seed: int = 42,
    ax=None,
):
    """Visualize the induced k-hop subgraph around the input genes.

    Colors:
      - predicted genes: blue
      - reference genes (e.g., druggable): orange
      - both: green
      - neighbors/others: light gray
    Node size ~ degree within the subgraph.
    """
    genes = [g for g in genes if g in G]
    ref = set([r for r in (ref_genes or []) if r in G])
    nodes = set()
    for g in genes:
        nodes.add(g)
        fringe = {g}
        for _ in range(max(0, hops)):
            nbrs = set()
            for u in fringe:
                nbrs.update(G.neighbors(u))
            nodes.update(nbrs)
            fringe = nbrs
    H = G.subgraph(nodes).copy()
    if H.number_of_nodes() > max_nodes:
        keep = set(genes)
        nbr_deg = sorted(((n, H.degree(n)) for n in H.nodes if n not in keep), key=lambda x: x[1], reverse=True)
        for n, _ in nbr_deg:
            if len(keep) >= max_nodes:
                break
            keep.add(n)
        H = H.subgraph(keep).copy()

    ax = _get_ax(ax, (8, 6))
    if layout == "spring":
        pos = nx.spring_layout(H, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.spring_layout(H, seed=seed)

    colors = []
    sizes = []
    for n in H.nodes:
        in_pred = n in genes
        in_ref = n in ref
        if in_pred and in_ref:
            colors.append("#2ca02c")  
        elif in_pred:
            colors.append("#1f77b4")  
        elif in_ref:
            colors.append("#ff7f0e")  
        else:
            colors.append("#d3d3d3")  
        sizes.append(100 + 30 * H.degree(n))

    nx.draw_networkx_edges(H, pos=pos, ax=ax, alpha=0.15)
    nx.draw_networkx_nodes(H, pos=pos, node_color=colors, node_size=sizes, ax=ax, linewidths=0.2, edgecolors="#444444")
    labels = {n: n for n in genes if n in H}
    nx.draw_networkx_labels(H, pos=pos, labels=labels, font_size=8, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"Network context (hops={hops}, nodes={H.number_of_nodes()})")
    plt.tight_layout()
    return ax


# ---- Utility analytical plots ----

def scatter_score_vs_degree(df: pd.DataFrame, x: str = "Score", y: str = "degree", hue: Optional[str] = None, ax=None):
    ax = _get_ax(ax, (5, 4))
    data = df.copy()
    if hue and hue not in data.columns:
        hue = None
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"{y} vs {x}")
    plt.tight_layout()
    return ax


def histogram_distance_to_ref(df: pd.DataFrame, col: str = "shortest_dist_to_reference", ax=None):
    ax = _get_ax(ax, (5, 3))
    vals = df[col].dropna().values
    sns.histplot(vals, bins=np.arange(vals.min() - 0.5, vals.max() + 1.5, 1), ax=ax)
    ax.set_xlabel("Shortest distance to reference")
    ax.set_ylabel("Count")
    ax.set_title("Network distance to reference genes")
    plt.tight_layout()
    return ax


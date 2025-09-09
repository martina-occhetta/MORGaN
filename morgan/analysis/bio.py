from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import networkx as nx   

def load_results(path: str) -> pd.DataFrame:
    """Load results CSV with columns like Gene, Label, Pred, Score.

    Returns a DataFrame with canonical column names and dtypes.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ren = {}
    if "gene" not in cols:
        for c in df.columns:
            if c.lower() in {"symbol", "gene_symbol", "hgnc", "hgnc_symbol"}:
                ren[c] = "Gene"
                break
    if "Gene" not in df.columns:
        df.rename(columns=ren, inplace=True)
    for c in ["Label", "Pred"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round().astype(int)
    return df


def select_gene_set(
    df: pd.DataFrame,
    which: str = "pred_pos",
    top_n: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[str]:
    """Select a gene set from results.

    which: one of
      - "pred_pos": Pred==1
      - "false_pos": Label==0 & Pred==1
      - "true_pos": Label==1 & Pred==1
      - "all": all genes
      - "top": top by Score (use top_n)
    """
    df2 = df.copy()
    if min_score is not None and "Score" in df2.columns:
        df2 = df2[df2["Score"] >= float(min_score)]
    if which == "pred_pos" and "Pred" in df2.columns:
        df2 = df2[df2["Pred"] == 1]
    elif which == "false_pos" and {"Label", "Pred"} <= set(df2.columns):
        df2 = df2[(df2["Label"] == 0) & (df2["Pred"] == 1)]
    elif which == "true_pos" and {"Label", "Pred"} <= set(df2.columns):
        df2 = df2[(df2["Label"] == 1) & (df2["Pred"] == 1)]
    elif which == "top" and "Score" in df2.columns:
        df2 = df2.sort_values("Score", ascending=False)
        if top_n is not None:
            df2 = df2.head(int(top_n))
    if "Gene" not in df2.columns:
        raise ValueError("Results must contain a 'Gene' column")
    return df2["Gene"].astype(str).dropna().unique().tolist()



GO_CLASS_MAP: Dict[str, List[str]] = {
    "gpcr": ["GO:0004930", "G protein-coupled receptor activity"],
    "kinase": ["GO:0004672", "protein kinase activity", "GO:0016301", "kinase activity"],
    "ion_channel": [
        "GO:0005216",
        "ion channel activity",
        "GO:0022836",
        "gated channel activity",
        "GO:0005244",
        "voltage-gated",
    ],
    "transcription_factor": ["GO:0003700", "DNA-binding transcription factor activity"],
    "nuclear_receptor": ["GO:0004879", "nuclear receptor activity"],
    "cytokine": ["GO:0005125", "cytokine activity"],
    "cytokine_receptor": ["GO:0004896", "cytokine receptor activity"],
    "transporter": ["GO:0005215", "transporter activity"],
    "enzyme": ["GO:0003824", "catalytic activity"],
    "receptor_tyr_kinase": ["GO:0004714", "transmembrane receptor protein tyrosine kinase activity"],
}


def _match_go_to_classes(go_terms: List[Dict[str, str]]) -> Set[str]:
    classes: Set[str] = set()
    terms = []
    for t in go_terms or []:
        if isinstance(t, dict):
            terms.append(str(t.get("id", "")).lower())
            terms.append(str(t.get("term", "")).lower())
        else:
            terms.append(str(t).lower())
    for cls, patterns in GO_CLASS_MAP.items():
        for p in patterns:
            p = p.lower()
            if any(p in s for s in terms):
                classes.add(cls)
                break
    return classes


def annotate_protein_classes(
    genes: Iterable[str],
    extra_keywords: Optional[Dict[str, List[str]]] = None,
    as_counts: bool = False,
) -> pd.DataFrame:
    """Annotate protein classes using MyGene GO MF terms and UniProt keywords if available.

    Requires internet to query mygene. If not available, returns an empty annotation.

    Returns a DataFrame with columns: Gene, classes (list[str]), and one-hot columns per class.
    If as_counts=True, returns class counts instead of per-gene annotation.
    """
    try:
        import mygene 
    except Exception:
        ann = pd.DataFrame({"Gene": list(genes), "classes": [[] for _ in genes]})
        if as_counts:
            return pd.Series(dtype=int)
        return ann

    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        list(genes),
        scopes="symbol",
        fields=["go.MF", "go.BP", "uniprot.keyword", "type_of_gene"],
        species="human",
        as_dataframe=True,
    )
    res.reset_index(inplace=True)
    res.rename(columns={"query": "Gene"}, inplace=True)

    rows: List[Tuple[str, Set[str]]] = []
    for _, r in res.iterrows():
        gene = str(r.get("Gene", ""))
        classes: Set[str] = set()
        go_mf = r.get("go.MF")
        if isinstance(go_mf, list):
            classes |= _match_go_to_classes(go_mf)
        elif isinstance(go_mf, dict):
            classes |= _match_go_to_classes([go_mf])
        kw = r.get("uniprot.keyword")
        kws = []
        if isinstance(kw, list):
            kws = [str(x).lower() for x in kw]
        elif isinstance(kw, str):
            kws = [kw.lower()]
        if kws:
            if any("g-protein coupled receptor" in k for k in kws):
                classes.add("gpcr")
            if any("kinase" in k for k in kws):
                classes.add("kinase")
            if any("ion channel" in k for k in kws):
                classes.add("ion_channel")
            if any("transcription factor" in k for k in kws):
                classes.add("transcription_factor")
            if any("nuclear receptor" in k for k in kws):
                classes.add("nuclear_receptor")
            if any("cytokine" == k for k in kws) or any("cytokine" in k for k in kws):
                classes.add("cytokine")
        if extra_keywords:
            for cls, patterns in extra_keywords.items():
                for p in patterns:
                    if any(p.lower() in k for k in kws):
                        classes.add(cls)
                        break
        rows.append((gene, classes))

    ann = pd.DataFrame(rows, columns=["Gene", "classes"]).drop_duplicates("Gene")
    all_classes = sorted({c for _, cs in rows for c in cs})
    for c in all_classes:
        ann[c] = ann["classes"].apply(lambda lst: int(c in lst))
    if as_counts:
        counts = ann[all_classes].sum().sort_values(ascending=False)
        return counts
    return ann

def enrich_pathways(
    genes: Iterable[str],
    outdir: Optional[str] = None,
    gene_sets: Optional[List[str]] = None,
    cutoff: float = 0.05,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run Enrichr (gseapy) pathway enrichment for the provided gene list.

    gene_sets: list of Enrichr libraries. Reasonable defaults are used if None.
    Returns mapping library -> results DataFrame (significant only by adjusted p).
    """
    try:
        import gseapy as gp 
    except Exception:
        return {}

    if gene_sets is None:
        gene_sets = [
            "GO_Biological_Process_2021",
            "Reactome_2016",
            "KEGG_2019_Human",
            "WikiPathways_2019_Human",
        ]
    glist = list(dict.fromkeys([g for g in genes if isinstance(g, str) and g]))
    if not glist:
        return {}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}
    for lib in gene_sets:
        try:
            enr = gp.enrichr(gene_list=glist, gene_sets=lib, organism="Human", outdir=outdir or "./")
            df = enr.results.copy()
            if "Adjusted P-value" in df.columns:
                df = df[df["Adjusted P-value"] <= cutoff]
            elif "Adjusted P-value".lower() in (c.lower() for c in df.columns):
                for c in df.columns:
                    if c.lower() == "adjusted p-value":
                        df = df[df[c] <= cutoff]
                        break
            df.sort_values([col for col in df.columns if "P-value" in col or "Combined Score" in col], inplace=True, ascending=True)
            results[lib] = df
        except Exception:
            if verbose:
                print(f"Enrichr failed for {lib}")
    return results



IO_GENE_SETS: Dict[str, Set[str]] = {
    # Immune checkpoints (inhibitory and stimulatory)
    "checkpoints": {
        "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "TIGIT", "HAVCR2",
        "VISTA", "VSIR", "SIGLEC15", "BTLA", "CD96", "IDO1", "IDO2",
        "CD276", "VTCN1", "TNFRSF9", "TNFRSF18", "ICOS", "CD28",
        "TNFRSF4", "TNFSF4", "TNFSF9", "TNFRSF14", "CD40", "CD40LG",
    },
    # Cytokines and receptors (selected)
    "cytokines": {
        "IL2", "IL2RA", "IL2RB", "IL7", "IL7R", "IL12A", "IL12B", "IL12RB1", "IL12RB2",
        "IL15", "IL15RA", "IL21", "IL21R", "IFNG", "IFNGR1", "IFNGR2", "TNF", "TNFRSF1A",
        "TNFRSF1B", "CXCL9", "CXCL10", "CCR5", "CXCR3",
    },
    # Antigen presentation
    "mhc": {
        "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1", "B2M",
        "TAP1", "TAP2",
    },
    # NK/cytotoxic markers
    "cytotoxic": {"PRF1", "GZMB", "GZMA", "NKG2D", "KLRD1", "KLRK1"},
    # TLR/STING
    "pattern_recognition": {"TLR3", "TLR4", "TLR7", "TLR9", "TMEM173"},
}


def screen_immuno_targets(genes: Iterable[str], custom_sets: Optional[Dict[str, Set[str]]] = None) -> pd.DataFrame:
    """Compute overlap of gene list with curated immuno-oncology target sets.

    Returns long-form DataFrame columns: category, gene
    """
    sets = IO_GENE_SETS.copy()
    if custom_sets:
        for k, v in custom_sets.items():
            sets[k] = set(v)
    gene_set = set([g for g in genes if isinstance(g, str) and g])
    rows: List[Tuple[str, str]] = []
    for cat, s in sets.items():
        for g in sorted(gene_set & s):
            rows.append((cat, g))
    return pd.DataFrame(rows, columns=["category", "gene"])



def load_network_from_csv(path: str, threshold: float = 0.0) -> "nx.Graph":
    """Load an adjacency CSV (square matrix with header and row labels) into a NetworkX graph.

    Edges with weight > threshold are added. Assumes symmetrical matrix.
    """
    import networkx as nx

    adj = pd.read_csv(path, index_col=0)
    adj = adj.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nodes = adj.index.intersection(adj.columns)
    adj = adj.loc[nodes, nodes]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    vals = adj.values
    cols = adj.columns.to_list()
    n = len(cols)
    for i in range(n):
        row = vals[i]
        for j in range(i + 1, n):
            w = row[j]
            if w > threshold:
                G.add_edge(cols[i], cols[j], weight=float(w))
    return G


def network_context(
    genes: Iterable[str],
    G: nx.Graph,
    reference_genes: Optional[Set[str]] = None,
    compute_centrality: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Summarize network context for the genes on graph G.

    Returns (per-gene summary DataFrame, global summary metrics).
    Per-gene columns: in_graph, degree, weighted_degree, closeness (optional), shortest_dist_to_reference (if reference provided)
    """
    import networkx as nx

    genes_l = [g for g in genes if isinstance(g, str) and g]
    sub_nodes = [g for g in genes_l if g in G]
    deg = dict(G.degree(sub_nodes))
    wdeg = dict(G.degree(sub_nodes, weight="weight"))
    rows = []
    for g in genes_l:
        rows.append(
            {
                "Gene": g,
                "in_graph": int(g in G),
                "degree": deg.get(g, 0),
                "weighted_degree": float(wdeg.get(g, 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if compute_centrality and sub_nodes:
        closeness = {}
        for cc in nx.connected_components(G.subgraph(sub_nodes)):
            cG = G.subgraph(cc)
            cc_clo = nx.closeness_centrality(cG)
            closeness.update(cc_clo)
        df["closeness"] = df["Gene"].map(closeness).fillna(0.0)
    if reference_genes:
        ref = [r for r in reference_genes if r in G]
        if ref:
            dist = {}
            for r in ref:
                sp = nx.single_source_shortest_path_length(G, r)
                for k, v in sp.items():
                    if k not in dist or v < dist[k]:
                        dist[k] = v
            df["shortest_dist_to_reference"] = df["Gene"].map(dist).fillna(pd.NA)
    summary = {
        "n_input_genes": len(genes_l),
        "n_in_graph": int(df["in_graph"].sum()),
        "mean_degree": float(df.loc[df["in_graph"] == 1, "degree"].mean() or 0.0),
        "mean_weighted_degree": float(df.loc[df["in_graph"] == 1, "weighted_degree"].mean() or 0.0),
    }
    return df, summary


def load_druggable_symbols(path: str) -> Set[str]:
    """Load a set of druggable gene symbols from provided labels TSV/CSV.

    Supports the NIHMS80906-small_mol-and-bio-druggable.tsv file in this repo.
    """
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    # Prefer 'symbol' column; fall back to 'hgnc_names'
    col = None
    for c in ["symbol", "hgnc_names", "Gene", "gene", "hgnc"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return set()
    return set(df[col].astype(str).str.replace(r";.*$", "", regex=True).str.strip())


"""Build electrolyte descriptor tables and graph datasets.

The electrolyte CSV keeps experimental/computed electrolyte properties as
primitive tabular features. RDKit descriptors are added as molecular-structure
features. For per-task prediction, the current target is masked from the
primitive feature vector to avoid training on the answer directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .compute_properties import (
    canonicalize_smiles,
    compute_morgan_fingerprint,
    compute_rdkit_descriptors,
)
from .config import (
    DATA_DIR,
    ELECTROLYTE_GENERATION_CONFIG,
    ELECTROLYTE_TARGETS,
    EXPANDED_RDKIT_DESCRIPTOR_NAMES,
    MORGAN_FINGERPRINT_BITS,
    PROCESSED_DIR,
    RDKIT_DESCRIPTOR_NAMES,
)
from .mol_to_graph import attach_graph_features, feature_metadata, mol_to_graph
from .motif_matching import learn_motif_weights_from_table


DEFAULT_ELECTROLYTE_CSV = DATA_DIR / "RE&WSE.csv"
DEFAULT_DESCRIPTOR_CSV = PROCESSED_DIR / "electrolytes_with_rdkit_descriptors.csv"
DEFAULT_DATASET_PT = PROCESSED_DIR / "electrolyte_graph_dataset.pt"
DEFAULT_MOTIF_WEIGHTS_CSV = PROCESSED_DIR / "electrolyte_motif_weights.csv"


DERIVED_ELECTROLYTE_FEATURES = {
    "derived_orbital_gap": {"LUMO_sol (eV)", "HOMO_sol (eV)"},
    "derived_orbital_midpoint": {"LUMO_sol (eV)", "HOMO_sol (eV)"},
    "derived_abs_homo": {"HOMO_sol (eV)"},
    "derived_log_dielectric": {"Dielectric constant of solvents"},
    "derived_lumo_times_log_dielectric": {
        "LUMO_sol (eV)",
        "Dielectric constant of solvents",
    },
    "derived_gap_times_log_dielectric": {
        "LUMO_sol (eV)",
        "HOMO_sol (eV)",
        "Dielectric constant of solvents",
    },
}


def add_derived_electrolyte_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    lumo = table["LUMO_sol (eV)"]
    homo = table["HOMO_sol (eV)"]
    dielectric = table["Dielectric constant of solvents"]
    log_dielectric = np.log1p(dielectric.clip(lower=0))
    gap = lumo - homo
    table["derived_orbital_gap"] = gap
    table["derived_orbital_midpoint"] = 0.5 * (lumo + homo)
    table["derived_abs_homo"] = homo.abs()
    table["derived_log_dielectric"] = log_dielectric
    table["derived_lumo_times_log_dielectric"] = lumo * log_dielectric
    table["derived_gap_times_log_dielectric"] = gap * log_dielectric
    return table


def load_cluster_annotations(data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    cluster_paths = sorted(data_dir.glob("cluster_*.csv"))
    if not cluster_paths:
        return None

    frames: list[pd.DataFrame] = []
    for path in cluster_paths:
        frame = pd.read_csv(path)
        required = {"EP ID", "cluster"}
        if not required.issubset(frame.columns):
            continue
        keep_cols = [
            col
            for col in ["EP ID", "binary_features", "cluster", "PC1", "PC2", "PC3"]
            if col in frame.columns
        ]
        frames.append(frame[keep_cols])
    if not frames:
        return None

    annotations = pd.concat(frames, ignore_index=True).drop_duplicates("EP ID")
    if "binary_features" in annotations.columns:
        parsed = annotations["binary_features"].apply(parse_binary_features)
        max_len = max((len(values) for values in parsed), default=0)
        for idx in range(max_len):
            annotations[f"cluster_binary_{idx}"] = parsed.apply(
                lambda values: float(values[idx]) if idx < len(values) else 0.0
            )
        annotations = annotations.drop(columns=["binary_features"])
    return annotations


def parse_binary_features(value: object) -> list[float]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [float(item) for item in parsed]


def deduplicate_by_canonical_smiles(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated molecular graphs before descriptor generation."""

    if "canonical_smiles" not in table.columns:
        raise ValueError("canonical_smiles column is required for deduplication")

    numeric_cols = [
        col
        for col in table.columns
        if col != "canonical_smiles" and pd.api.types.is_numeric_dtype(table[col])
    ]
    non_numeric_cols = [
        col
        for col in table.columns
        if col not in numeric_cols and col != "canonical_smiles"
    ]
    aggregations = {col: "median" for col in numeric_cols}
    aggregations.update({col: "first" for col in non_numeric_cols})
    deduplicated = (
        table.groupby("canonical_smiles", as_index=False)
        .agg(aggregations)
        .reset_index(drop=True)
    )
    counts = table.groupby("canonical_smiles").size().rename("source_row_count")
    return deduplicated.merge(counts, on="canonical_smiles", how="left")


def write_electrolyte_motif_weights(
    table: pd.DataFrame,
    output_csv: Path = DEFAULT_MOTIF_WEIGHTS_CSV,
) -> pd.DataFrame:
    learned = learn_motif_weights_from_table(
        table,
        ELECTROLYTE_GENERATION_CONFIG.target_motifs,
        ELECTROLYTE_GENERATION_CONFIG.targets,
    )
    rows = [motif.__dict__ for motif in learned.values()]
    weights = pd.DataFrame(
        rows,
        columns=[
            "name",
            "weight",
            "support_good",
            "support_bad",
            "enrichment",
            "odds_ratio",
        ],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output_csv, index=False)
    return weights


def build_electrolyte_descriptor_table(
    input_csv: Path = DEFAULT_ELECTROLYTE_CSV,
    output_csv: Path = DEFAULT_DESCRIPTOR_CSV,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "SMILES" not in df.columns:
        raise ValueError(f"{input_csv} must contain a 'SMILES' column")

    df = df.copy()
    df["canonical_smiles"] = df["SMILES"].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    df = deduplicate_by_canonical_smiles(df)

    records: list[dict] = []
    for _, row in df.iterrows():
        descriptors = compute_rdkit_descriptors(row["canonical_smiles"])
        fingerprint = compute_morgan_fingerprint(row["canonical_smiles"])
        if descriptors is None or fingerprint is None:
            continue
        record = row.to_dict()
        record.update({f"rdkit_{name}": value for name, value in descriptors.items()})
        record.update(fingerprint)
        records.append(record)

    table = pd.DataFrame(records)
    cluster_annotations = load_cluster_annotations(input_csv.parent)
    if cluster_annotations is not None:
        table = table.merge(cluster_annotations, on="EP ID", how="left")
        if "cluster" in table.columns:
            for cluster_id in sorted(table["cluster"].dropna().astype(int).unique()):
                table[f"cluster_onehot_{cluster_id}"] = (
                    table["cluster"].fillna(-1).astype(int) == cluster_id
                ).astype(float)
    table = add_derived_electrolyte_features(table)
    write_electrolyte_motif_weights(table)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return table


def electrolyte_feature_columns(
    table: pd.DataFrame,
    include_primitive_features: bool = True,
    include_derived_features: bool = True,
    rdkit_descriptor_set: str = "base",
) -> list[str]:
    primitive = []
    if include_primitive_features:
        primitive = [
            col
            for col in ELECTROLYTE_TARGETS
            if col in table.columns and pd.api.types.is_numeric_dtype(table[col])
        ]
    cluster_features = []
    if "cluster" in table.columns:
        cluster_features.append("cluster")
    cluster_features.extend(
        col
        for col in table.columns
        if col.startswith("cluster_onehot_") and pd.api.types.is_numeric_dtype(table[col])
    )
    cluster_features.extend(
        col
        for col in ["PC1", "PC2", "PC3"]
        if col in table.columns and pd.api.types.is_numeric_dtype(table[col])
    )
    cluster_features.extend(
        col
        for col in table.columns
        if col.startswith("cluster_binary_") and pd.api.types.is_numeric_dtype(table[col])
    )
    if rdkit_descriptor_set == "base":
        rdkit_names = RDKIT_DESCRIPTOR_NAMES
    elif rdkit_descriptor_set == "expanded":
        rdkit_names = EXPANDED_RDKIT_DESCRIPTOR_NAMES
    else:
        raise ValueError(f"Unknown RDKit descriptor set: {rdkit_descriptor_set}")
    rdkit = [f"rdkit_{name}" for name in rdkit_names]
    derived = []
    if include_derived_features:
        derived = [col for col in DERIVED_ELECTROLYTE_FEATURES if col in table.columns]
    morgan = [f"morgan_{idx}" for idx in range(MORGAN_FINGERPRINT_BITS)]
    return primitive + derived + cluster_features + rdkit + morgan


def _fit_feature_matrix(table: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, dict]:
    raw = table[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(raw)
    scaled = scaler.fit_transform(imputed)
    stats = {
        "feature_columns": feature_cols,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return scaled.astype(np.float32), stats


def build_electrolyte_graph_dataset(
    descriptor_csv: Path = DEFAULT_DESCRIPTOR_CSV,
    output_pt: Path = DEFAULT_DATASET_PT,
    prediction_target: str | None = None,
    mask_prediction_target: bool = True,
    include_primitive_features: bool = True,
    include_derived_features: bool = True,
    rdkit_descriptor_set: str = "base",
) -> dict:
    if not descriptor_csv.exists():
        build_electrolyte_descriptor_table(DEFAULT_ELECTROLYTE_CSV, descriptor_csv)

    table = pd.read_csv(descriptor_csv)
    available_targets = [target for target in ELECTROLYTE_TARGETS if target in table.columns]
    target_cols = [prediction_target] if prediction_target else available_targets
    stratify_target_cols = [
        target
        for target in ELECTROLYTE_TARGETS
        if target in table.columns and pd.api.types.is_numeric_dtype(table[target])
    ]
    missing_targets = [target for target in target_cols if target not in table.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    split_required_targets = list(dict.fromkeys(target_cols + stratify_target_cols))
    table = table.dropna(subset=split_required_targets).reset_index(drop=True)
    feature_cols = electrolyte_feature_columns(
        table,
        include_primitive_features,
        include_derived_features,
        rdkit_descriptor_set,
    )
    missing_feature_cols = [col for col in feature_cols if col not in table.columns]
    if missing_feature_cols:
        build_electrolyte_descriptor_table(DEFAULT_ELECTROLYTE_CSV, descriptor_csv)
        table = pd.read_csv(descriptor_csv)
        table = table.dropna(subset=split_required_targets).reset_index(drop=True)
        feature_cols = electrolyte_feature_columns(
            table,
            include_primitive_features,
            include_derived_features,
            rdkit_descriptor_set,
        )
    if mask_prediction_target and prediction_target in feature_cols:
        feature_cols = [col for col in feature_cols if col != prediction_target]
    elif mask_prediction_target and prediction_target is None:
        feature_cols = [col for col in feature_cols if col not in available_targets]
    if mask_prediction_target and prediction_target is not None:
        feature_cols = [
            col
            for col in feature_cols
            if prediction_target not in DERIVED_ELECTROLYTE_FEATURES.get(col, set())
        ]

    graph_features, feature_stats = _fit_feature_matrix(table, feature_cols)

    graphs: list[dict] = []
    for idx, row in table.iterrows():
        y = [float(row[target]) for target in target_cols]
        graph = mol_to_graph(str(row["canonical_smiles"]), y)
        graph = attach_graph_features(graph, graph_features[idx])
        graph["id"] = str(row.get("EP ID", idx))
        graph["stratify_targets"] = [
            float(row[target]) if pd.notna(row[target]) else float("nan")
            for target in stratify_target_cols
        ]
        if "cluster" in row and pd.notna(row["cluster"]):
            graph["cluster"] = int(row["cluster"])
        graphs.append(graph)

    metadata = feature_metadata()
    metadata.update(
        {
            "task_type": "electrolyte",
            "graph_feature_dim": len(feature_cols),
            "target_columns": target_cols,
            "stratify_target_columns": stratify_target_cols,
            "mask_prediction_target": mask_prediction_target,
            "include_primitive_features": include_primitive_features,
            "include_derived_features": include_derived_features,
            "rdkit_descriptor_set": rdkit_descriptor_set,
            **feature_stats,
        }
    )
    payload = {
        "graphs": graphs,
        "property_names": target_cols,
        "metadata": metadata,
        "descriptor_csv": str(descriptor_csv),
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ELECTROLYTE_CSV)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTOR_CSV)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PT)
    parser.add_argument("--target", choices=ELECTROLYTE_TARGETS, default=None)
    parser.add_argument(
        "--allow-target-as-feature",
        action="store_true",
        help="Keep the current target in graph-level features. This is useful only for ablations.",
    )
    parser.add_argument(
        "--rdkit-only-features",
        action="store_true",
        help="Use only RDKit descriptors and Morgan fingerprints as graph-level features.",
    )
    parser.add_argument(
        "--no-derived-features",
        action="store_true",
        help="Do not include derived electrolyte interaction features.",
    )
    parser.add_argument(
        "--expanded-rdkit-descriptors",
        action="store_true",
        help="Use the expanded RDKit descriptor set with EState and VSA descriptors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_electrolyte_descriptor_table(args.input, args.descriptors)
    payload = build_electrolyte_graph_dataset(
        descriptor_csv=args.descriptors,
        output_pt=args.dataset,
        prediction_target=args.target,
        mask_prediction_target=not args.allow_target_as_feature,
        include_primitive_features=not args.rdkit_only_features,
        include_derived_features=not args.no_derived_features,
        rdkit_descriptor_set="expanded" if args.expanded_rdkit_descriptors else "base",
    )
    print(f"Wrote {len(table)} descriptor rows to {args.descriptors}")
    print(f"Wrote {len(payload['graphs'])} electrolyte graphs to {args.dataset}")
    print(f"Graph-level features: {payload['metadata']['graph_feature_dim']}")


if __name__ == "__main__":
    main()

"""Train electrolyte property GNNs.

Each prediction task gets its own model by default. The current target column is
removed from the auxiliary graph-level feature vector, while the remaining
Electrolytes.csv properties and RDKit descriptors are retained as requested.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .config import ELECTROLYTE_TARGETS, MODELS_DIR, RESULTS_DIR
from .electrolyte_data import (
    DEFAULT_DATASET_PT,
    DEFAULT_DESCRIPTOR_CSV,
    DEFAULT_ELECTROLYTE_CSV,
    build_electrolyte_descriptor_table,
    build_electrolyte_graph_dataset,
)
from .train_gnn import train


GENERATION_MODEL_SUBDIR = "electrolyte_generation_ready"


TARGET_MODEL_SETTINGS = {
    "Es-Ea (eV)": {
        "hidden_dim": 192,
        "num_layers": 3,
        "dropout": 0.10,
        "lr": 6.0e-4,
        "target_transform": "none",
        "loss_name": "smoothl1",
        "use_graph_feature_encoder": True,
        "message_passing": "edge_gated",
        "monitor_metric": "r2",
    },
    "LUMO_sol (eV)": {
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.06,
        "lr": 7.0e-4,
        "target_transform": "none",
        "use_graph_feature_encoder": True,
        "loss_name": "smoothl1",
        "message_passing": "edge_gated",
    },
    "HOMO_sol (eV)": {
        "hidden_dim": 224,
        "num_layers": 4,
        "dropout": 0.08,
        "lr": 4.0e-4,
        "batch_size": 16,
        "target_transform": "none",
        "loss_name": "smoothl1",
        "message_passing": "edge_gated",
        "rdkit_descriptor_set": "expanded",
    },
    "Dielectric constant of solvents": {
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.06,
        "lr": 7.0e-4,
        "target_transform": "log1p",
        "use_graph_feature_encoder": True,
        "loss_name": "smoothl1",
        "include_derived_features": False,
        "sample_weight_mode": "none",
        "message_passing": "gcn",
        "monitor_metric": "r2",
    },
}

for _settings in TARGET_MODEL_SETTINGS.values():
    _settings.setdefault("use_graph_feature_encoder", False)
    _settings.setdefault("loss_name", "smoothl1")
    _settings.setdefault("include_derived_features", True)
    _settings.setdefault("sample_weight_mode", "none")
    _settings.setdefault("message_passing", "gcn")
    _settings.setdefault("monitor_metric", None)
    _settings.setdefault("batch_size", None)
    _settings.setdefault("rdkit_descriptor_set", "base")


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return slug or "target"


def train_per_task_models(
    epochs: int,
    batch_size: int,
    seed: int,
    allow_target_as_feature: bool = False,
    patience: int = 35,
    include_primitive_features: bool = True,
    model_subdir: str = "electrolyte",
    results_subdir: str = "electrolyte",
    monitor_metric: str = "loss",
    split_strategy: str = "multitarget_stratified",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    message_passing: str = "gcn",
) -> pd.DataFrame:
    build_electrolyte_descriptor_table(DEFAULT_ELECTROLYTE_CSV, DEFAULT_DESCRIPTOR_CSV)

    rows: list[dict] = []
    for target in ELECTROLYTE_TARGETS:
        settings = TARGET_MODEL_SETTINGS[target]
        slug = slugify(target)
        dataset_path = DEFAULT_DATASET_PT.with_name(f"electrolyte_graph_dataset_{slug}.pt")
        model_path = MODELS_DIR / model_subdir / f"{slug}_gnn.pt"
        metrics_path = RESULTS_DIR / results_subdir / f"{slug}_metrics.csv"
        build_electrolyte_graph_dataset(
            descriptor_csv=DEFAULT_DESCRIPTOR_CSV,
            output_pt=dataset_path,
            prediction_target=target,
            mask_prediction_target=not allow_target_as_feature,
            include_primitive_features=include_primitive_features,
            include_derived_features=settings["include_derived_features"],
            rdkit_descriptor_set=settings["rdkit_descriptor_set"],
        )
        summary = train(
            dataset_path=dataset_path,
            model_path=model_path,
            metrics_path=metrics_path,
            epochs=epochs,
            batch_size=settings["batch_size"] or batch_size,
            hidden_dim=settings["hidden_dim"],
            num_layers=settings["num_layers"],
            dropout=settings["dropout"],
            lr=settings["lr"],
            seed=seed,
            patience=patience,
            target_transform=settings["target_transform"],
            use_graph_feature_encoder=settings["use_graph_feature_encoder"],
            loss_name=settings["loss_name"],
            split_seed=seed,
            sample_weight_mode=settings["sample_weight_mode"],
            monitor_metric=settings["monitor_metric"] or monitor_metric,
            split_strategy=split_strategy,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            message_passing=settings.get("message_passing", message_passing),
            readout=settings.get("readout", "mean_max"),
            jk_mode=settings.get("jk_mode", "last"),
        )
        rows.append(
            {
                "target": target,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "hidden_dim": settings["hidden_dim"],
                "num_layers": settings["num_layers"],
                "dropout": settings["dropout"],
                "lr": settings["lr"],
                "batch_size": settings["batch_size"] or batch_size,
                "target_transform": settings["target_transform"],
                "use_graph_feature_encoder": settings["use_graph_feature_encoder"],
                "loss_name": settings["loss_name"],
                "message_passing": settings["message_passing"],
                "readout": settings.get("readout", "mean_max"),
                "jk_mode": settings.get("jk_mode", "last"),
                "rdkit_descriptor_set": settings["rdkit_descriptor_set"],
                "sample_weight_mode": settings["sample_weight_mode"],
                "MAE": summary[f"{target}_MAE"],
                "RMSE": summary[f"{target}_RMSE"],
                "R2": summary[f"{target}_R2"],
                "test_MAE": summary[f"{target}_test_MAE"],
                "test_RMSE": summary[f"{target}_test_RMSE"],
                "test_R2": summary[f"{target}_test_R2"],
                "target_masked_from_features": not allow_target_as_feature,
                "include_primitive_features": include_primitive_features,
                "include_derived_features": settings["include_derived_features"],
            }
        )

    summary_table = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / results_subdir / "per_task_model_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(summary_path, index=False)
    print(f"Wrote electrolyte model summary to {summary_path}")
    return summary_table


def train_single_task_model(
    target: str,
    epochs: int,
    batch_size: int,
    seed: int,
    allow_target_as_feature: bool = False,
    patience: int = 45,
    include_primitive_features: bool = False,
    model_subdir: str = "electrolyte_generation",
    results_subdir: str = "electrolyte_generation",
    split_strategy: str = "multitarget_stratified",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> pd.DataFrame:
    """Retrain one generation-compatible electrolyte property model."""

    if target not in TARGET_MODEL_SETTINGS:
        raise KeyError(f"Unknown electrolyte target: {target}")
    build_electrolyte_descriptor_table(DEFAULT_ELECTROLYTE_CSV, DEFAULT_DESCRIPTOR_CSV)

    settings = TARGET_MODEL_SETTINGS[target]
    slug = slugify(target)
    dataset_path = DEFAULT_DATASET_PT.with_name(f"electrolyte_graph_dataset_{slug}.pt")
    model_path = MODELS_DIR / model_subdir / f"{slug}_gnn.pt"
    metrics_path = RESULTS_DIR / results_subdir / f"{slug}_metrics.csv"
    build_electrolyte_graph_dataset(
        descriptor_csv=DEFAULT_DESCRIPTOR_CSV,
        output_pt=dataset_path,
        prediction_target=target,
        mask_prediction_target=not allow_target_as_feature,
        include_primitive_features=include_primitive_features,
        include_derived_features=settings["include_derived_features"],
        rdkit_descriptor_set=settings["rdkit_descriptor_set"],
    )
    summary = train(
        dataset_path=dataset_path,
        model_path=model_path,
        metrics_path=metrics_path,
        epochs=epochs,
        batch_size=settings["batch_size"] or batch_size,
        hidden_dim=settings["hidden_dim"],
        num_layers=settings["num_layers"],
        dropout=settings["dropout"],
        lr=settings["lr"],
        seed=seed,
        patience=patience,
        target_transform=settings["target_transform"],
        use_graph_feature_encoder=settings["use_graph_feature_encoder"],
        loss_name=settings["loss_name"],
        split_seed=seed,
        sample_weight_mode=settings["sample_weight_mode"],
        monitor_metric=settings["monitor_metric"] or "r2",
        split_strategy=split_strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        message_passing=settings["message_passing"],
        readout=settings.get("readout", "mean_max"),
        jk_mode=settings.get("jk_mode", "last"),
    )
    row = {
        "target": target,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "hidden_dim": settings["hidden_dim"],
        "num_layers": settings["num_layers"],
        "dropout": settings["dropout"],
        "lr": settings["lr"],
        "batch_size": settings["batch_size"] or batch_size,
        "target_transform": settings["target_transform"],
        "loss_name": settings["loss_name"],
        "message_passing": settings["message_passing"],
        "monitor_metric": settings["monitor_metric"] or "r2",
        "rdkit_descriptor_set": settings["rdkit_descriptor_set"],
        "MAE": summary[f"{target}_MAE"],
        "RMSE": summary[f"{target}_RMSE"],
        "R2": summary[f"{target}_R2"],
        "test_MAE": summary[f"{target}_test_MAE"],
        "test_RMSE": summary[f"{target}_test_RMSE"],
        "test_R2": summary[f"{target}_test_R2"],
        "target_masked_from_features": not allow_target_as_feature,
        "include_primitive_features": include_primitive_features,
        "include_derived_features": settings["include_derived_features"],
    }
    summary_table = pd.DataFrame([row])
    summary_path = RESULTS_DIR / results_subdir / f"{slug}_single_task_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(summary_path, index=False)
    print(f"Wrote single-task electrolyte model summary to {summary_path}")
    return summary_table


def train_multitask_model(
    epochs: int,
    batch_size: int,
    seed: int,
    allow_target_as_feature: bool = False,
    patience: int = 35,
    split_strategy: str = "multitarget_stratified",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    message_passing: str = "gcn",
) -> pd.DataFrame:
    build_electrolyte_descriptor_table(DEFAULT_ELECTROLYTE_CSV, DEFAULT_DESCRIPTOR_CSV)
    dataset_path = DEFAULT_DATASET_PT
    build_electrolyte_graph_dataset(
        descriptor_csv=DEFAULT_DESCRIPTOR_CSV,
        output_pt=dataset_path,
        prediction_target=None,
        mask_prediction_target=not allow_target_as_feature,
        include_primitive_features=False,
        include_derived_features=False,
        rdkit_descriptor_set="base",
    )
    train(
        dataset_path=dataset_path,
        model_path=MODELS_DIR / "electrolyte" / "multitask_gnn.pt",
        metrics_path=RESULTS_DIR / "electrolyte" / "multitask_metrics.csv",
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=192,
        num_layers=4,
        dropout=0.15,
        lr=8.0e-4,
        seed=seed,
        patience=patience,
        target_transform="none",
        use_graph_feature_encoder=True,
        loss_name="smoothl1",
        split_seed=seed,
        sample_weight_mode="none",
        monitor_metric="loss",
        split_strategy=split_strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        message_passing=message_passing,
        readout="mean_max",
        jk_mode="last",
    )
    return pd.read_csv(RESULTS_DIR / "electrolyte" / "multitask_metrics.csv")


def write_recommended_model_summary(
    per_task_summary: pd.DataFrame | None = None,
    multitask_metrics: pd.DataFrame | None = None,
    results_subdir: str = "electrolyte",
) -> pd.DataFrame:
    """Select the strongest available GNN checkpoint for each electrolyte target."""
    results_dir = RESULTS_DIR / results_subdir
    if per_task_summary is None:
        per_task_summary = pd.read_csv(results_dir / "per_task_model_summary.csv")
    if multitask_metrics is None:
        multitask_metrics = pd.read_csv(results_dir / "multitask_metrics.csv")

    multitask_by_target = {
        row["property"]: row for _, row in multitask_metrics.iterrows()
    }
    rows: list[dict] = []
    for _, per_task_row in per_task_summary.iterrows():
        target = per_task_row["target"]
        candidates = [
            {
                "target": target,
                "selected_model": "per-task",
                "model_path": per_task_row["model_path"],
                "metrics_path": per_task_row["metrics_path"],
                "MAE": per_task_row["MAE"],
                "RMSE": per_task_row["RMSE"],
                "R2": per_task_row["R2"],
                "test_MAE": per_task_row["test_MAE"],
                "test_RMSE": per_task_row["test_RMSE"],
                "test_R2": per_task_row["test_R2"],
            }
        ]
        multitask_row = multitask_by_target.get(target)
        if multitask_row is not None:
            candidates.append(
                {
                    "target": target,
                    "selected_model": "multitask",
                    "model_path": str(MODELS_DIR / "electrolyte" / "multitask_gnn.pt"),
                    "metrics_path": str(results_dir / "multitask_metrics.csv"),
                    "MAE": multitask_row["MAE"],
                    "RMSE": multitask_row["RMSE"],
                    "R2": multitask_row["R2"],
                    "test_MAE": multitask_row["test_MAE"],
                    "test_RMSE": multitask_row["test_RMSE"],
                    "test_R2": multitask_row["test_R2"],
                }
            )
        rows.append(max(candidates, key=lambda row: row["R2"]))

    summary = pd.DataFrame(rows)
    output_path = results_dir / "recommended_model_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Wrote recommended electrolyte model summary to {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "cluster_stratified", "multitarget_stratified"],
        default="multitarget_stratified",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--message-passing", choices=["gcn", "edge_gated", "gine"], default="gcn")
    parser.add_argument(
        "--mode",
        choices=["per-task", "multitask", "both", "dielectric"],
        default="per-task",
        help="Per-task models are the recommended leakage-safe setting.",
    )
    parser.add_argument(
        "--allow-target-as-feature",
        action="store_true",
        help="Ablation only: keeps target values as input features.",
    )
    parser.add_argument(
        "--rdkit-only-features",
        action="store_true",
        help="Train generation-compatible models using no primitive electrolyte properties.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_task_summary = None
    multitask_metrics = None
    results_subdir = "electrolyte_generation" if args.rdkit_only_features else "electrolyte"
    if args.mode == "dielectric":
        train_single_task_model(
            target="Dielectric constant of solvents",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            allow_target_as_feature=args.allow_target_as_feature,
            patience=args.patience,
            include_primitive_features=False,
            model_subdir=GENERATION_MODEL_SUBDIR,
            results_subdir="electrolyte_generation",
            split_strategy=args.split_strategy,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
        return
    if args.mode in {"per-task", "both"}:
        per_task_summary = train_per_task_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            allow_target_as_feature=args.allow_target_as_feature,
            patience=args.patience,
            include_primitive_features=not args.rdkit_only_features,
            model_subdir=GENERATION_MODEL_SUBDIR if args.rdkit_only_features else "electrolyte",
            results_subdir=results_subdir,
            split_strategy=args.split_strategy,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            message_passing=args.message_passing,
        )
    if args.mode in {"multitask", "both"}:
        multitask_metrics = train_multitask_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            allow_target_as_feature=args.allow_target_as_feature,
            patience=args.patience,
            split_strategy=args.split_strategy,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            message_passing=args.message_passing,
        )
    if args.mode == "both" and not args.rdkit_only_features:
        write_recommended_model_summary(per_task_summary, multitask_metrics, results_subdir)


if __name__ == "__main__":
    main()

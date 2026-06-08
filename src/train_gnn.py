"""Train the molecular graph property predictor."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODELS_DIR, PROCESSED_DIR, PROPERTY_NAMES, RESULTS_DIR
from .gnn_model import MolecularGNN, collate_graphs


def split_dataset(graphs: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    graphs = list(graphs)
    rng.shuffle(graphs)
    val_size = max(1, int(len(graphs) * val_fraction))
    return graphs[val_size:], graphs[:val_size]


def _split_sizes(n: int, val_fraction: float, test_fraction: float) -> dict[str, int]:
    test_size = max(1, int(n * test_fraction))
    val_size = max(1, int(n * val_fraction))
    if val_size + test_size >= n:
        overflow = val_size + test_size - n + 1
        while overflow > 0 and val_size >= test_size and val_size > 1:
            val_size -= 1
            overflow -= 1
        while overflow > 0 and test_size > 1:
            test_size -= 1
            overflow -= 1
    train_size = n - val_size - test_size
    return {"train": train_size, "val": val_size, "test": test_size}


def _graph_id(graph: dict, fallback: int) -> str:
    return str(graph.get("id", fallback))


def _stratify_matrix(graphs: list[dict]) -> np.ndarray:
    values: list[list[float]] = []
    for graph in graphs:
        raw = graph.get("stratify_targets")
        if raw is None:
            raw = graph["y"]
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().flatten().tolist()
        values.append([float(value) for value in raw])
    return np.asarray(values, dtype=float)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    bins = np.full(values.shape[0], -1, dtype=int)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return bins
    order = finite[np.argsort(values[finite], kind="mergesort")]
    usable_bins = max(1, min(n_bins, len(order)))
    for rank, idx in enumerate(order):
        bins[idx] = min(usable_bins - 1, int(rank * usable_bins / len(order)))
    return bins


def _multitarget_stratified_split(
    graphs: list[dict],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    indexed_graphs = list(enumerate(graphs))
    n = len(indexed_graphs)
    sizes = _split_sizes(n, val_fraction, test_fraction)
    fractions = {name: size / n for name, size in sizes.items()}

    target_values = _stratify_matrix(graphs)
    n_bins = max(2, min(10, n // 20))
    labels_by_index: dict[int, list[str]] = {idx: [] for idx, _ in indexed_graphs}
    for target_idx in range(target_values.shape[1]):
        bins = _quantile_bins(target_values[:, target_idx], n_bins)
        for graph_idx, bin_idx in enumerate(bins):
            if bin_idx >= 0:
                labels_by_index[graph_idx].append(f"target_{target_idx}_bin_{bin_idx}")

    for graph_idx, graph in indexed_graphs:
        if "cluster" in graph:
            labels_by_index[graph_idx].append(f"cluster_{graph['cluster']}")
        if not labels_by_index[graph_idx]:
            labels_by_index[graph_idx].append("all")

    label_totals = Counter(
        label
        for labels in labels_by_index.values()
        for label in labels
    )
    desired = {
        split_name: {
            label: total * fractions[split_name]
            for label, total in label_totals.items()
        }
        for split_name in sizes
    }
    current = {split_name: Counter() for split_name in sizes}
    split_indices = {split_name: [] for split_name in sizes}

    order = sorted(
        range(n),
        key=lambda idx: (
            min(label_totals[label] for label in labels_by_index[idx]),
            rng.random(),
            _graph_id(graphs[idx], idx),
        ),
    )

    for graph_idx in order:
        labels = labels_by_index[graph_idx]
        best_split: str | None = None
        best_score = -float("inf")
        for split_name, split_size in sizes.items():
            if len(split_indices[split_name]) >= split_size:
                continue
            label_score = 0.0
            for label in labels:
                wanted = desired[split_name][label]
                have = current[split_name][label]
                if wanted <= 0:
                    continue
                label_score += (wanted - have) / max(1.0, wanted)
            size_score = (split_size - len(split_indices[split_name])) / max(1, split_size)
            score = label_score + 0.05 * size_score + rng.random() * 1.0e-9
            if score > best_score:
                best_score = score
                best_split = split_name
        if best_split is None:
            raise RuntimeError("Unable to assign graph to a split")
        split_indices[best_split].append(graph_idx)
        current[best_split].update(labels)

    train_graphs = [graphs[idx] for idx in split_indices["train"]]
    val_graphs = [graphs[idx] for idx in split_indices["val"]]
    test_graphs = [graphs[idx] for idx in split_indices["test"]]
    rng.shuffle(train_graphs)
    rng.shuffle(val_graphs)
    rng.shuffle(test_graphs)
    return train_graphs, val_graphs, test_graphs


def split_dataset_three_way(
    graphs: list[dict],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    strategy: str = "random",
) -> tuple[list[dict], list[dict], list[dict]]:
    if strategy == "random":
        rng = random.Random(seed)
        shuffled = list(graphs)
        rng.shuffle(shuffled)
        sizes = _split_sizes(len(shuffled), val_fraction, test_fraction)
        test_graphs = shuffled[: sizes["test"]]
        val_graphs = shuffled[sizes["test"] : sizes["test"] + sizes["val"]]
        train_graphs = shuffled[sizes["test"] + sizes["val"] :]
        return train_graphs, val_graphs, test_graphs
    if strategy == "multitarget_stratified":
        return _multitarget_stratified_split(graphs, val_fraction, test_fraction, seed)
    if strategy != "cluster_stratified":
        raise ValueError(f"Unknown split strategy: {strategy}")

    rng = random.Random(seed)
    groups: dict[object, list[dict]] = {}
    for graph in graphs:
        groups.setdefault(graph.get("cluster", "unknown"), []).append(graph)

    train_graphs: list[dict] = []
    val_graphs: list[dict] = []
    test_graphs: list[dict] = []
    for cluster, cluster_graphs in groups.items():
        ordered = sorted(cluster_graphs, key=lambda graph: float(graph["y"][0]))
        buckets: list[list[dict]] = [[] for _ in range(5)]
        for idx, graph in enumerate(ordered):
            buckets[idx % len(buckets)].append(graph)
        for bucket in buckets:
            rng.shuffle(bucket)
            n = len(bucket)
            if n == 0:
                continue
            test_size = max(1, round(n * test_fraction)) if n >= 6 else int(n >= 3)
            val_size = max(1, round(n * val_fraction)) if n >= 6 else int(n >= 4)
            test_graphs.extend(bucket[:test_size])
            val_graphs.extend(bucket[test_size : test_size + val_size])
            train_graphs.extend(bucket[test_size + val_size :])

    rng.shuffle(train_graphs)
    rng.shuffle(val_graphs)
    rng.shuffle(test_graphs)
    return train_graphs, val_graphs, test_graphs


def batch_iter(graphs: list[dict], batch_size: int, shuffle: bool = True):
    indices = list(range(len(graphs)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [graphs[i] for i in indices[start : start + batch_size]]


def transform_targets(y: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "none":
        return y
    if transform == "log1p":
        if torch.any(y <= -1):
            raise ValueError("log1p target transform requires all targets > -1")
        return torch.log1p(y)
    if transform == "sqrt":
        if torch.any(y < 0):
            raise ValueError("sqrt target transform requires all targets >= 0")
        return torch.sqrt(y)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_transform_targets(y: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "none":
        return y
    if transform == "log1p":
        return torch.expm1(y)
    if transform == "sqrt":
        return torch.square(y).clamp_min(0.0)
    raise ValueError(f"Unknown target transform: {transform}")


def make_sample_weights(
    y_raw: torch.Tensor,
    mode: str,
    high_threshold: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "none":
        weights = torch.ones(y_raw.size(0), dtype=y_raw.dtype, device=y_raw.device)
    elif mode == "sqrt_target":
        weights = torch.sqrt(torch.clamp(y_raw.mean(dim=1), min=0.0) + 1.0)
    elif mode == "target":
        weights = torch.clamp(y_raw.mean(dim=1), min=0.0) + 1.0
    elif mode == "high_tail":
        if high_threshold is None:
            raise ValueError("high_tail sample weighting requires high_threshold")
        weights = torch.ones(y_raw.size(0), dtype=y_raw.dtype, device=y_raw.device)
        weights = weights + 4.0 * (y_raw.mean(dim=1) >= high_threshold.to(y_raw.device)).float()
    else:
        raise ValueError(f"Unknown sample weight mode: {mode}")
    return weights / weights.mean().clamp_min(1.0e-6)


def train(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    seed: int,
    patience: int = 25,
    target_transform: str = "none",
    use_graph_feature_encoder: bool = False,
    loss_name: str = "smoothl1",
    split_seed: int | None = None,
    sample_weight_mode: str = "none",
    monitor_metric: str = "loss",
    split_strategy: str = "random",
    val_fraction: float = 0.1,
    test_fraction: float = 0.15,
    message_passing: str = "gcn",
    readout: str = "mean_max",
    jk_mode: str = "last",
    weight_decay: float = 1.0e-4,
) -> dict[str, float]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    graphs = payload["graphs"]
    property_names = payload.get("property_names", PROPERTY_NAMES)
    if len(graphs) < 5:
        raise ValueError("Need at least 5 molecules to train a validation split")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_graphs, val_graphs, test_graphs = split_dataset_three_way(
        graphs,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed if split_seed is None else split_seed,
        strategy=split_strategy,
    )
    y_train_raw = torch.stack([graph["y"] for graph in train_graphs])
    y_train = transform_targets(y_train_raw, target_transform)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp_min(1.0e-6)
    high_threshold = torch.quantile(y_train_raw.mean(dim=1), 0.90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MolecularGNN(
        num_node_features=payload["metadata"]["num_node_features"],
        num_outputs=len(property_names),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        graph_feature_dim=payload["metadata"].get("graph_feature_dim", 0),
        use_graph_feature_encoder=use_graph_feature_encoder,
        num_edge_features=payload["metadata"].get("num_edge_features", 0),
        message_passing=message_passing,
        readout=readout,
        jk_mode=jk_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.6,
        patience=max(3, patience // 4),
    )
    if loss_name == "smoothl1":
        loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    elif loss_name == "mse":
        loss_fn = torch.nn.MSELoss(reduction="none")
    elif loss_name == "mae":
        loss_fn = torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    best_state = deepcopy(model.state_dict())
    best_monitor_value = -float("inf") if monitor_metric == "r2" else float("inf")
    stale_epochs = 0

    def validation_stats() -> tuple[float, float]:
        model.eval()
        losses: list[float] = []
        pred_parts: list[np.ndarray] = []
        true_parts: list[np.ndarray] = []
        with torch.no_grad():
            for graphs_batch in batch_iter(val_graphs, batch_size=batch_size, shuffle=False):
                batch = collate_graphs(graphs_batch)
                graph_features = batch.get("graph_features")
                graph_features = graph_features.to(device) if graph_features is not None else None
                edge_attr = batch.get("edge_attr")
                edge_attr = edge_attr.to(device) if edge_attr is not None else None
                pred = model(
                    batch["x"].to(device),
                    batch["edge_index"].to(device),
                    batch["batch"].to(device),
                    graph_features,
                    edge_attr,
                )
                y_raw = batch["y"]
                y = ((transform_targets(y_raw, target_transform) - y_mean) / y_std).to(device)
                raw_loss = loss_fn(pred, y).mean(dim=1)
                weights = make_sample_weights(y_raw.to(device), sample_weight_mode, high_threshold)
                losses.append(float((raw_loss * weights).mean().item()))
                pred_original = inverse_transform_targets(
                    pred.cpu() * y_std + y_mean,
                    target_transform,
                )
                pred_parts.append(pred_original.numpy())
                true_parts.append(y_raw.numpy())
        y_pred_val = np.vstack(pred_parts)
        y_true_val = np.vstack(true_parts)
        return float(np.mean(losses)), float(r2_score(y_true_val, y_pred_val))

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for graphs_batch in batch_iter(train_graphs, batch_size=batch_size, shuffle=True):
            batch = collate_graphs(graphs_batch)
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            graph_batch = batch["batch"].to(device)
            graph_features = batch.get("graph_features")
            graph_features = graph_features.to(device) if graph_features is not None else None
            edge_attr = batch.get("edge_attr")
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            y = ((transform_targets(batch["y"], target_transform) - y_mean) / y_std).to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, edge_index, graph_batch, graph_features, edge_attr)
            raw_loss = loss_fn(pred, y).mean(dim=1)
            weights = make_sample_weights(
                batch["y"].to(device),
                sample_weight_mode,
                high_threshold,
            )
            loss = (raw_loss * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            val_loss, val_r2 = validation_stats()
            print(
                f"epoch={epoch:03d} train_loss={np.mean(losses):.4f} "
                f"val_loss={val_loss:.4f} val_r2={val_r2:.4f}"
            )
        else:
            val_loss, val_r2 = validation_stats()

        scheduler.step(val_loss)
        monitor_value = val_r2 if monitor_metric == "r2" else val_loss
        improved = (
            monitor_value > best_monitor_value + 1.0e-5
            if monitor_metric == "r2"
            else monitor_value < best_monitor_value - 1.0e-5
        )
        if improved:
            best_monitor_value = monitor_value
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            print(
                f"early_stop epoch={epoch:03d} "
                f"best_{monitor_metric}={best_monitor_value:.4f}"
            )
            break

    model.load_state_dict(best_state)

    def evaluate(graphs_to_eval: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        pred_parts: list[np.ndarray] = []
        true_parts: list[np.ndarray] = []
        with torch.no_grad():
            for graphs_batch in batch_iter(graphs_to_eval, batch_size=batch_size, shuffle=False):
                batch = collate_graphs(graphs_batch)
                graph_features = batch.get("graph_features")
                graph_features = graph_features.to(device) if graph_features is not None else None
                edge_attr = batch.get("edge_attr")
                edge_attr = edge_attr.to(device) if edge_attr is not None else None
                pred_scaled = model(
                    batch["x"].to(device),
                    batch["edge_index"].to(device),
                    batch["batch"].to(device),
                    graph_features,
                    edge_attr,
                ).cpu()
                pred = inverse_transform_targets(pred_scaled * y_std + y_mean, target_transform)
                pred_parts.append(pred.numpy())
                true_parts.append(batch["y"].numpy())
        return np.vstack(true_parts), np.vstack(pred_parts)

    y_true, y_pred = evaluate(val_graphs)
    y_test_true, y_test_pred = evaluate(test_graphs)

    rows: list[dict[str, float | str]] = []
    summary: dict[str, float] = {}
    for idx, name in enumerate(property_names):
        mae = float(mean_absolute_error(y_true[:, idx], y_pred[:, idx]))
        rmse = float(mean_squared_error(y_true[:, idx], y_pred[:, idx]) ** 0.5)
        r2 = float(r2_score(y_true[:, idx], y_pred[:, idx]))
        test_mae = float(mean_absolute_error(y_test_true[:, idx], y_test_pred[:, idx]))
        test_rmse = float(mean_squared_error(y_test_true[:, idx], y_test_pred[:, idx]) ** 0.5)
        test_r2 = float(r2_score(y_test_true[:, idx], y_test_pred[:, idx]))
        rows.append(
            {
                "property": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "test_MAE": test_mae,
                "test_RMSE": test_rmse,
                "test_R2": test_r2,
            }
        )
        summary[f"{name}_MAE"] = mae
        summary[f"{name}_RMSE"] = rmse
        summary[f"{name}_R2"] = r2
        summary[f"{name}_test_MAE"] = test_mae
        summary[f"{name}_test_RMSE"] = test_rmse
        summary[f"{name}_test_R2"] = test_r2

    # Keep split diagnostics with the checkpoint for reproducibility.
    split_counts = {
        "train": len(train_graphs),
        "val": len(val_graphs),
        "test": len(test_graphs),
    }
    cluster_counts = {}
    for split_name, split_graphs in [
        ("train", train_graphs),
        ("val", val_graphs),
        ("test", test_graphs),
    ]:
        counts: dict[str, int] = {}
        for graph in split_graphs:
            key = str(graph.get("cluster", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        cluster_counts[split_name] = counts

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "property_names": property_names,
            "metadata": payload["metadata"],
            "y_mean": y_mean,
            "y_std": y_std,
            "model_kwargs": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "graph_feature_dim": payload["metadata"].get("graph_feature_dim", 0),
                "use_graph_feature_encoder": use_graph_feature_encoder,
                "num_edge_features": payload["metadata"].get("num_edge_features", 0),
                "message_passing": message_passing,
                "readout": readout,
                "jk_mode": jk_mode,
            },
            "target_transform": target_transform,
            "loss_name": loss_name,
            "split_seed": seed if split_seed is None else split_seed,
            "sample_weight_mode": sample_weight_mode,
            "high_weight_threshold": float(high_threshold),
            "monitor_metric": monitor_metric,
            "weight_decay": weight_decay,
            "split_strategy": split_strategy,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_counts": split_counts,
            "cluster_counts": cluster_counts,
            "split_ids": {
                "train": [_graph_id(graph, idx) for idx, graph in enumerate(train_graphs)],
                "val": [_graph_id(graph, idx) for idx, graph in enumerate(val_graphs)],
                "test": [_graph_id(graph, idx) for idx, graph in enumerate(test_graphs)],
            },
        },
        model_path,
    )
    print(f"Wrote model to {model_path}")
    print(f"Wrote metrics to {metrics_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROCESSED_DIR / "graph_dataset.pt")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "gnn_model.pt")
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "prediction_metrics.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--target-transform", choices=["none", "log1p", "sqrt"], default="none")
    parser.add_argument("--graph-feature-encoder", action="store_true")
    parser.add_argument("--loss", choices=["smoothl1", "mse", "mae"], default="smoothl1")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--sample-weight",
        choices=["none", "sqrt_target", "target", "high_tail"],
        default="none",
    )
    parser.add_argument("--monitor", choices=["loss", "r2"], default="loss")
    parser.add_argument(
        "--split-strategy",
        choices=["random", "cluster_stratified", "multitarget_stratified"],
        default="random",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--message-passing", choices=["gcn", "edge_gated", "gine"], default="gcn")
    parser.add_argument(
        "--readout",
        choices=["mean_max", "attention_mean_max"],
        default="mean_max",
    )
    parser.add_argument("--jk-mode", choices=["last", "concat"], default="last")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset_path=args.dataset,
        model_path=args.model,
        metrics_path=args.metrics,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        target_transform=args.target_transform,
        use_graph_feature_encoder=args.graph_feature_encoder,
        loss_name=args.loss,
        split_seed=args.split_seed,
        sample_weight_mode=args.sample_weight,
        monitor_metric=args.monitor,
        split_strategy=args.split_strategy,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        message_passing=args.message_passing,
        readout=args.readout,
        jk_mode=args.jk_mode,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()

"""Run the electrolyte-specific TargetMolGraph workflow."""

from __future__ import annotations

import argparse

from .electrolyte_data import (
    DEFAULT_DESCRIPTOR_CSV,
    DEFAULT_ELECTROLYTE_CSV,
    build_electrolyte_descriptor_table,
)
from .generate_electrolytes import generate
from .train_electrolyte_gnns import (
    train_multitask_model,
    train_per_task_models,
    write_recommended_model_summary,
)
from .visualize import make_electrolyte_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_electrolyte_descriptor_table(DEFAULT_ELECTROLYTE_CSV, DEFAULT_DESCRIPTOR_CSV)
    per_task_summary = train_per_task_models(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=35,
    )
    multitask_metrics = train_multitask_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=35,
    )
    write_recommended_model_summary(per_task_summary, multitask_metrics)
    generate()
    make_electrolyte_figures()


if __name__ == "__main__":
    main()

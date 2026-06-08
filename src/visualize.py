"""Create figures for metrics and generated molecules."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/targetmolgraph-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/targetmolgraph-cache")

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from .config import FIGURES_DIR, GENERATED_DIR, RESULTS_DIR


def plot_metrics(metrics_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(metrics_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["property"], df["MAE"], color="#3b82f6")
    ax.set_ylabel("MAE")
    ax.set_title("GNN validation error by molecular property")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def plot_generated_properties(generated_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(generated_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, ["rdkit_MolLogP", "rdkit_TPSA", "rdkit_QED"], strict=True):
        if col in df:
            ax.hist(df[col].dropna(), bins=12, color="#10b981", edgecolor="white")
            ax.set_title(col.replace("rdkit_", ""))
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def draw_top_molecules(generated_csv: Path, output_png: Path, n: int = 12) -> None:
    df = pd.read_csv(generated_csv).head(n)
    mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    legends = [
        f"#{int(row.rank)} QED={row.rdkit_QED:.2f}"
        if "rdkit_QED" in df.columns
        else f"#{int(row.rank)}"
        for row in df.itertuples()
    ]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(260, 180),
        legends=legends,
        returnPNG=False,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def make_all_figures(
    metrics_csv: Path = RESULTS_DIR / "prediction_metrics.csv",
    generated_csv: Path = GENERATED_DIR / "topk_molecules.csv",
    figures_dir: Path = FIGURES_DIR,
) -> None:
    if metrics_csv.exists():
        plot_metrics(metrics_csv, figures_dir / "prediction_metrics.png")
    if generated_csv.exists():
        plot_generated_properties(generated_csv, figures_dir / "generated_properties.png")
        draw_top_molecules(generated_csv, figures_dir / "top_molecules.png")


def make_electrolyte_figures(
    summary_csv: Path = RESULTS_DIR / "electrolyte" / "per_task_model_summary.csv",
    generated_csv: Path = GENERATED_DIR / "topk_electrolytes.csv",
    figures_dir: Path = FIGURES_DIR / "electrolyte",
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(df["target"], df["MAE"], color="#2563eb")
        ax.set_ylabel("MAE")
        ax.set_title("Electrolyte per-task GNN validation error")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures_dir / "electrolyte_model_mae.png", dpi=180)
        plt.close(fig)

    if generated_csv.exists():
        df = pd.read_csv(generated_csv)
        cols = [
            "pred_Es-Ea (eV)",
            "pred_LUMO_sol (eV)",
            "pred_HOMO_sol (eV)",
            "pred_Dielectric constant of solvents",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, col in zip(axes.ravel(), cols, strict=True):
            if col in df:
                ax.hist(df[col].dropna(), bins=10, color="#059669", edgecolor="white")
                ax.set_title(col.replace("pred_", ""))
        fig.tight_layout()
        fig.savefig(figures_dir / "generated_electrolyte_properties.png", dpi=180)
        plt.close(fig)

        mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"].head(12)]
        legends = [
            f"#{int(row['rank'])} eps={row['pred_Dielectric constant of solvents']:.1f}"
            for _, row in df.head(12).iterrows()
        ]
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(260, 180),
            legends=legends,
            returnPNG=False,
        )
        image.save(figures_dir / "top_electrolyte_candidates.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "prediction_metrics.csv")
    parser.add_argument("--generated", type=Path, default=GENERATED_DIR / "topk_molecules.csv")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_all_figures(args.metrics, args.generated, args.figures_dir)
    print(f"Wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()

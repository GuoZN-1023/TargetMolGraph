"""RDKit and graph-theory validation utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .compute_properties import compute_properties
from .config import GENERATED_DIR, GENERATION_CONFIG, RESULTS_DIR


FUNCTIONAL_GROUP_SMARTS = {
    "hydroxyl": "[OX2H]",
    "amine": "[NX3;H2,H1,H0]",
    "carbonyl": "[CX3]=[OX1]",
    "ester": "[CX3](=O)[OX2H0]",
    "ether": "[OD2]([#6])[#6]",
    "aromatic_ring": "a1aaaaa1",
}


def ring_rank(mol: Chem.Mol) -> int:
    nodes = mol.GetNumAtoms()
    edges = mol.GetNumBonds()
    components = len(Chem.GetMolFrags(mol))
    return int(edges - nodes + components)


def canonical_valid_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def graph_constraints_ok(
    smiles: str,
    max_atoms: int = GENERATION_CONFIG.max_atoms,
    max_ring_rank: int = GENERATION_CONFIG.max_ring_rank,
) -> bool:
    canonical = canonical_valid_smiles(smiles)
    if canonical is None:
        return False
    mol = Chem.MolFromSmiles(canonical)
    assert mol is not None
    if mol.GetNumAtoms() > max_atoms:
        return False
    if ring_rank(mol) > max_ring_rank:
        return False
    return True


def has_functional_group(smiles: str, group_name: str) -> bool:
    if group_name not in FUNCTIONAL_GROUP_SMARTS:
        raise KeyError(f"Unknown group: {group_name}")
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts(FUNCTIONAL_GROUP_SMARTS[group_name])
    return bool(mol is not None and pattern is not None and mol.HasSubstructMatch(pattern))


def validate_table(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    records = []
    for _, row in df.iterrows():
        smiles = str(row["smiles"])
        canonical = canonical_valid_smiles(smiles)
        valid = canonical is not None and graph_constraints_ok(canonical)
        props = compute_properties(canonical) if valid and canonical else None
        records.append(
            {
                **row.to_dict(),
                "canonical_smiles": canonical,
                "valid": valid,
                "ring_rank": ring_rank(Chem.MolFromSmiles(canonical)) if valid else None,
                **({f"rdkit_{k}": v for k, v in props.items()} if props else {}),
            }
        )
    table = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=GENERATED_DIR / "topk_molecules.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "validated_molecules.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = validate_table(args.input, args.output)
    print(f"Wrote {len(table)} validation rows to {args.output}")


if __name__ == "__main__":
    main()

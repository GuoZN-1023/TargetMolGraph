"""Compute molecular property labels with RDKit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

from .config import (
    EXPANDED_RDKIT_DESCRIPTOR_NAMES,
    MORGAN_FINGERPRINT_BITS,
    PROPERTY_NAMES,
    PROCESSED_DIR,
    RAW_DIR,
    RDKIT_DESCRIPTOR_NAMES,
)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_properties(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return {
        "MolLogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "QED": float(QED.qed(mol)),
        "MolWt": float(Descriptors.MolWt(mol)),
        "RingCount": float(Descriptors.RingCount(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
    }


def compute_rdkit_descriptors(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    def smarts_count(pattern: str) -> float:
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            return 0.0
        return float(len(mol.GetSubstructMatches(query)))

    atom_counts = {
        symbol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)
        for symbol in ("C", "O", "N", "S", "P", "F", "Cl")
    }

    descriptors = {
        "MolLogP": float(Crippen.MolLogP(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "QED": float(QED.qed(mol)),
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "RingCount": float(Descriptors.RingCount(mol)),
        "NumAliphaticRings": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "NumAromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "NumSaturatedRings": float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "NumHeteroatoms": float(Descriptors.NumHeteroatoms(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "NumValenceElectrons": float(Descriptors.NumValenceElectrons(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "BertzCT": float(Descriptors.BertzCT(mol)),
        "BalabanJ": float(Descriptors.BalabanJ(mol)),
        "HallKierAlpha": float(Descriptors.HallKierAlpha(mol)),
        "Kappa1": float(Descriptors.Kappa1(mol)),
        "Kappa2": float(Descriptors.Kappa2(mol)),
        "Kappa3": float(Descriptors.Kappa3(mol)),
        "Chi0v": float(Descriptors.Chi0v(mol)),
        "Chi1v": float(Descriptors.Chi1v(mol)),
        "Chi2v": float(Descriptors.Chi2v(mol)),
        "MaxPartialCharge": float(Descriptors.MaxPartialCharge(mol)),
        "MinPartialCharge": float(Descriptors.MinPartialCharge(mol)),
        "MaxAbsPartialCharge": float(Descriptors.MaxAbsPartialCharge(mol)),
        "MinAbsPartialCharge": float(Descriptors.MinAbsPartialCharge(mol)),
        "NumCarbonAtoms": float(atom_counts["C"]),
        "NumOxygenAtoms": float(atom_counts["O"]),
        "NumNitrogenAtoms": float(atom_counts["N"]),
        "NumSulfurAtoms": float(atom_counts["S"]),
        "NumPhosphorusAtoms": float(atom_counts["P"]),
        "NumFluorineAtoms": float(atom_counts["F"]),
        "NumChlorineAtoms": float(atom_counts["Cl"]),
        "NumCarbonylGroups": smarts_count("[CX3]=[OX1]"),
        "NumCarbonateGroups": smarts_count("[OX2][CX3](=[OX1])[OX2]"),
        "NumEtherOxygens": smarts_count("[OD2]([#6])[#6]"),
        "NumCFBonds": smarts_count("[#6]-[F]"),
        "NumCF2Groups": smarts_count("[#6]([F])([F])"),
        "NumCF3Groups": smarts_count("[#6]([F])([F])([F])"),
    }
    descriptor_function_names = [
        "MaxEStateIndex",
        "MinEStateIndex",
        "MaxAbsEStateIndex",
        "MinAbsEStateIndex",
        "FpDensityMorgan1",
        "FpDensityMorgan2",
        "FpDensityMorgan3",
        "NHOHCount",
        "NOCount",
        "NumAromaticCarbocycles",
        "NumAromaticHeterocycles",
        "NumAliphaticCarbocycles",
        "NumAliphaticHeterocycles",
        "NumSaturatedCarbocycles",
        "NumSaturatedHeterocycles",
        "Ipc",
        *[f"PEOE_VSA{idx}" for idx in range(1, 15)],
        *[f"SMR_VSA{idx}" for idx in range(1, 11)],
        *[f"SlogP_VSA{idx}" for idx in range(1, 13)],
        *[f"EState_VSA{idx}" for idx in range(1, 12)],
        *[f"VSA_EState{idx}" for idx in range(1, 11)],
    ]
    for name in descriptor_function_names:
        descriptors[name] = float(getattr(Descriptors, name)(mol))
    descriptors["NumBridgeheadAtoms"] = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    descriptors["NumSpiroAtoms"] = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    return {name: descriptors[name] for name in EXPANDED_RDKIT_DESCRIPTOR_NAMES}


def compute_morgan_fingerprint(
    smiles: str,
    n_bits: int = MORGAN_FINGERPRINT_BITS,
    radius: int = 2,
) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bitvect = generator.GetFingerprint(mol)
    return {f"morgan_{idx}": float(bitvect.GetBit(idx)) for idx in range(n_bits)}


def build_property_table(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "smiles" not in df.columns:
        raise ValueError(f"{input_csv} must contain a 'smiles' column")

    records: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for raw_smiles in df["smiles"].dropna():
        smiles = canonicalize_smiles(raw_smiles)
        if smiles is None or smiles in seen:
            continue
        props = compute_properties(smiles)
        if props is None:
            continue
        seen.add(smiles)
        records.append({"smiles": smiles, **props})

    table = pd.DataFrame(records, columns=["smiles", *PROPERTY_NAMES])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR / "molecules.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "properties.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_property_table(args.input, args.output)
    print(f"Wrote {len(table)} molecules to {args.output}")


if __name__ == "__main__":
    main()

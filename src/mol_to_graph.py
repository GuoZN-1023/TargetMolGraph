"""Convert RDKit molecules into tensor graph dictionaries."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from rdkit import Chem

from .config import FEATURE_CONFIG, FeatureConfig


def one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices] + [
        0.0 if value in choices else 1.0
    ]


def atom_features(atom: Chem.Atom, cfg: FeatureConfig = FEATURE_CONFIG) -> list[float]:
    values: list[float] = []
    symbol = atom.GetSymbol()
    values.extend(one_hot(symbol, cfg.atom_types))
    if cfg.include_atomic_number:
        values.append(atom.GetAtomicNum() / 100.0)
    if cfg.include_degree:
        values.append(atom.GetDegree() / 6.0)
    if cfg.include_total_hs:
        values.append(atom.GetTotalNumHs() / 4.0)
    if cfg.include_formal_charge:
        values.append(float(atom.GetFormalCharge()))
    if cfg.include_aromatic:
        values.append(float(atom.GetIsAromatic()))
    if cfg.include_ring:
        values.append(float(atom.IsInRing()))
    if cfg.include_hybridization:
        values.extend(one_hot(str(atom.GetHybridization()), cfg.hybridizations))
    return values


def bond_features(bond: Chem.Bond, cfg: FeatureConfig = FEATURE_CONFIG) -> list[float]:
    bond_type = str(bond.GetBondType())
    values = one_hot(bond_type, cfg.bond_types)
    values.extend([float(bond.GetIsConjugated()), float(bond.IsInRing())])
    return values


def mol_to_graph(smiles: str, y: list[float] | np.ndarray | None = None) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.SanitizeMol(mol)

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    edges: list[tuple[int, int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        attr = bond_features(bond)
        edges.extend([(i, j), (j, i)])
        edge_attrs.extend([attr, attr])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_features_length())), dtype=torch.float32)

    graph = {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    if y is not None:
        graph["y"] = torch.tensor(y, dtype=torch.float32)
    return graph


def attach_graph_features(graph: dict, features: list[float] | np.ndarray) -> dict:
    graph = dict(graph)
    graph["graph_features"] = torch.tensor(features, dtype=torch.float32)
    return graph


def bond_features_length(cfg: FeatureConfig = FEATURE_CONFIG) -> list[float]:
    return [0.0] * (len(cfg.bond_types) + 1 + 2)


def feature_metadata() -> dict:
    probe = Chem.MolFromSmiles("CCO")
    assert probe is not None
    return {
        "feature_config": asdict(FEATURE_CONFIG),
        "num_node_features": len(atom_features(probe.GetAtomWithIdx(0))),
        "num_edge_features": len(bond_features(probe.GetBondWithIdx(0))),
    }

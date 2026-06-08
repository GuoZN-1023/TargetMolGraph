"""Dataset-grounded functional-motif matching for generation scoring.

The matching layer is intentionally separate from the GNN. The GNN learns
continuous target-property prediction. Matching learns which C/O/F subgraphs are
enriched in strong RE&WSE examples, then adds a transparent structural steering
term to beam search.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from .config import TargetRange


@dataclass(frozen=True)
class MotifSpec:
    name: str
    smarts: str
    weight: float
    max_matches: int = 1


MOTIF_LIBRARY: dict[str, MotifSpec] = {
    "ether": MotifSpec("ether", "[OD2]([#6])[#6]", 0.45, 2),
    "carbonyl": MotifSpec("carbonyl", "[CX3]=[OX1]", 0.55, 1),
    "ester": MotifSpec("ester", "[CX3](=O)[OX2][#6]", 0.80, 1),
    "carbonate": MotifSpec("carbonate", "[OX2][CX3](=[OX1])[OX2]", 0.90, 1),
    "acetal": MotifSpec("acetal", "[OD2][CX4]([OD2])", 0.70, 1),
    "fluoroalkyl": MotifSpec("fluoroalkyl", "[#6]-[F]", 0.25, 3),
    "trifluoromethyl": MotifSpec("trifluoromethyl", "[#6]([F])([F])([F])", 0.75, 1),
    "lactone": MotifSpec("lactone", "[CX3;R](=[OX1])([OX2;R])[#6;R]", 0.85, 1),
    "beta_lactone": MotifSpec("beta_lactone", "O=C1OCC1", 0.70, 1),
    "gamma_lactone": MotifSpec("gamma_lactone", "O=C1OCCC1", 0.85, 1),
    "delta_lactone": MotifSpec("delta_lactone", "O=C1OCCCC1", 0.80, 1),
    "cyclic_carbonate": MotifSpec("cyclic_carbonate", "O=C1OCCO1", 0.95, 1),
    "oxetane": MotifSpec("oxetane", "[OD2;R]1[CX4;R][CX4;R][CX4;R]1", 0.35, 1),
    "thf": MotifSpec("thf", "[OD2;R]1[CX4;R][CX4;R][CX4;R][CX4;R]1", 0.55, 1),
    "thp": MotifSpec("thp", "[OD2;R]1[CX4;R][CX4;R][CX4;R][CX4;R][CX4;R]1", 0.50, 1),
    "oxepane": MotifSpec("oxepane", "[OD2;R]1[CX4;R][CX4;R][CX4;R][CX4;R][CX4;R][CX4;R]1", 0.35, 1),
    "dioxolane_1_3": MotifSpec("dioxolane_1_3", "[OD2;R]1[CX4;R][OD2;R][CX4;R][CX4;R]1", 0.75, 1),
    "dioxane_1_3": MotifSpec("dioxane_1_3", "[OD2;R]1[CX4;R][OD2;R][CX4;R][CX4;R][CX4;R]1", 0.70, 1),
    "dioxane_1_4": MotifSpec("dioxane_1_4", "[OD2;R]1[CX4;R][CX4;R][OD2;R][CX4;R][CX4;R]1", 0.70, 1),
    "glyme_chain": MotifSpec("glyme_chain", "[OD2]([#6])[#6][#6][OD2]([#6])", 0.60, 2),
}


@dataclass(frozen=True)
class LearnedMotif:
    name: str
    weight: float
    support_good: float
    support_bad: float
    enrichment: float
    odds_ratio: float


@dataclass(frozen=True)
class LearnedSubstructure:
    name: str
    smarts: str
    weight: float
    support_good: float
    support_bad: float
    enrichment: float
    odds_ratio: float


@dataclass(frozen=True)
class MotifProfile:
    motif_weights: dict[str, float]
    learned_substructures: tuple[LearnedSubstructure, ...]
    reference_smiles: tuple[str, ...]


def _motif_specs(names: tuple[str, ...]) -> list[MotifSpec]:
    unknown = [name for name in names if name not in MOTIF_LIBRARY]
    if unknown:
        raise KeyError(f"Unknown motif names: {unknown}")
    return [MOTIF_LIBRARY[name] for name in names]


def motif_occurrences(smiles: str, motif_names: tuple[str, ...]) -> dict[str, list[tuple[int, ...]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: [] for name in motif_names}

    occurrences: dict[str, list[tuple[int, ...]]] = {}
    for spec in _motif_specs(motif_names):
        query = Chem.MolFromSmarts(spec.smarts)
        if query is None:
            occurrences[spec.name] = []
            continue
        matches = sorted({tuple(match) for match in mol.GetSubstructMatches(query)})
        occurrences[spec.name] = matches
    return occurrences


def matched_motif_names(smiles: str, motif_names: tuple[str, ...]) -> tuple[str, ...]:
    occurrences = motif_occurrences(smiles, motif_names)
    return tuple(name for name in motif_names if occurrences.get(name))


def motif_match_score(
    smiles: str,
    motif_names: tuple[str, ...],
    motif_weights: dict[str, float] | None = None,
) -> float:
    """Maximum-weight motif assignment score.

    Target motif slots form one side of the bipartite graph. Actual molecule
    substructure occurrences form the other side. An edge exists when an
    occurrence satisfies a target motif slot, with either a learned RE&WSE
    enrichment weight or the motif's curated fallback weight. The dynamic
    program below solves the maximum-weight assignment for the small motif sets
    used during generation.
    """

    specs = _motif_specs(motif_names)
    occurrences_by_name = motif_occurrences(smiles, motif_names)
    target_slots: list[MotifSpec] = []
    occurrence_keys: list[tuple[str, tuple[int, ...]]] = []
    negative_score = 0.0
    for spec in specs:
        matches = occurrences_by_name.get(spec.name, [])
        weight = motif_weights.get(spec.name, spec.weight) if motif_weights else spec.weight
        if weight < 0.0:
            negative_score += weight * min(len(matches), spec.max_matches)
            continue
        target_slots.extend(spec for _ in range(spec.max_matches))
        occurrence_keys.extend((spec.name, atoms) for atoms in matches)

    if not target_slots or not occurrence_keys:
        return negative_score

    best: dict[int, float] = {0: 0.0}
    for slot in target_slots:
        updated = dict(best)
        for used_mask, score in best.items():
            for idx, (motif_name, _) in enumerate(occurrence_keys):
                if motif_name != slot.name or used_mask & (1 << idx):
                    continue
                new_mask = used_mask | (1 << idx)
                weight = motif_weights.get(slot.name, slot.weight) if motif_weights else slot.weight
                updated[new_mask] = max(updated.get(new_mask, 0.0), score + weight)
        best = updated
    return max(best.values(), default=0.0) + negative_score


def _target_distance(row: pd.Series, targets: dict[str, TargetRange]) -> float:
    miss = 0.0
    used = 0
    for name, target in targets.items():
        column = name if name in row else f"rdkit_{name}"
        if column not in row or pd.isna(row[column]):
            continue
        value = float(row[column])
        if target.lower is not None and value < target.lower:
            miss += target.weight * (target.lower - value)
        elif target.upper is not None and value > target.upper:
            miss += target.weight * (value - target.upper)
        used += 1
    return miss if used else float("inf")


def learn_motif_weights_from_table(
    table: pd.DataFrame,
    motif_names: tuple[str, ...],
    targets: dict[str, TargetRange],
    top_fraction: float = 0.25,
    min_support_good: float = 0.03,
) -> dict[str, LearnedMotif]:
    """Learn motif weights from RE&WSE target enrichment.

    Molecules are ranked by weighted target-distance. The best quantile is the
    target-success group; the worst quantile is the contrast group. Motifs that
    occur more often in the success group get larger positive weights.
    """

    smiles_col = "canonical_smiles" if "canonical_smiles" in table.columns else "SMILES"
    if smiles_col not in table.columns:
        raise ValueError("table must contain canonical_smiles or SMILES")

    scored = table.copy()
    scored["_target_distance"] = scored.apply(lambda row: _target_distance(row, targets), axis=1)
    scored = scored.replace([float("inf"), -float("inf")], pd.NA).dropna(
        subset=[smiles_col, "_target_distance"]
    )
    if scored.empty:
        return {}

    group_size = max(1, int(len(scored) * top_fraction))
    ranked = scored.sort_values("_target_distance", ascending=True)
    good = ranked.head(group_size)
    bad = ranked.tail(group_size)

    learned: dict[str, LearnedMotif] = {}
    for spec in _motif_specs(motif_names):
        good_hits = good[smiles_col].apply(lambda smiles: bool(motif_occurrences(str(smiles), (spec.name,))[spec.name]))
        bad_hits = bad[smiles_col].apply(lambda smiles: bool(motif_occurrences(str(smiles), (spec.name,))[spec.name]))
        support_good = float(good_hits.mean()) if len(good_hits) else 0.0
        support_bad = float(bad_hits.mean()) if len(bad_hits) else 0.0
        enrichment = support_good - support_bad
        odds_ratio = (support_good + 1.0e-6) / (support_bad + 1.0e-6)
        if support_bad >= min_support_good and enrichment < 0.0:
            inverse_odds = (support_bad + 1.0e-6) / (support_good + 1.0e-6)
            weight = -spec.weight * (1.0 + abs(enrichment)) * min(2.0, inverse_odds)
        elif support_good < min_support_good or enrichment <= 0.0:
            weight = max(0.05, spec.weight * 0.25)
        else:
            weight = spec.weight * (1.0 + enrichment) * min(3.0, odds_ratio)
        learned[spec.name] = LearnedMotif(
            name=spec.name,
            weight=float(weight),
            support_good=support_good,
            support_bad=support_bad,
            enrichment=enrichment,
            odds_ratio=float(odds_ratio),
        )
    return learned


def load_or_learn_motif_weights(
    descriptor_csv: Path,
    motif_names: tuple[str, ...],
    targets: dict[str, TargetRange],
) -> dict[str, float]:
    if not descriptor_csv.exists():
        return {spec.name: spec.weight for spec in _motif_specs(motif_names)}
    table = pd.read_csv(descriptor_csv)
    learned = learn_motif_weights_from_table(table, motif_names, targets)
    if not learned:
        return {spec.name: spec.weight for spec in _motif_specs(motif_names)}
    return {name: motif.weight for name, motif in learned.items()}


def _ranked_target_groups(
    table: pd.DataFrame,
    targets: dict[str, TargetRange],
    top_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    smiles_col = "canonical_smiles" if "canonical_smiles" in table.columns else "SMILES"
    if smiles_col not in table.columns:
        return pd.DataFrame(), pd.DataFrame(), smiles_col

    scored = table.copy()
    scored["_target_distance"] = scored.apply(lambda row: _target_distance(row, targets), axis=1)
    scored = scored.replace([float("inf"), -float("inf")], pd.NA).dropna(
        subset=[smiles_col, "_target_distance"]
    )
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame(), smiles_col

    group_size = max(1, int(len(scored) * top_fraction))
    ranked = scored.sort_values("_target_distance", ascending=True)
    return ranked.head(group_size), ranked.tail(group_size), smiles_col


def _bond_smarts(bond: Chem.Bond) -> str:
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "="
    if bond.GetBondType() == Chem.BondType.TRIPLE:
        return "#"
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        return ":"
    return "-"


def _atom_smarts(atom: Chem.Atom) -> str:
    atomic_num = atom.GetAtomicNum()
    if atom.IsInRing():
        return f"[#{atomic_num};R]"
    return f"[#{atomic_num}]"


def _fragment_smarts(smiles: str, min_path_atoms: int = 3, max_path_atoms: int = 5) -> set[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return set()

    fragments: set[str] = set()
    allowed = {"C", "O", "F"}

    def dfs(path: list[int]) -> None:
        if len(path) >= min_path_atoms:
            atom_parts = [_atom_smarts(mol.GetAtomWithIdx(atom_idx)) for atom_idx in path]
            bond_parts: list[str] = []
            for prev_idx, atom_idx in zip(path[:-1], path[1:], strict=True):
                bond = mol.GetBondBetweenAtoms(prev_idx, atom_idx)
                if bond is None:
                    return
                bond_parts.append(_bond_smarts(bond))
            forward = atom_parts[0]
            for bond_part, atom_part in zip(bond_parts, atom_parts[1:], strict=True):
                forward += bond_part + atom_part
            reverse = atom_parts[-1]
            for bond_part, atom_part in zip(reversed(bond_parts), reversed(atom_parts[:-1]), strict=True):
                reverse += bond_part + atom_part
            smarts = min(forward, reverse)
            if Chem.MolFromSmarts(smarts) is not None:
                fragments.add(smarts)
        if len(path) >= max_path_atoms:
            return
        current = mol.GetAtomWithIdx(path[-1])
        for neighbor in current.GetNeighbors():
            idx = neighbor.GetIdx()
            if idx in path or neighbor.GetSymbol() not in allowed:
                continue
            dfs([*path, idx])

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in allowed:
            dfs([atom.GetIdx()])
    return fragments


def learn_substructure_priors_from_table(
    table: pd.DataFrame,
    targets: dict[str, TargetRange],
    top_fraction: float = 0.25,
    min_support: float = 0.06,
    max_substructures: int = 16,
    max_mining_molecules: int = 32,
) -> tuple[LearnedSubstructure, ...]:
    """Mine small C/O/F fragments enriched in target-close or target-far molecules."""

    good, bad, smiles_col = _ranked_target_groups(table, targets, top_fraction)
    if good.empty or bad.empty:
        return ()
    good = good.head(max_mining_molecules)
    bad = bad.tail(max_mining_molecules)

    good_counts: Counter[str] = Counter()
    bad_counts: Counter[str] = Counter()
    for smiles in good[smiles_col]:
        good_counts.update(_fragment_smarts(str(smiles)))
    for smiles in bad[smiles_col]:
        bad_counts.update(_fragment_smarts(str(smiles)))

    learned: list[LearnedSubstructure] = []
    all_fragments = sorted(set(good_counts) | set(bad_counts))
    for smarts in all_fragments:
        support_good = good_counts[smarts] / len(good)
        support_bad = bad_counts[smarts] / len(bad)
        enrichment = support_good - support_bad
        if max(support_good, support_bad) < min_support or abs(enrichment) < min_support:
            continue
        odds_ratio = (support_good + 1.0e-6) / (support_bad + 1.0e-6)
        if enrichment > 0:
            weight = 0.30 * (1.0 + enrichment) * min(3.0, odds_ratio)
        else:
            inverse_odds = (support_bad + 1.0e-6) / (support_good + 1.0e-6)
            weight = -0.30 * (1.0 + abs(enrichment)) * min(2.0, inverse_odds)
        learned.append(
            LearnedSubstructure(
                name="",
                smarts=smarts,
                weight=float(weight),
                support_good=float(support_good),
                support_bad=float(support_bad),
                enrichment=float(enrichment),
                odds_ratio=float(odds_ratio),
            )
        )

    learned.sort(key=lambda item: abs(item.weight), reverse=True)
    named = [
        LearnedSubstructure(
            name=f"learned_{idx:02d}_{'pos' if item.weight > 0 else 'neg'}",
            smarts=item.smarts,
            weight=item.weight,
            support_good=item.support_good,
            support_bad=item.support_bad,
            enrichment=item.enrichment,
            odds_ratio=item.odds_ratio,
        )
        for idx, item in enumerate(learned[:max_substructures], start=1)
    ]
    return tuple(named)


def learned_substructure_score(
    smiles: str,
    substructures: tuple[LearnedSubstructure, ...],
    score_cap: float = 4.0,
) -> float:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0.0
    score = 0.0
    for substructure in substructures:
        query = Chem.MolFromSmarts(substructure.smarts)
        if query is None:
            continue
        if mol.HasSubstructMatch(query):
            score += substructure.weight
    return float(max(-score_cap, min(score_cap, score)))


def matched_learned_substructure_names(
    smiles: str,
    substructures: tuple[LearnedSubstructure, ...],
) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ()
    names: list[str] = []
    for substructure in substructures:
        query = Chem.MolFromSmarts(substructure.smarts)
        if query is not None and mol.HasSubstructMatch(query):
            names.append(substructure.name)
    return tuple(names)


def _reference_smiles_from_table(
    table: pd.DataFrame,
    targets: dict[str, TargetRange],
    top_fraction: float = 0.15,
    max_references: int = 64,
) -> tuple[str, ...]:
    good, _, smiles_col = _ranked_target_groups(table, targets, top_fraction)
    if good.empty:
        return ()
    references: list[str] = []
    for smiles in good[smiles_col]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        references.append(Chem.MolToSmiles(mol, canonical=True))
        if len(references) >= max_references:
            break
    return tuple(dict.fromkeys(references))


def reference_smiles_from_table(
    descriptor_csv: Path,
    max_references: int | None = None,
) -> tuple[str, ...]:
    if not descriptor_csv.exists():
        return ()
    table = pd.read_csv(descriptor_csv)
    smiles_col = "canonical_smiles" if "canonical_smiles" in table.columns else "SMILES"
    if smiles_col not in table.columns:
        return ()

    references: list[str] = []
    for smiles in table[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        references.append(Chem.MolToSmiles(mol, canonical=True))
        if max_references is not None and len(references) >= max_references:
            break
    return tuple(dict.fromkeys(references))


class FingerprintSimilarityScorer:
    def __init__(self, reference_smiles: tuple[str, ...]):
        self.reference_fps = []
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            self.reference_fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    def score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or not self.reference_fps:
            return 0.0
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, self.reference_fps), default=0.0))


def load_or_learn_motif_profile(
    descriptor_csv: Path,
    motif_names: tuple[str, ...],
    targets: dict[str, TargetRange],
) -> MotifProfile:
    motif_weights = {spec.name: spec.weight for spec in _motif_specs(motif_names)}
    if not descriptor_csv.exists():
        return MotifProfile(motif_weights, (), ())

    table = pd.read_csv(descriptor_csv)
    learned_motifs = learn_motif_weights_from_table(table, motif_names, targets)
    if learned_motifs:
        motif_weights = {name: motif.weight for name, motif in learned_motifs.items()}

    learned_substructures = learn_substructure_priors_from_table(table, targets)
    reference_smiles = _reference_smiles_from_table(table, targets)
    return MotifProfile(motif_weights, learned_substructures, reference_smiles)

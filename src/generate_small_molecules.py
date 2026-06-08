"""Generate small C/H/O/F molecules with constrained graph beam search."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from .beam_search import interval_distance
from .compute_properties import compute_properties
from .config import (
    ELECTROLYTE_TARGETS,
    GENERATED_DIR,
    PROPERTY_NAMES,
    RESULTS_DIR,
    SMALL_MOLECULE_GENERATION_CONFIG,
    SmallMoleculeGenerationConfig,
    TargetRange,
)
from .generate_electrolytes import ElectrolytePredictorEnsemble
from .validate_rdkit import ring_rank


RDLogger.DisableLog("rdApp.*")

SINGLE = Chem.BondType.SINGLE
DOUBLE = Chem.BondType.DOUBLE


@dataclass(frozen=True)
class SmallMoleculeCandidate:
    smiles: str
    score: float
    predicted: dict[str, float]
    rdkit: dict[str, float]
    satisfied: bool


def heavy_atom_symbols(mol: Chem.Mol) -> set[str]:
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def oxygen_hydrogen_bonded(mol: Chem.Mol) -> bool:
    return any(atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0 for atom in mol.GetAtoms())


def forbidden_heavy_bonded(mol: Chem.Mol, cfg: SmallMoleculeGenerationConfig) -> bool:
    forbidden = {frozenset(pair) for pair in getattr(cfg, "forbidden_bond_pairs", ())}
    return any(
        frozenset((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())) in forbidden
        for bond in mol.GetBonds()
    )


def ring_sizes_ok(mol: Chem.Mol, cfg: SmallMoleculeGenerationConfig) -> bool:
    allowed = set(getattr(cfg, "allowed_ring_sizes", ()))
    if not allowed:
        return True
    return all(len(ring) in allowed for ring in mol.GetRingInfo().AtomRings())


def bridged_rings_ok(mol: Chem.Mol, cfg: SmallMoleculeGenerationConfig) -> bool:
    if not getattr(cfg, "forbid_bridged_rings", False):
        return True
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol) == 0


def forbidden_smarts_ok(mol: Chem.Mol, cfg: SmallMoleculeGenerationConfig) -> bool:
    for _, smarts in getattr(cfg, "forbidden_smarts", ()):
        pattern = Chem.MolFromSmarts(str(smarts))
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return False
    return True


def triple_bonds_ok(mol: Chem.Mol, cfg: SmallMoleculeGenerationConfig) -> bool:
    if not getattr(cfg, "reject_triple_bond_in_ring", False):
        return True
    return not any(
        bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing()
        for bond in mol.GetBonds()
    )


def canonical_small_smiles(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> str | None:
    """Return canonical SMILES if the molecule satisfies all hard graph rules."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if not heavy_atom_symbols(mol).issubset(set(cfg.allowed_heavy_atoms)):
        return None
    heavy_atom_count = Descriptors.HeavyAtomCount(mol)
    if heavy_atom_count > cfg.max_heavy_atoms or heavy_atom_count < getattr(cfg, "min_heavy_atoms", 1):
        return None
    if getattr(cfg, "require_oxygen", False) and "O" not in heavy_atom_symbols(mol):
        return None
    if getattr(cfg, "forbid_oxygen_hydrogen", False) and oxygen_hydrogen_bonded(mol):
        return None
    if forbidden_heavy_bonded(mol, cfg):
        return None
    max_mol_weight = getattr(cfg, "max_mol_weight", cfg.max_mol_wt)
    if Descriptors.MolWt(mol) >= max_mol_weight:
        return None
    if ring_rank(mol) > cfg.max_ring_rank:
        return None
    if not ring_sizes_ok(mol, cfg):
        return None
    if not bridged_rings_ok(mol, cfg):
        return None
    if not triple_bonds_ok(mol, cfg):
        return None
    if not forbidden_smarts_ok(mol, cfg):
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def graph_constraints_ok(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> bool:
    return canonical_small_smiles(smiles, cfg) is not None


def try_canonical(
    mol: Chem.Mol,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> str | None:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return canonical_small_smiles(Chem.MolToSmiles(mol, canonical=True), cfg)


def allowed_double_bond(mol: Chem.Mol, atom_idx: int, new_symbol: str) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    pair = {atom.GetSymbol(), new_symbol}
    return "F" not in pair and pair in ({"C"}, {"C", "O"})


def expand_by_atom(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Descriptors.HeavyAtomCount(mol) >= cfg.max_heavy_atoms:
        return set()

    outputs: set[str] = set()
    for atom_idx in range(mol.GetNumAtoms()):
        for symbol in cfg.atom_choices:
            bond_types = [SINGLE]
            if allowed_double_bond(mol, atom_idx, symbol):
                bond_types.append(DOUBLE)
            for bond_type in bond_types:
                rw = Chem.RWMol(mol)
                new_idx = rw.AddAtom(Chem.Atom(symbol))
                rw.AddBond(atom_idx, new_idx, bond_type)
                canonical = try_canonical(rw.GetMol(), cfg)
                if canonical:
                    outputs.add(canonical)
    return outputs


def expand_by_atom_substitution(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    outputs: set[str] = set()
    periodic_table = Chem.GetPeriodicTable()
    for atom_idx in range(mol.GetNumAtoms()):
        current = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        for symbol in cfg.atom_choices:
            if symbol == current:
                continue
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetAtomicNum(periodic_table.GetAtomicNumber(symbol))
            canonical = try_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
    return outputs


def expand_by_bond_insertion(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Descriptors.HeavyAtomCount(mol) >= cfg.max_heavy_atoms:
        return set()

    outputs: set[str] = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() not in {SINGLE, DOUBLE}:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        for symbol in cfg.atom_choices:
            rw = Chem.RWMol(mol)
            rw.RemoveBond(begin, end)
            new_idx = rw.AddAtom(Chem.Atom(symbol))
            rw.AddBond(begin, new_idx, SINGLE)
            rw.AddBond(new_idx, end, SINGLE)
            canonical = try_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
    return outputs


def expand_by_bond_edit(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    outputs: set[str] = set()
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_idx = begin.GetIdx()
        end_idx = end.GetIdx()
        if bond.GetBondType() == SINGLE:
            if "F" in {begin.GetSymbol(), end.GetSymbol()}:
                continue
            if {begin.GetSymbol(), end.GetSymbol()} not in ({"C"}, {"C", "O"}):
                continue
            new_bond = DOUBLE
        elif bond.GetBondType() == DOUBLE:
            new_bond = SINGLE
        else:
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveBond(begin_idx, end_idx)
        rw.AddBond(begin_idx, end_idx, new_bond)
        canonical = try_canonical(rw.GetMol(), cfg)
        if canonical:
            outputs.add(canonical)
    return outputs


def expand_by_terminal_atom_deletion(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Descriptors.HeavyAtomCount(mol) <= getattr(cfg, "min_heavy_atoms", 1):
        return set()

    outputs: set[str] = set()
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 1:
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(atom.GetIdx())
        canonical = try_canonical(rw.GetMol(), cfg)
        if canonical:
            outputs.add(canonical)
    return outputs


def expand_by_ring_closure(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 4 or ring_rank(mol) >= cfg.max_ring_rank:
        return set()

    outputs: set[str] = set()
    distances = Chem.GetDistanceMatrix(mol)
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue
            if distances[i, j] < 3:
                continue
            ring_size = int(distances[i, j]) + 1
            if ring_size not in set(getattr(cfg, "allowed_ring_sizes", ())):
                continue
            if "F" in {mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()}:
                continue
            rw = Chem.RWMol(mol)
            rw.AddBond(i, j, SINGLE)
            canonical = try_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
    return outputs


def expand_by_fragment(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    outputs: set[str] = set()
    max_outputs = cfg.max_fragment_expansions_per_molecule
    for fragment_smiles in cfg.fragment_smiles:
        fragment = Chem.MolFromSmiles(fragment_smiles)
        if fragment is None:
            continue
        if mol.GetNumAtoms() + fragment.GetNumAtoms() > cfg.max_heavy_atoms:
            continue
        combined = Chem.CombineMols(mol, fragment)
        offset = mol.GetNumAtoms()
        for atom_idx in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "F":
                continue
            for frag_idx in range(fragment.GetNumAtoms()):
                if fragment.GetAtomWithIdx(frag_idx).GetSymbol() == "F":
                    continue
                rw = Chem.RWMol(combined)
                rw.AddBond(atom_idx, offset + frag_idx, SINGLE)
                canonical = try_canonical(rw.GetMol(), cfg)
                if canonical:
                    outputs.add(canonical)
                    if len(outputs) >= max_outputs:
                        return outputs
    return outputs


def expand_molecule(
    smiles: str,
    cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
) -> set[str]:
    """Apply graph edit actions to one molecular graph state."""

    enabled = set(getattr(cfg, "enabled_graph_edits", ()))
    outputs: set[str] = set()
    if "append_atom" in enabled:
        outputs.update(expand_by_atom(smiles, cfg))
    if "substitute_atom" in enabled:
        outputs.update(expand_by_atom_substitution(smiles, cfg))
    if "insert_atom_into_bond" in enabled:
        outputs.update(expand_by_bond_insertion(smiles, cfg))
    if "edit_bond" in enabled:
        outputs.update(expand_by_bond_edit(smiles, cfg))
    if "delete_terminal_atom" in enabled:
        outputs.update(expand_by_terminal_atom_deletion(smiles, cfg))
    if "ring_closure" in enabled:
        outputs.update(expand_by_ring_closure(smiles, cfg))
    if "fragment" in enabled:
        outputs.update(expand_by_fragment(smiles, cfg))
    return outputs


def satisfies_targets(
    properties: dict[str, float],
    targets: dict[str, TargetRange],
) -> bool:
    return all(
        name in properties and interval_distance(float(properties[name]), target) == 0.0
        for name, target in targets.items()
    )


def score_candidate(
    smiles: str,
    predicted: dict[str, float],
    cfg: SmallMoleculeGenerationConfig,
) -> float:
    miss = 0.0
    for name, target in cfg.targets.items():
        if name in predicted:
            miss += target.weight * interval_distance(float(predicted[name]), target)
        else:
            miss += 10.0 * target.weight

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    heavy_atoms = float(Descriptors.HeavyAtomCount(mol))
    mol_wt = float(Descriptors.MolWt(mol))
    size_penalty = 0.05 * max(0.0, heavy_atoms - 8.0)
    mass_penalty = 0.005 * max(0.0, mol_wt - 160.0)
    ring_penalty = 0.15 * max(0, ring_rank(mol) - 1)
    hit_bonus = 2.0 if satisfies_targets(predicted, cfg.targets) else 0.0
    return float(-(miss + size_penalty + mass_penalty + ring_penalty) + hit_bonus)


class SmallMoleculeGraphGenerator:
    def __init__(
        self,
        cfg: SmallMoleculeGenerationConfig = SMALL_MOLECULE_GENERATION_CONFIG,
        predictor: ElectrolytePredictorEnsemble | None = None,
    ):
        self.cfg = cfg
        self.predictor = predictor or ElectrolytePredictorEnsemble()

    def _make_candidates(self, smiles_list: list[str]) -> list[SmallMoleculeCandidate]:
        unique = sorted(
            {
                canonical
                for smiles in smiles_list
                if (canonical := canonical_small_smiles(smiles, self.cfg)) is not None
            }
        )
        if not unique:
            return []

        predictions = self.predictor.predict(unique)
        candidates: list[SmallMoleculeCandidate] = []
        for smiles, predicted in zip(unique, predictions, strict=True):
            rdkit_props = compute_properties(smiles) or {}
            score = score_candidate(smiles, predicted, self.cfg)
            candidates.append(
                SmallMoleculeCandidate(
                    smiles=smiles,
                    score=score,
                    predicted=predicted,
                    rdkit=rdkit_props,
                    satisfied=satisfies_targets(predicted, self.cfg.targets),
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def search(self) -> list[SmallMoleculeCandidate]:
        frontier = self._make_candidates(list(self.cfg.seed_smiles))
        archive: dict[str, SmallMoleculeCandidate] = {
            candidate.smiles: candidate for candidate in frontier
        }

        for _ in range(self.cfg.max_steps):
            expanded: set[str] = set()
            for candidate in frontier[: self.cfg.beam_width]:
                expanded.update(expand_molecule(candidate.smiles, self.cfg))
            if len(expanded) > self.cfg.max_expanded_per_step:
                expanded = set(sorted(expanded)[: self.cfg.max_expanded_per_step])

            for candidate in self._make_candidates(list(expanded)):
                existing = archive.get(candidate.smiles)
                if existing is None or candidate.score > existing.score:
                    archive[candidate.smiles] = candidate
            frontier = sorted(archive.values(), key=lambda item: item.score, reverse=True)[
                : self.cfg.beam_width
            ]

        return sorted(
            archive.values(),
            key=lambda item: (item.satisfied, item.score),
            reverse=True,
        )[: self.cfg.top_k]


def candidate_rows(candidates: list[SmallMoleculeCandidate]) -> list[dict]:
    rows: list[dict] = []
    for rank, candidate in enumerate(candidates, start=1):
        mol = Chem.MolFromSmiles(candidate.smiles)
        assert mol is not None
        row = {
            "rank": rank,
            "smiles": candidate.smiles,
            "score": candidate.score,
            "satisfied": candidate.satisfied,
            "heavy_atom_count": int(Descriptors.HeavyAtomCount(mol)),
            "allowed_atoms": ",".join(sorted(heavy_atom_symbols(mol))),
        }
        row.update({f"pred_{name}": candidate.predicted.get(name) for name in ELECTROLYTE_TARGETS})
        row.update({f"pred_{name}": candidate.predicted.get(name) for name in PROPERTY_NAMES})
        row.update({f"rdkit_{name}": candidate.rdkit.get(name) for name in PROPERTY_NAMES})
        rows.append(row)
    return rows


def generate(
    output_csv: Path = GENERATED_DIR / "topk_small_cof_molecules.csv",
    beam_width: int = SMALL_MOLECULE_GENERATION_CONFIG.beam_width,
    max_steps: int = SMALL_MOLECULE_GENERATION_CONFIG.max_steps,
    top_k: int = SMALL_MOLECULE_GENERATION_CONFIG.top_k,
) -> pd.DataFrame:
    cfg = replace(
        SMALL_MOLECULE_GENERATION_CONFIG,
        beam_width=beam_width,
        max_steps=max_steps,
        top_k=top_k,
    )
    generator = SmallMoleculeGraphGenerator(cfg=cfg)
    table = pd.DataFrame(candidate_rows(generator.search()))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)

    results_csv = RESULTS_DIR / "small_molecule_generation" / "generated_small_cof_molecules.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_csv, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=GENERATED_DIR / "topk_small_cof_molecules.csv")
    parser.add_argument("--beam-width", type=int, default=SMALL_MOLECULE_GENERATION_CONFIG.beam_width)
    parser.add_argument("--max-steps", type=int, default=SMALL_MOLECULE_GENERATION_CONFIG.max_steps)
    parser.add_argument("--top-k", type=int, default=SMALL_MOLECULE_GENERATION_CONFIG.top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = generate(args.output, args.beam_width, args.max_steps, args.top_k)
    hit_rate = float(table["satisfied"].mean()) if len(table) else 0.0
    print(f"Wrote {len(table)} small C/H/O/F candidates to {args.output}")
    print(f"Predicted electrolyte target hit rate: {hit_rate:.2%}")


if __name__ == "__main__":
    main()

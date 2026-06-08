"""Target-oriented molecular graph search."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from .compute_properties import (
    compute_morgan_fingerprint,
    compute_properties,
    compute_rdkit_descriptors,
)
from .config import GENERATION_CONFIG, MODELS_DIR, PROPERTY_NAMES, GenerationConfig, TargetRange
from .gnn_model import MolecularGNN, collate_graphs
from .mol_to_graph import mol_to_graph
from .motif_matching import matched_motif_names, motif_match_score
from .validate_rdkit import canonical_valid_smiles, graph_constraints_ok as base_graph_constraints_ok, ring_rank


SINGLE = Chem.BondType.SINGLE
DOUBLE = Chem.BondType.DOUBLE
RDLogger.DisableLog("rdApp.*")


@contextmanager
def dropout_enabled_only(model: nn.Module):
    previous_training = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        yield
    finally:
        for module, was_training in previous_training.items():
            module.train(was_training)


@dataclass
class Candidate:
    smiles: str
    score: float
    predicted: dict[str, float]
    rdkit: dict[str, float]
    satisfied: bool
    motif_match_score: float = 0.0
    matched_motifs: tuple[str, ...] = ()
    learned_substructure_score: float = 0.0
    matched_learned_substructures: tuple[str, ...] = ()
    similarity_score: float = 0.0
    training_similarity: float = 0.0
    uncertainty_score: float = 1.0
    target_miss: float = 0.0
    pareto_rank: int = 0
    robust_score_mean: float = 0.0
    robust_score_std: float = 0.0
    target_hit_probability: float = 0.0
    structural_filter_passed: bool = True
    structural_filter_failures: tuple[str, ...] = ()
    synthesis_proxy: float = 0.0
    failure_probability_mean: float = 0.0
    normalized_violation_mean: float = 0.0
    normalized_margin_min: float = 0.0
    target_failure_probabilities: dict[str, float] = field(default_factory=dict)
    target_normalized_violations: dict[str, float] = field(default_factory=dict)
    target_normalized_margins: dict[str, float] = field(default_factory=dict)


def heavy_atom_symbols(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def _max_heavy_atoms(cfg: GenerationConfig) -> int:
    return int(getattr(cfg, "max_heavy_atoms", cfg.max_atoms))


def _min_heavy_atoms(cfg: GenerationConfig) -> int:
    return int(getattr(cfg, "min_heavy_atoms", 1))


def _forbidden_bond_pairs(cfg: GenerationConfig) -> set[frozenset[str]]:
    return {
        frozenset(pair)
        for pair in getattr(cfg, "forbidden_bond_pairs", ())
    }


def _oxygen_hydrogen_bonded(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
            return True
    return False


def _forbidden_heavy_bonded(mol: Chem.Mol, cfg: GenerationConfig) -> bool:
    forbidden = _forbidden_bond_pairs(cfg)
    if not forbidden:
        return False
    for bond in mol.GetBonds():
        pair = frozenset(
            (
                bond.GetBeginAtom().GetSymbol(),
                bond.GetEndAtom().GetSymbol(),
            )
        )
        if pair in forbidden:
            return True
    return False


def _ring_sizes_ok(mol: Chem.Mol, cfg: GenerationConfig) -> bool:
    allowed_ring_sizes = getattr(cfg, "allowed_ring_sizes", None)
    if allowed_ring_sizes is None:
        return True
    allowed = set(allowed_ring_sizes)
    return all(len(ring) in allowed for ring in mol.GetRingInfo().AtomRings())


def _bridged_rings_ok(mol: Chem.Mol, cfg: GenerationConfig) -> bool:
    if not getattr(cfg, "forbid_bridged_rings", False):
        return True
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol) == 0


def _forbidden_smarts_ok(mol: Chem.Mol, cfg: GenerationConfig) -> bool:
    for _, smarts in getattr(cfg, "forbidden_smarts", ()):
        pattern = Chem.MolFromSmarts(str(smarts))
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return False
    return True


def _triple_bonds_ok(mol: Chem.Mol, cfg: GenerationConfig) -> bool:
    if not getattr(cfg, "reject_triple_bond_in_ring", False):
        return True
    return not any(
        bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing()
        for bond in mol.GetBonds()
    )


def graph_constraints_ok(
    smiles: str,
    max_atoms: int = GENERATION_CONFIG.max_atoms,
    max_ring_rank: int = GENERATION_CONFIG.max_ring_rank,
    allowed_heavy_atoms: tuple[str, ...] | None = None,
    cfg: GenerationConfig | None = None,
) -> bool:
    if not base_graph_constraints_ok(smiles, max_atoms, max_ring_rank):
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if allowed_heavy_atoms is not None and not heavy_atom_symbols(smiles).issubset(set(allowed_heavy_atoms)):
        return False
    if cfg is None:
        return True
    heavy_atoms = mol.GetNumAtoms()
    if heavy_atoms > _max_heavy_atoms(cfg) or heavy_atoms < _min_heavy_atoms(cfg):
        return False
    max_mol_weight = getattr(cfg, "max_mol_weight", None)
    if max_mol_weight is not None and Descriptors.MolWt(mol) > float(max_mol_weight):
        return False
    if getattr(cfg, "require_oxygen", False) and "O" not in heavy_atom_symbols(smiles):
        return False
    if getattr(cfg, "forbid_oxygen_hydrogen", False) and _oxygen_hydrogen_bonded(mol):
        return False
    if _forbidden_heavy_bonded(mol, cfg):
        return False
    if not _ring_sizes_ok(mol, cfg):
        return False
    if not _bridged_rings_ok(mol, cfg):
        return False
    if not _triple_bonds_ok(mol, cfg):
        return False
    return _forbidden_smarts_ok(mol, cfg)


class PropertyPredictor:
    def __init__(self, checkpoint_path: Path = MODELS_DIR / "gnn_model.pt"):
        self.checkpoint_path = checkpoint_path
        self.model: MolecularGNN | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None
        self.property_names = PROPERTY_NAMES
        self.metadata: dict = {}
        self.target_transform = "none"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path.exists():
            self._load(checkpoint_path)

    @property
    def uses_gnn(self) -> bool:
        return self.model is not None

    def _load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        kwargs = checkpoint.get("model_kwargs", {})
        self.property_names = checkpoint.get("property_names", PROPERTY_NAMES)
        self.metadata = checkpoint.get("metadata", {})
        self.target_transform = checkpoint.get("target_transform", "none")
        self.y_mean = checkpoint["y_mean"]
        self.y_std = checkpoint["y_std"]
        self.model = MolecularGNN(
            num_node_features=self.metadata["num_node_features"],
            num_outputs=len(self.property_names),
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.1),
            graph_feature_dim=kwargs.get("graph_feature_dim", self.metadata.get("graph_feature_dim", 0)),
            use_graph_feature_encoder=kwargs.get("use_graph_feature_encoder", False),
            num_edge_features=kwargs.get("num_edge_features", self.metadata.get("num_edge_features", 0)),
            message_passing=kwargs.get("message_passing", "gcn"),
            readout=kwargs.get("readout", "mean_max"),
            jk_mode=kwargs.get("jk_mode", "last"),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        smiles_list: list[str],
        auxiliary_values: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        if self.model is None:
            return [compute_properties(smiles) or {} for smiles in smiles_list]

        graphs = [mol_to_graph(smiles) for smiles in smiles_list]
        graph_features = self._graph_features_for_smiles(smiles_list, auxiliary_values)
        if graph_features is not None:
            for graph, features in zip(graphs, graph_features, strict=True):
                graph["graph_features"] = features
        batch = collate_graphs(graphs)
        with torch.no_grad():
            scaled = self.model(
                batch["x"].to(self.device),
                batch["edge_index"].to(self.device),
                batch["batch"].to(self.device),
                batch.get("graph_features").to(self.device)
                if batch.get("graph_features") is not None
                else None,
                batch.get("edge_attr").to(self.device)
                if batch.get("edge_attr") is not None
                else None,
            ).cpu()
        assert self.y_mean is not None and self.y_std is not None
        values = scaled * self.y_std + self.y_mean
        if self.target_transform == "log1p":
            values = torch.expm1(values)
        elif self.target_transform == "sqrt":
            values = torch.square(values).clamp_min(0.0)
        return [
            {name: float(row[idx]) for idx, name in enumerate(self.property_names)}
            for row in values
        ]

    def predict_mc(
        self,
        smiles_list: list[str],
        auxiliary_values: list[dict[str, float]] | None = None,
        samples: int = 8,
    ) -> list[list[dict[str, float]]]:
        if samples <= 0:
            return []
        if self.model is None:
            return [self.predict(smiles_list, auxiliary_values) for _ in range(samples)]
        with dropout_enabled_only(self.model):
            return [self.predict(smiles_list, auxiliary_values) for _ in range(samples)]

    def _graph_features_for_smiles(
        self,
        smiles_list: list[str],
        auxiliary_values: list[dict[str, float]] | None = None,
    ) -> list[torch.Tensor] | None:
        feature_cols = self.metadata.get("feature_columns")
        if not feature_cols:
            return None
        imputer_stats = self.metadata.get("imputer_statistics")
        mean = self.metadata.get("scaler_mean")
        scale = self.metadata.get("scaler_scale")
        if imputer_stats is None or mean is None or scale is None:
            return None

        output: list[torch.Tensor] = []
        for row_idx, smiles in enumerate(smiles_list):
            descriptors = compute_rdkit_descriptors(smiles) or {}
            fingerprint = compute_morgan_fingerprint(smiles) or {}
            aux = auxiliary_values[row_idx] if auxiliary_values is not None else {}
            raw: list[float] = []
            for idx, col in enumerate(feature_cols):
                if col.startswith("rdkit_"):
                    value = descriptors.get(col.removeprefix("rdkit_"))
                    raw.append(float(value) if value is not None else float(imputer_stats[idx]))
                elif col.startswith("morgan_"):
                    raw.append(float(fingerprint.get(col, 0.0)))
                elif col in aux:
                    raw.append(float(aux[col]))
                else:
                    raw.append(float(imputer_stats[idx]))
            scaled = [
                (value - float(mean[idx])) / max(float(scale[idx]), 1.0e-12)
                for idx, value in enumerate(raw)
            ]
            output.append(torch.tensor(scaled, dtype=torch.float32))
        return output


def interval_distance(value: float, target: TargetRange) -> float:
    if target.lower is not None and value < target.lower:
        return target.lower - value
    if target.upper is not None and value > target.upper:
        return value - target.upper
    return 0.0


def satisfies_targets(properties: dict[str, float], targets: dict[str, TargetRange]) -> bool:
    return all(interval_distance(float(properties[name]), target) == 0.0 for name, target in targets.items())


def score_candidate(
    smiles: str,
    predicted: dict[str, float],
    targets: dict[str, TargetRange],
    cfg: GenerationConfig,
    motif_weights: dict[str, float] | None = None,
) -> float:
    miss = 0.0
    for name, target in targets.items():
        if name not in predicted:
            continue
        miss += target.weight * interval_distance(float(predicted[name]), target)

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    complexity_penalty = 0.03 * max(0, mol.GetNumAtoms() - 12)
    ring_penalty = 0.2 * max(0, ring_rank(mol) - cfg.max_ring_rank)
    motif_bonus = getattr(cfg, "motif_match_weight", 0.0) * motif_match_score(
        smiles,
        getattr(cfg, "target_motifs", ()),
        motif_weights,
    )
    return -(miss + complexity_penalty + ring_penalty) + motif_bonus


def try_canonical(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return canonical_valid_smiles(Chem.MolToSmiles(mol, canonical=True))


def try_constrained_canonical(mol: Chem.Mol, cfg: GenerationConfig) -> str | None:
    canonical = try_canonical(mol)
    if (
        canonical
        and graph_constraints_ok(
            canonical,
            cfg.max_atoms,
            cfg.max_ring_rank,
            getattr(cfg, "allowed_heavy_atoms", None),
            cfg=cfg,
        )
    ):
        return canonical
    return None


def allowed_double_bond(mol: Chem.Mol, atom_idx: int, other_symbol: str) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    pair = {atom.GetSymbol(), other_symbol}
    return "F" not in pair and pair in ({"C"}, {"C", "O"})


def expand_by_atom(smiles: str, cfg: GenerationConfig) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() >= _max_heavy_atoms(cfg):
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
                canonical = try_constrained_canonical(rw.GetMol(), cfg)
                if canonical:
                    outputs.add(canonical)
    return outputs


def expand_by_atom_substitution(smiles: str, cfg: GenerationConfig) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    outputs: set[str] = set()
    for atom_idx in range(mol.GetNumAtoms()):
        current = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        for symbol in cfg.atom_choices:
            if symbol == current:
                continue
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
            canonical = try_constrained_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
    return outputs


def expand_by_bond_insertion(smiles: str, cfg: GenerationConfig) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() >= _max_heavy_atoms(cfg):
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
            canonical = try_constrained_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
    return outputs


def expand_by_bond_edit(smiles: str, cfg: GenerationConfig) -> set[str]:
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
        canonical = try_constrained_canonical(rw.GetMol(), cfg)
        if canonical:
            outputs.add(canonical)
    return outputs


def expand_by_terminal_atom_deletion(smiles: str, cfg: GenerationConfig) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() <= _min_heavy_atoms(cfg):
        return set()

    outputs: set[str] = set()
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 1:
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(atom.GetIdx())
        canonical = try_constrained_canonical(rw.GetMol(), cfg)
        if canonical:
            outputs.add(canonical)
    return outputs


def expand_by_ring_closure(smiles: str, cfg: GenerationConfig) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 4:
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
            allowed_ring_sizes = getattr(cfg, "allowed_ring_sizes", None)
            if allowed_ring_sizes is not None and ring_size not in set(allowed_ring_sizes):
                continue
            rw = Chem.RWMol(mol)
            rw.AddBond(i, j, SINGLE)
            canonical = try_constrained_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
    return outputs


def expand_by_fragment(smiles: str, cfg: GenerationConfig) -> set[str]:
    fragment_smiles = getattr(cfg, "fragment_smiles", ())
    if not fragment_smiles:
        return set()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    outputs: set[str] = set()
    max_outputs = getattr(cfg, "max_fragment_expansions_per_molecule", 10_000)
    for frag_smiles in fragment_smiles:
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            continue
        if mol.GetNumAtoms() + frag.GetNumAtoms() > cfg.max_atoms:
            continue
        combined = Chem.CombineMols(mol, frag)
        offset = mol.GetNumAtoms()
        for atom_idx in range(mol.GetNumAtoms()):
            rw = Chem.RWMol(combined)
            rw.AddBond(atom_idx, offset, SINGLE)
            canonical = try_constrained_canonical(rw.GetMol(), cfg)
            if canonical:
                outputs.add(canonical)
            if len(outputs) >= max_outputs:
                return outputs
    return outputs


def expand_molecule(smiles: str, cfg: GenerationConfig) -> set[str]:
    enabled = set(
        getattr(
            cfg,
            "enabled_graph_edits",
            ("append_atom", "ring_closure", "fragment"),
        )
    )
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


class BeamSearchGenerator:
    def __init__(
        self,
        cfg: GenerationConfig = GENERATION_CONFIG,
        predictor: PropertyPredictor | None = None,
    ):
        self.cfg = cfg
        self.predictor = predictor or PropertyPredictor()

    def _make_candidates(self, smiles_list: list[str]) -> list[Candidate]:
        unique = sorted(
            {
                s
                for s in smiles_list
                if graph_constraints_ok(
                    s,
                    self.cfg.max_atoms,
                    self.cfg.max_ring_rank,
                    getattr(self.cfg, "allowed_heavy_atoms", None),
                    cfg=self.cfg,
                )
            }
        )
        predictions = self.predictor.predict(unique)
        candidates: list[Candidate] = []
        for smiles, predicted in zip(unique, predictions, strict=True):
            rdkit_props = compute_properties(smiles) or {}
            pred_score = score_candidate(smiles, predicted, self.cfg.targets, self.cfg)
            rdkit_score = score_candidate(smiles, rdkit_props, self.cfg.targets, self.cfg)
            verified_hit = satisfies_targets(rdkit_props, self.cfg.targets)
            score = 0.7 * pred_score + 0.3 * rdkit_score + (2.0 if verified_hit else 0.0)
            candidates.append(
                Candidate(
                    smiles=smiles,
                    score=float(score),
                    predicted=predicted,
                    rdkit=rdkit_props,
                    satisfied=verified_hit,
                    motif_match_score=motif_match_score(
                        smiles,
                        getattr(self.cfg, "target_motifs", ()),
                    ),
                    matched_motifs=matched_motif_names(
                        smiles,
                        getattr(self.cfg, "target_motifs", ()),
                    ),
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def search(self) -> list[Candidate]:
        seed_smiles = [canonical_valid_smiles(s) for s in self.cfg.seed_smiles]
        frontier = self._make_candidates([s for s in seed_smiles if s])
        archive: dict[str, Candidate] = {candidate.smiles: candidate for candidate in frontier}

        for _ in range(self.cfg.max_steps):
            expanded: set[str] = set()
            for candidate in frontier[: self.cfg.beam_width]:
                expanded.update(expand_molecule(candidate.smiles, self.cfg))
            max_expanded = getattr(self.cfg, "max_expanded_per_step", None)
            if max_expanded is not None and len(expanded) > max_expanded:
                expanded = set(sorted(expanded)[:max_expanded])
            new_candidates = self._make_candidates(list(expanded))
            for candidate in new_candidates:
                existing = archive.get(candidate.smiles)
                if existing is None or candidate.score > existing.score:
                    archive[candidate.smiles] = candidate
            frontier = sorted(archive.values(), key=lambda item: item.score, reverse=True)[
                : self.cfg.beam_width
            ]

        final = sorted(
            archive.values(),
            key=lambda item: (item.satisfied, item.score),
            reverse=True,
        )
        return final[: self.cfg.top_k]


def candidate_rows(candidates: list[Candidate]) -> list[dict[str, float | str | bool | int]]:
    rows: list[dict[str, float | str | bool | int]] = []
    for rank, candidate in enumerate(candidates, start=1):
        row: dict[str, float | str | bool | int] = {
            "rank": rank,
            "smiles": candidate.smiles,
            "score": candidate.score,
            "satisfied": candidate.satisfied,
        }
        row.update({f"pred_{name}": candidate.predicted.get(name, np.nan) for name in PROPERTY_NAMES})
        row.update({f"rdkit_{name}": candidate.rdkit.get(name, np.nan) for name in PROPERTY_NAMES})
        rows.append(row)
    return rows

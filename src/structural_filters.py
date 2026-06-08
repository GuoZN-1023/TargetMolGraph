"""Structural stability and synthesis filters for generated molecules."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .validate_rdkit import ring_rank


FILTER_SMARTS: dict[str, str] = {
    "acid_fluoride": "[CX3](=O)[F]",
    "ketene": "[C]=[C]=[O]",
    "allene_cumulene": "[C]=[C]=[C]",
}


@dataclass(frozen=True)
class StructuralFilterResult:
    smiles: str
    passed: bool
    failures: tuple[str, ...]
    synthesis_proxy: float


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def synthesis_accessibility_proxy(mol: Chem.Mol) -> float:
    """Small, local proxy for synthetic difficulty.

    This is not a full Ertl SA score. It is a conservative penalty for
    complexity signals available directly from RDKit without external data.
    """

    heavy_atoms = float(Descriptors.HeavyAtomCount(mol))
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    bridgeheads = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    spiro = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    fluorines = float(sum(atom.GetSymbol() == "F" for atom in mol.GetAtoms()))
    chiral_centers = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
    return (
        0.15 * heavy_atoms
        + 0.50 * ring_count
        + 1.50 * aromatic_rings
        + 0.75 * bridgeheads
        + 0.75 * spiro
        + 0.20 * max(0.0, fluorines - 3.0)
        + 0.20 * chiral_centers
    )


def evaluate_structural_filters(smiles: str, cfg=None) -> StructuralFilterResult:
    failures: list[str] = []
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return StructuralFilterResult(str(smiles), False, ("invalid_or_disconnected",), float("inf"))

    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        failures.append("formal_charge")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        failures.append("radical")
    if any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in mol.GetBonds()):
        failures.append("triple_bond")
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        failures.append("aromatic_system")
    if (
        getattr(cfg, "forbid_bridged_rings", False)
        and rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0
    ):
        failures.append("bridged_ring")

    allowed_ring_sizes = set(getattr(cfg, "allowed_ring_sizes", ())) if cfg is not None else set()
    if allowed_ring_sizes:
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) not in allowed_ring_sizes:
                failures.append("disallowed_ring_size")
                break

    max_ring_rank = getattr(cfg, "max_ring_rank", None)
    if max_ring_rank is not None and ring_rank(mol) > int(max_ring_rank):
        failures.append("too_many_rings")

    smarts_items: list[tuple[str, str]] = list(FILTER_SMARTS.items())
    smarts_items.extend(
        (str(name), str(smarts))
        for name, smarts in getattr(cfg, "forbidden_smarts", ())
    )
    for name, smarts in smarts_items:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            failures.append(name)

    synthesis_proxy = synthesis_accessibility_proxy(mol)
    max_synthesis_proxy = getattr(cfg, "max_synthesis_proxy", None)
    if max_synthesis_proxy is not None and synthesis_proxy > float(max_synthesis_proxy):
        failures.append("high_synthesis_proxy")

    unique_failures = tuple(sorted(set(failures)))
    return StructuralFilterResult(
        Chem.MolToSmiles(mol, canonical=True),
        not unique_failures,
        unique_failures,
        float(synthesis_proxy),
    )


def structural_filters_ok(smiles: str, cfg=None) -> bool:
    return evaluate_structural_filters(smiles, cfg).passed

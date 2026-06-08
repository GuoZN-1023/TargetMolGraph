"""Central project configuration.

Change atom types, node features, bond features, property names, or target
ranges here to affect dataset building, model training, and generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"


PROPERTY_NAMES = [
    "MolLogP",
    "TPSA",
    "QED",
    "MolWt",
    "RingCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
]


ELECTROLYTE_TARGETS = [
    "Es-Ea (eV)",
    "LUMO_sol (eV)",
    "HOMO_sol (eV)",
    "Dielectric constant of solvents",
]


BASE_RDKIT_DESCRIPTOR_NAMES = [
    "MolLogP",
    "MolMR",
    "TPSA",
    "QED",
    "MolWt",
    "ExactMolWt",
    "RingCount",
    "NumAliphaticRings",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "FractionCSP3",
    "NumValenceElectrons",
    "LabuteASA",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
]


EXTRA_RDKIT_DESCRIPTOR_NAMES = [
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
    "NumBridgeheadAtoms",
    "NumSpiroAtoms",
    "Ipc",
    "PEOE_VSA1",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "SMR_VSA1",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SMR_VSA10",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "EState_VSA1",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "EState_VSA10",
    "EState_VSA11",
    "VSA_EState1",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "VSA_EState10",
]


RDKIT_DESCRIPTOR_NAMES = BASE_RDKIT_DESCRIPTOR_NAMES
EXPANDED_RDKIT_DESCRIPTOR_NAMES = BASE_RDKIT_DESCRIPTOR_NAMES + EXTRA_RDKIT_DESCRIPTOR_NAMES


MORGAN_FINGERPRINT_BITS = 256


@dataclass(frozen=True)
class FeatureConfig:
    atom_types: tuple[str, ...] = (
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
    )
    hybridizations: tuple[str, ...] = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
    include_atomic_number: bool = True
    include_degree: bool = True
    include_total_hs: bool = True
    include_formal_charge: bool = True
    include_aromatic: bool = True
    include_ring: bool = True
    include_hybridization: bool = True
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


@dataclass(frozen=True)
class TargetRange:
    lower: float | None = None
    upper: float | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class GenerationConfig:
    targets: dict[str, TargetRange] = field(
        default_factory=lambda: {
            "MolLogP": TargetRange(1.0, 2.5, 1.0),
            "TPSA": TargetRange(40.0, 90.0, 0.03),
            "QED": TargetRange(0.60, None, 3.0),
            "MolWt": TargetRange(None, 350.0, 0.01),
            "RingCount": TargetRange(None, 1.0, 0.5),
        }
    )
    allowed_heavy_atoms: tuple[str, ...] = ("C", "O", "F")
    atom_choices: tuple[str, ...] = ("C", "O", "F")
    seed_smiles: tuple[str, ...] = (
        "COC",
        "CCOC",
        "CCOCC",
        "COCOC",
        "COC(=O)OC",
    )
    max_atoms: int = 11
    max_heavy_atoms: int = 11
    min_heavy_atoms: int = 3
    max_mol_weight: float = 350.0
    require_oxygen: bool = True
    forbid_oxygen_hydrogen: bool = True
    forbidden_bond_pairs: tuple[tuple[str, str], ...] = (("O", "O"), ("O", "F"))
    forbidden_smarts: tuple[tuple[str, str], ...] = (
        ("allene_cumulene", "[C]=[C]=[C]"),
        ("ketene", "[C]=[C]=[O]"),
    )
    allowed_ring_sizes: tuple[int, ...] = (5, 6, 7)
    forbid_bridged_rings: bool = True
    reject_triple_bond_in_ring: bool = True
    max_ring_rank: int = 1
    enabled_graph_edits: tuple[str, ...] = (
        "append_atom",
        "substitute_atom",
        "insert_atom_into_bond",
        "edit_bond",
        "delete_terminal_atom",
        "ring_closure",
    )
    target_motifs: tuple[str, ...] = ("ether", "carbonyl", "fluoroalkyl")
    motif_match_weight: float = 0.20
    beam_width: int = 64
    max_steps: int = 5
    top_k: int = 20


@dataclass(frozen=True)
class ElectrolyteGenerationConfig:
    """Target ranges for electrolyte solvent candidate generation.

    The defaults favor molecular electrolyte solvents with a reasonable
    frontier-orbital window, moderate-to-high dielectric constant, and compact
    molecular size for practical liquid electrolyte screening.
    """

    targets: dict[str, TargetRange] = field(
        default_factory=lambda: {
            "Es-Ea (eV)": TargetRange(0.25, None, 2.5),
            "LUMO_sol (eV)": TargetRange(7.5, None, 1.0),
            "HOMO_sol (eV)": TargetRange(None, -7.5, 1.2),
            "Dielectric constant of solvents": TargetRange(None, 10.0, 0.15),
            "MolWt": TargetRange(None, 320.0, 0.006),
            "NumHDonors": TargetRange(None, 1.0, 0.5),
            "RingCount": TargetRange(None, 1.0, 0.5),
        }
    )
    allowed_heavy_atoms: tuple[str, ...] = ("C", "O", "F")
    atom_choices: tuple[str, ...] = ("C", "O", "F")
    seed_smiles: tuple[str, ...] = (
        "CCOC(C)(OC)OC",
        "COC",
        "CCOC",
        "CCOCC",
        "COCOC",
        "CCOC(C)OC",
        "COC(C)(OC)OC",
        "COC(OC)OC",
        "COC(=O)OC",
        "CCOC(=O)OC",
        "CCOC(=O)OCC",
        "COC(=O)C(F)(F)F",
        "CCOC(F)(F)F",
        "COC(F)(F)F",
        "FC(F)(F)COC",
    )
    fragment_smiles: tuple[str, ...] = (
        "OC",
        "OCC",
        "C(F)(F)F",
        "C(=O)OC",
        "C(=O)OCC",
        "COC",
    )
    max_atoms: int = 11
    max_heavy_atoms: int = 11
    min_heavy_atoms: int = 3
    max_mol_weight: float = 320.0
    require_oxygen: bool = True
    forbid_oxygen_hydrogen: bool = True
    forbidden_bond_pairs: tuple[tuple[str, str], ...] = (("O", "O"), ("O", "F"))
    forbidden_smarts: tuple[tuple[str, str], ...] = (
        ("allene_cumulene", "[C]=[C]=[C]"),
        ("ketene", "[C]=[C]=[O]"),
    )
    allowed_ring_sizes: tuple[int, ...] = (5, 6, 7)
    forbid_bridged_rings: bool = True
    reject_triple_bond_in_ring: bool = True
    max_ring_rank: int = 1
    enabled_graph_edits: tuple[str, ...] = (
        "append_atom",
        "substitute_atom",
        "insert_atom_into_bond",
        "edit_bond",
        "delete_terminal_atom",
        "ring_closure",
    )
    target_motifs: tuple[str, ...] = (
        "ether",
        "carbonyl",
        "ester",
        "carbonate",
        "acetal",
        "fluoroalkyl",
        "trifluoromethyl",
        "lactone",
        "beta_lactone",
        "gamma_lactone",
        "delta_lactone",
        "cyclic_carbonate",
        "oxetane",
        "thf",
        "thp",
        "oxepane",
        "dioxolane_1_3",
        "dioxane_1_3",
        "dioxane_1_4",
        "glyme_chain",
    )
    motif_match_weight: float = 0.25
    learned_substructure_match_weight: float = 0.18
    similarity_match_weight: float = 0.35
    applicability_min_similarity: float = 0.08
    applicability_penalty_weight: float = 0.45
    enforce_structural_filters: bool = True
    structural_filter_penalty: float = 5.0
    max_synthesis_proxy: float = 4.0
    final_diversity_max_similarity: float = 0.82
    beam_diversity_max_similarity: float = 0.72
    mc_dropout_enabled: bool = True
    mc_dropout_samples: int = 8
    prediction_noise_scale: float = 0.25
    prediction_noise_samples: int = 16
    prediction_noise_seed: int = 42
    robust_score_std_penalty: float = 0.5
    robust_min_hit_probability: float = 0.50
    prediction_batch_size: int = 512
    max_runtime_seconds: int = 0
    progress_enabled: bool = True
    beam_width: int = 128
    max_steps: int = 10
    top_k: int = 100
    max_fragment_expansions_per_molecule: int = 80
    max_expanded_per_step: int = 5000


@dataclass(frozen=True)
class SmallMoleculeGenerationConfig:
    """Strict graph-search settings for small C/H/O/F molecule generation.

    Hydrogens are represented implicitly by RDKit, so graph edit actions only
    add and connect heavy atoms from C/O/F.
    """

    targets: dict[str, TargetRange] = field(
        default_factory=lambda: {
            "Es-Ea (eV)": TargetRange(0.25, None, 2.5),
            "LUMO_sol (eV)": TargetRange(7.5, None, 1.0),
            "HOMO_sol (eV)": TargetRange(None, -7.5, 1.2),
            "Dielectric constant of solvents": TargetRange(None, 9.0, 0.2),
        }
    )
    allowed_heavy_atoms: tuple[str, ...] = ("C", "O", "F")
    atom_choices: tuple[str, ...] = ("C", "O", "F")
    seed_smiles: tuple[str, ...] = (
        "COC",
        "CCOC",
        "CCOCC",
        "COCOC",
        "COC(=O)OC",
        "COC(F)(F)F",
    )
    fragment_smiles: tuple[str, ...] = (
        "O",
        "C",
        "CO",
        "OC",
        "COC",
        "C=O",
        "C(F)(F)F",
        "OC(F)(F)F",
        "OCO",
    )
    max_mol_wt: float = 200.0
    max_mol_weight: float = 200.0
    max_heavy_atoms: int = 11
    min_heavy_atoms: int = 3
    require_oxygen: bool = True
    forbid_oxygen_hydrogen: bool = True
    forbidden_bond_pairs: tuple[tuple[str, str], ...] = (("O", "O"), ("O", "F"))
    forbidden_smarts: tuple[tuple[str, str], ...] = (
        ("allene_cumulene", "[C]=[C]=[C]"),
        ("ketene", "[C]=[C]=[O]"),
    )
    allowed_ring_sizes: tuple[int, ...] = (5, 6, 7)
    forbid_bridged_rings: bool = True
    reject_triple_bond_in_ring: bool = True
    max_ring_rank: int = 1
    enabled_graph_edits: tuple[str, ...] = (
        "append_atom",
        "substitute_atom",
        "insert_atom_into_bond",
        "edit_bond",
        "delete_terminal_atom",
        "ring_closure",
    )
    prediction_noise_scale: float = 0.25
    prediction_noise_seed: int = 42
    beam_width: int = 64
    max_steps: int = 5
    top_k: int = 50
    max_fragment_expansions_per_molecule: int = 300
    max_expanded_per_step: int = 3000


FEATURE_CONFIG = FeatureConfig()
GENERATION_CONFIG = GenerationConfig()
ELECTROLYTE_GENERATION_CONFIG = ElectrolyteGenerationConfig()
SMALL_MOLECULE_GENERATION_CONFIG = SmallMoleculeGenerationConfig()

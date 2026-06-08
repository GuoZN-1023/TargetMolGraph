import torch
import pandas as pd
from rdkit import Chem

from src.compute_properties import compute_properties
from src.config import (
    ELECTROLYTE_GENERATION_CONFIG,
    GENERATION_CONFIG,
    SMALL_MOLECULE_GENERATION_CONFIG,
)
from src.generate_small_molecules import (
    SmallMoleculeGraphGenerator,
    canonical_small_smiles,
    expand_molecule,
    graph_constraints_ok,
)
from src.mol_to_graph import mol_to_graph
from src.motif_matching import (
    MOTIF_LIBRARY,
    learn_motif_weights_from_table,
    motif_match_score,
    motif_occurrences,
)
from src.structural_filters import evaluate_structural_filters
from src.train_gnn import split_dataset_three_way


def test_compute_properties_and_graph():
    props = compute_properties("CCO")
    assert props is not None
    assert props["MolWt"] > 40
    graph = mol_to_graph("CCO", list(props.values()))
    assert graph["x"].shape[0] == 3
    assert graph["edge_index"].shape[0] == 2


def test_multitarget_stratified_split_is_shared_across_targets():
    stratify_rows = [
        [float(idx), float(idx % 7), float(39 - idx), float((idx * 3) % 11)]
        for idx in range(40)
    ]
    graphs_for_first_target = [
        {
            "id": f"ep-{idx}",
            "y": torch.tensor([row[0]], dtype=torch.float32),
            "stratify_targets": row,
            "cluster": idx % 4,
        }
        for idx, row in enumerate(stratify_rows)
    ]
    graphs_for_second_target = [
        {
            "id": f"ep-{idx}",
            "y": torch.tensor([row[1]], dtype=torch.float32),
            "stratify_targets": row,
            "cluster": idx % 4,
        }
        for idx, row in enumerate(stratify_rows)
    ]

    first_splits = split_dataset_three_way(
        graphs_for_first_target,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        strategy="multitarget_stratified",
    )
    second_splits = split_dataset_three_way(
        graphs_for_second_target,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        strategy="multitarget_stratified",
    )

    assert [len(split) for split in first_splits] == [32, 4, 4]
    assert [
        {graph["id"] for graph in split}
        for split in first_splits
    ] == [
        {graph["id"] for graph in split}
        for split in second_splits
    ]


def test_small_molecule_constraints_reject_disallowed_atoms_and_large_graphs():
    assert canonical_small_smiles("COC") == "COC"
    assert canonical_small_smiles("CCO") is None
    assert canonical_small_smiles("CO") is None
    assert canonical_small_smiles("COO") is None
    assert canonical_small_smiles("COF") is None
    assert canonical_small_smiles("C1CC2CC1CO2") is None
    assert canonical_small_smiles("C=C=C") is None
    assert canonical_small_smiles("C=C=O") is None
    assert canonical_small_smiles("CCN") is None
    assert canonical_small_smiles("CCCl") is None
    assert canonical_small_smiles("CCCCCCCCCCCC") is None
    assert not graph_constraints_ok("CCCCCCCCCCCC")


def test_structural_filters_block_unstable_motifs():
    assert evaluate_structural_filters("COC", ELECTROLYTE_GENERATION_CONFIG).passed
    acid_fluoride = evaluate_structural_filters("COC(=O)F", ELECTROLYTE_GENERATION_CONFIG)
    assert not acid_fluoride.passed
    assert "acid_fluoride" in acid_fluoride.failures
    bridged = evaluate_structural_filters("C1CC2CC1CO2", ELECTROLYTE_GENERATION_CONFIG)
    assert not bridged.passed
    assert "bridged_ring" in bridged.failures
    ketene = evaluate_structural_filters("C=C=O", ELECTROLYTE_GENERATION_CONFIG)
    assert not ketene.passed
    assert "ketene" in ketene.failures
    allene = evaluate_structural_filters("C=C=C", ELECTROLYTE_GENERATION_CONFIG)
    assert not allene.passed
    assert "allene_cumulene" in allene.failures


def test_small_molecule_electrolyte_target_thresholds():
    targets = SMALL_MOLECULE_GENERATION_CONFIG.targets
    assert targets["Es-Ea (eV)"].lower == 0.25
    assert targets["LUMO_sol (eV)"].lower == 7.5
    assert targets["HOMO_sol (eV)"].upper == -7.5
    assert targets["Dielectric constant of solvents"].upper == 9.0


def test_electrolyte_generation_is_limited_to_chof_heavy_atoms():
    blocked = {"N", "P", "Cl", "S"}
    for cfg in (GENERATION_CONFIG, ELECTROLYTE_GENERATION_CONFIG):
        allowed = set(cfg.allowed_heavy_atoms)
        assert allowed == {"C", "O", "F"}
        assert set(cfg.atom_choices) <= allowed
        assert all(atom not in cfg.atom_choices for atom in blocked)
        for smiles in (*cfg.seed_smiles, *getattr(cfg, "fragment_smiles", ())):
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            assert {atom.GetSymbol() for atom in mol.GetAtoms()} <= allowed


def test_motif_matching_scores_electrolyte_functional_groups():
    motifs = ELECTROLYTE_GENERATION_CONFIG.target_motifs
    assert motif_match_score("CCOCC", motifs) > 0.0
    assert motif_match_score("CCOC(=O)OC", motifs) > motif_match_score("CCOCC", motifs)
    occurrences = motif_occurrences("COC(=O)C(F)(F)F", motifs)
    assert occurrences["carbonyl"]
    assert occurrences["fluoroalkyl"]


def test_motif_weights_can_be_learned_from_target_enrichment():
    table = pd.DataFrame(
        [
            {
                "canonical_smiles": "COC(=O)OC",
                "Es-Ea (eV)": 0.5,
                "LUMO_sol (eV)": 8.0,
                "HOMO_sol (eV)": -8.0,
                "Dielectric constant of solvents": 5.0,
                "rdkit_MolWt": 104.0,
                "rdkit_NumHDonors": 0.0,
                "rdkit_RingCount": 0.0,
            },
            {
                "canonical_smiles": "CCOC(=O)OC",
                "Es-Ea (eV)": 0.4,
                "LUMO_sol (eV)": 7.8,
                "HOMO_sol (eV)": -7.8,
                "Dielectric constant of solvents": 6.0,
                "rdkit_MolWt": 118.0,
                "rdkit_NumHDonors": 0.0,
                "rdkit_RingCount": 0.0,
            },
            {
                "canonical_smiles": "CCC",
                "Es-Ea (eV)": 0.0,
                "LUMO_sol (eV)": 6.0,
                "HOMO_sol (eV)": -6.0,
                "Dielectric constant of solvents": 30.0,
                "rdkit_MolWt": 44.0,
                "rdkit_NumHDonors": 0.0,
                "rdkit_RingCount": 0.0,
            },
            {
                "canonical_smiles": "CCCC",
                "Es-Ea (eV)": 0.0,
                "LUMO_sol (eV)": 6.1,
                "HOMO_sol (eV)": -6.1,
                "Dielectric constant of solvents": 25.0,
                "rdkit_MolWt": 58.0,
                "rdkit_NumHDonors": 0.0,
                "rdkit_RingCount": 0.0,
            },
        ]
    )
    learned = learn_motif_weights_from_table(
        table,
        ("carbonyl",),
        ELECTROLYTE_GENERATION_CONFIG.targets,
    )
    assert learned["carbonyl"].support_good > learned["carbonyl"].support_bad
    assert learned["carbonyl"].weight > MOTIF_LIBRARY["carbonyl"].weight


def test_small_molecule_expansion_keeps_cof_hard_constraints():
    expanded = expand_molecule("CO", SMALL_MOLECULE_GENERATION_CONFIG)
    assert expanded
    for smiles in expanded:
        assert canonical_small_smiles(smiles) == smiles
        props = compute_properties(smiles)
        assert props is not None
        assert props["MolWt"] < SMALL_MOLECULE_GENERATION_CONFIG.max_mol_wt


def test_small_molecule_generator_runs_tiny_search():
    class FakeElectrolytePredictor:
        def predict(self, smiles_list):
            rows = []
            for smiles in smiles_list:
                props = compute_properties(smiles)
                assert props is not None
                rows.append(
                    {
                        **props,
                        "Es-Ea (eV)": 0.3,
                        "LUMO_sol (eV)": 7.6,
                        "HOMO_sol (eV)": -7.6,
                        "Dielectric constant of solvents": 8.5,
                    }
                )
            return rows

    cfg = SMALL_MOLECULE_GENERATION_CONFIG.__class__(
        targets=SMALL_MOLECULE_GENERATION_CONFIG.targets,
        allowed_heavy_atoms=SMALL_MOLECULE_GENERATION_CONFIG.allowed_heavy_atoms,
        atom_choices=SMALL_MOLECULE_GENERATION_CONFIG.atom_choices,
        seed_smiles=("COC", "CCOC"),
        fragment_smiles=("COC", "C"),
        max_mol_wt=SMALL_MOLECULE_GENERATION_CONFIG.max_mol_wt,
        max_heavy_atoms=5,
        max_ring_rank=SMALL_MOLECULE_GENERATION_CONFIG.max_ring_rank,
        beam_width=4,
        max_steps=1,
        top_k=5,
        max_fragment_expansions_per_molecule=20,
        max_expanded_per_step=100,
    )
    candidates = SmallMoleculeGraphGenerator(cfg=cfg, predictor=FakeElectrolytePredictor()).search()
    assert candidates
    assert len(candidates) <= 5
    assert all(canonical_small_smiles(candidate.smiles, cfg) for candidate in candidates)
    assert all(candidate.satisfied for candidate in candidates)

"""Generate electrolyte solvent candidates with target-oriented graph search."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from dataclasses import fields, replace
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from .beam_search import (
    Candidate,
    PropertyPredictor,
    expand_molecule,
    graph_constraints_ok,
    interval_distance,
    score_candidate,
)
from .compute_properties import compute_morgan_fingerprint, compute_rdkit_descriptors
from .motif_matching import (
    FingerprintSimilarityScorer,
    learned_substructure_score,
    load_or_learn_motif_profile,
    matched_learned_substructure_names,
    matched_motif_names,
    motif_match_score,
    reference_smiles_from_table,
)
from .structural_filters import evaluate_structural_filters
from .generation_io import (
    config_to_jsonable,
    copy_file,
    setup_run_logger,
    timestamped_run_dir,
    write_json,
    write_table,
)
from .config import (
    ELECTROLYTE_GENERATION_CONFIG,
    ELECTROLYTE_TARGETS,
    GENERATED_DIR,
    MODELS_DIR,
    RDKIT_DESCRIPTOR_NAMES,
    RESULTS_DIR,
)
from .electrolyte_data import DEFAULT_DESCRIPTOR_CSV
from .train_electrolyte_gnns import slugify


GENERATION_MODEL_DIR = MODELS_DIR / "electrolyte_generation_ready"
GENERATION_RUNS_DIR = RESULTS_DIR / "electrolyte_generation_runs"
DEFAULT_FINAL_CSV_NAME = "final_candidates.csv"
DEFAULT_ALL_GENERATED_CSV_NAME = "all_generated_molecules.csv"


def load_config_overrides(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("YAML config override requires PyYAML to be installed.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config override must be a YAML mapping: {path}")
    block = data.get("electrolyte_generation", data)
    if not isinstance(block, dict):
        raise ValueError("electrolyte_generation config block must be a mapping.")
    allowed = {field.name for field in fields(type(ELECTROLYTE_GENERATION_CONFIG))}
    return {key: value for key, value in block.items() if key in allowed}


def _is_materialized(path: Path) -> bool:
    stat = path.stat()
    return stat.st_size == 0 or getattr(stat, "st_blocks", 1) > 0


def preflight_generation_inputs(
    model_dir: Path = GENERATION_MODEL_DIR,
    descriptor_csv: Path = DEFAULT_DESCRIPTOR_CSV,
) -> None:
    """Fail fast when generation-critical data or checkpoints are unavailable."""

    problems: list[str] = []
    if not descriptor_csv.exists():
        problems.append(f"missing descriptor table: {descriptor_csv}")
    else:
        if not _is_materialized(descriptor_csv):
            problems.append(f"descriptor table is cloud-placeholder/unmaterialized: {descriptor_csv}")
        try:
            table = pd.read_csv(descriptor_csv, nrows=5)
            if "canonical_smiles" not in table.columns and "SMILES" not in table.columns:
                problems.append(f"descriptor table lacks canonical_smiles or SMILES: {descriptor_csv}")
            missing_targets = [target for target in ELECTROLYTE_TARGETS if target not in table.columns]
            if missing_targets:
                problems.append(f"descriptor table lacks target columns: {missing_targets}")
        except Exception as exc:
            problems.append(f"descriptor table cannot be read: {descriptor_csv} ({type(exc).__name__}: {exc})")

    if not model_dir.exists():
        problems.append(f"missing model directory: {model_dir}")
    for target in ELECTROLYTE_TARGETS:
        path = model_dir / f"{slugify(target)}_gnn.pt"
        if not path.exists():
            problems.append(f"missing checkpoint for {target}: {path}")
            continue
        if not _is_materialized(path):
            problems.append(f"checkpoint is cloud-placeholder/unmaterialized for {target}: {path}")
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            problems.append(f"checkpoint cannot be loaded for {target}: {path} ({type(exc).__name__}: {exc})")
            continue
        required = {"model_state", "metadata", "y_mean", "y_std"}
        missing = sorted(required - set(checkpoint))
        if missing:
            problems.append(f"checkpoint missing keys for {target}: {missing}")

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(f"Generation preflight failed:\n{details}")


class ElectrolytePredictorEnsemble:
    def __init__(
        self,
        model_dir: Path = GENERATION_MODEL_DIR,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger
        if not model_dir.exists():
            model_dir = MODELS_DIR / "electrolyte"
        self.models: dict[str, PropertyPredictor] = {}
        self.load_errors: dict[str, str] = {}
        for target in ELECTROLYTE_TARGETS:
            path = model_dir / f"{slugify(target)}_gnn.pt"
            if path.exists():
                try:
                    self.models[target] = PropertyPredictor(path)
                except Exception as exc:
                    self.load_errors[target] = f"{type(exc).__name__}: {exc}"
        self.fallback = self._load_training_medians()
        for target, error in self.load_errors.items():
            message = f"skipped {target} checkpoint; using fallback values ({error})"
            if self.logger is not None:
                self.logger.warning(message)
            else:
                print(f"Warning: {message}")

    def _load_training_medians(self) -> dict[str, float]:
        if not DEFAULT_DESCRIPTOR_CSV.exists():
            return {}
        df = pd.read_csv(DEFAULT_DESCRIPTOR_CSV)
        medians: dict[str, float] = {}
        for target in ELECTROLYTE_TARGETS:
            if target in df:
                medians[target] = float(df[target].median())
        return medians

    def _initial_rows(self, smiles_list: list[str]) -> list[dict[str, float]]:
        rows = []
        for smiles in smiles_list:
            descriptors = compute_rdkit_descriptors(smiles) or {}
            fingerprint = compute_morgan_fingerprint(smiles) or {}
            row = {
                f"rdkit_{name}": descriptors[name]
                for name in RDKIT_DESCRIPTOR_NAMES
                if name in descriptors
            }
            row.update({name: descriptors[name] for name in RDKIT_DESCRIPTOR_NAMES if name in descriptors})
            row.update(fingerprint)
            rows.append(row)
        for row in rows:
            for target in ELECTROLYTE_TARGETS:
                row[target] = self.fallback.get(target, 0.0)
        return rows

    def predict(self, smiles_list: list[str]) -> list[dict[str, float]]:
        rows = self._initial_rows(smiles_list)
        for _ in range(3):
            for target in ELECTROLYTE_TARGETS:
                if target not in self.models:
                    continue
                preds = self.models[target].predict(smiles_list, auxiliary_values=rows)
                for row, pred in zip(rows, preds, strict=True):
                    row[target] = float(pred[target])
        return rows

    def predict_mc(self, smiles_list: list[str], samples: int = 8) -> list[list[dict[str, float]]]:
        if samples <= 0:
            return []
        sample_rows: list[list[dict[str, float]]] = []
        for _ in range(samples):
            rows = self._initial_rows(smiles_list)
            for _ in range(3):
                for target in ELECTROLYTE_TARGETS:
                    if target not in self.models:
                        continue
                    preds_by_sample = self.models[target].predict_mc(
                        smiles_list,
                        auxiliary_values=rows,
                        samples=1,
                    )
                    preds = preds_by_sample[0] if preds_by_sample else self.models[target].predict(
                        smiles_list,
                        auxiliary_values=rows,
                    )
                    for row, pred in zip(rows, preds, strict=True):
                        row[target] = float(pred[target])
            sample_rows.append(rows)
        return sample_rows


def satisfies_electrolyte_targets(properties: dict[str, float], cfg=None) -> bool:
    active_cfg = cfg or ELECTROLYTE_GENERATION_CONFIG
    for name, target in active_cfg.targets.items():
        lookup_name = name.removeprefix("rdkit_")
        if lookup_name not in properties:
            return False
        if interval_distance(float(properties[lookup_name]), target) > 0.0:
            return False
    return True


def target_miss(properties: dict[str, float], cfg) -> float:
    miss = 0.0
    for name, target in cfg.targets.items():
        lookup_name = name.removeprefix("rdkit_")
        if lookup_name not in properties:
            miss += 10.0 * target.weight
            continue
        miss += target.weight * interval_distance(float(properties[lookup_name]), target)
    return float(miss)


def target_scale(name: str, prediction_mae: dict[str, float]) -> float:
    if name in prediction_mae and prediction_mae[name] > 0:
        return float(prediction_mae[name])
    if name == "MolWt":
        return 50.0
    if name in {"RingCount", "NumHDonors"}:
        return 1.0
    return 1.0


def normalized_target_values(
    properties: dict[str, float],
    cfg,
    prediction_mae: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    violations: dict[str, float] = {}
    margins: dict[str, float] = {}
    for name, target in cfg.targets.items():
        lookup_name = name.removeprefix("rdkit_")
        if lookup_name not in properties:
            violations[name] = 10.0
            margins[name] = -10.0
            continue
        value = float(properties[lookup_name])
        scale = target_scale(lookup_name, prediction_mae)
        raw_violation = interval_distance(value, target)
        violations[name] = raw_violation / scale
        if target.lower is not None and target.upper is not None:
            raw_margin = min(value - target.lower, target.upper - value)
        elif target.lower is not None:
            raw_margin = value - target.lower
        elif target.upper is not None:
            raw_margin = target.upper - value
        else:
            raw_margin = 0.0
        margins[name] = raw_margin / scale
    return violations, margins


def aggregate_target_statistics(
    samples: list[dict[str, float]],
    cfg,
    prediction_mae: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float, float]:
    if not samples:
        return {}, {}, {}, 0.0, 0.0, 0.0
    target_names = list(cfg.targets)
    failures = {name: 0 for name in target_names}
    violation_values = {name: [] for name in target_names}
    margin_values = {name: [] for name in target_names}
    for sample in samples:
        violations, margins = normalized_target_values(sample, cfg, prediction_mae)
        for name in target_names:
            violation_values[name].append(float(violations.get(name, 10.0)))
            margin_values[name].append(float(margins.get(name, -10.0)))
            lookup_name = name.removeprefix("rdkit_")
            if lookup_name not in sample or interval_distance(float(sample[lookup_name]), cfg.targets[name]) > 0.0:
                failures[name] += 1
    n = float(len(samples))
    failure_probs = {name: failures[name] / n for name in target_names}
    mean_violations = {
        name: float(sum(values) / len(values)) if values else 0.0
        for name, values in violation_values.items()
    }
    mean_margins = {
        name: float(sum(values) / len(values)) if values else 0.0
        for name, values in margin_values.items()
    }
    failure_probability_mean = float(sum(failure_probs.values()) / len(failure_probs)) if failure_probs else 0.0
    normalized_violation_mean = float(sum(mean_violations.values()) / len(mean_violations)) if mean_violations else 0.0
    normalized_margin_min = float(min(mean_margins.values())) if mean_margins else 0.0
    return (
        failure_probs,
        mean_violations,
        mean_margins,
        failure_probability_mean,
        normalized_violation_mean,
        normalized_margin_min,
    )


def pareto_rank_candidates(candidates: list[Candidate], max_ranked: int = 1000) -> list[Candidate]:
    """Rank candidates by non-domination across target miss and structural priors."""

    score_sorted = sorted(candidates, key=lambda item: item.score, reverse=True)
    ranked_pool = score_sorted[:max_ranked]
    deferred = score_sorted[max_ranked:]
    remaining = list(ranked_pool)
    ranked: list[Candidate] = []
    rank = 0
    while remaining:
        front: list[Candidate] = []
        for candidate in remaining:
            candidate_values = (
                candidate.target_miss,
                candidate.normalized_violation_mean,
                candidate.failure_probability_mean,
                candidate.uncertainty_score,
                candidate.robust_score_std,
                -candidate.target_hit_probability,
                -candidate.normalized_margin_min,
                -candidate.motif_match_score,
                -candidate.learned_substructure_score,
                -candidate.similarity_score,
            )
            dominated = False
            for other in remaining:
                if other is candidate:
                    continue
                other_values = (
                    other.target_miss,
                    other.normalized_violation_mean,
                    other.failure_probability_mean,
                    other.uncertainty_score,
                    other.robust_score_std,
                    -other.target_hit_probability,
                    -other.normalized_margin_min,
                    -other.motif_match_score,
                    -other.learned_substructure_score,
                    -other.similarity_score,
                )
                no_worse = all(a <= b for a, b in zip(other_values, candidate_values, strict=True))
                better = any(a < b for a, b in zip(other_values, candidate_values, strict=True))
                if no_worse and better:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        for candidate in front:
            candidate.pareto_rank = rank
        ranked.extend(sorted(front, key=lambda item: (not item.satisfied, -item.score)))
        front_ids = {id(candidate) for candidate in front}
        remaining = [candidate for candidate in remaining if id(candidate) not in front_ids]
        rank += 1
    for candidate in deferred:
        candidate.pareto_rank = rank
    ranked.extend(deferred)
    return ranked


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    fp_a = _fingerprint(smiles_a)
    fp_b = _fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def select_diverse_candidates(candidates: list[Candidate], top_k: int, max_similarity: float) -> list[Candidate]:
    selected: list[Candidate] = []
    skipped: list[Candidate] = []
    for candidate in candidates:
        too_similar = any(
            tanimoto_similarity(candidate.smiles, existing.smiles) > max_similarity
            for existing in selected
        )
        if too_similar:
            skipped.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            return selected

    for candidate in skipped:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def numeric_mean(rows: list[dict[str, float]], base: dict[str, float]) -> dict[str, float]:
    output = dict(base)
    if not rows:
        return output
    numeric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, int | float)
    }
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], int | float)
        ]
        if values:
            output[key] = float(sum(values) / len(values))
    return output


class ElectrolyteBeamSearchGenerator:
    def __init__(
        self,
        predictor: ElectrolytePredictorEnsemble | None = None,
        cfg=None,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg or ELECTROLYTE_GENERATION_CONFIG
        self.logger = logger
        self.predictor = predictor or ElectrolytePredictorEnsemble(logger=logger)
        self.prediction_mae = self._load_prediction_mae()
        self.motif_profile = load_or_learn_motif_profile(
            DEFAULT_DESCRIPTOR_CSV,
            self.cfg.target_motifs,
            self.cfg.targets,
        )
        self.similarity_scorer = FingerprintSimilarityScorer(self.motif_profile.reference_smiles)
        self.training_similarity_scorer = FingerprintSimilarityScorer(
            reference_smiles_from_table(DEFAULT_DESCRIPTOR_CSV)
        )
        self.step_summaries: list[dict] = []
        self.started_at = time.monotonic()
        self.stopped_early = False
        self.stop_reason = ""
        self.archive: dict[str, Candidate] = {}

    def _elapsed_seconds(self) -> float:
        return float(time.monotonic() - self.started_at)

    def _log(self, message: str) -> None:
        if getattr(self.cfg, "progress_enabled", True):
            message = f"{message} | elapsed={self._elapsed_seconds():.1f}s"
            if self.logger is not None:
                self.logger.info(message)
            else:
                print(f"[generate_electrolytes] {message}", flush=True)

    def _runtime_exceeded(self) -> bool:
        max_runtime = getattr(self.cfg, "max_runtime_seconds", None)
        if max_runtime is None or float(max_runtime) <= 0.0:
            return False
        if self._elapsed_seconds() < float(max_runtime):
            return False
        self.stopped_early = True
        self.stop_reason = f"max_runtime_seconds={max_runtime} reached"
        return True

    def _load_prediction_mae(self) -> dict[str, float]:
        """Load model MAE for calibrated prediction noise."""

        output: dict[str, float] = {}
        metrics_dir = RESULTS_DIR / "electrolyte_generation"
        for target in ELECTROLYTE_TARGETS:
            path = metrics_dir / f"{slugify(target)}_metrics.csv"
            if not path.exists():
                continue
            try:
                metrics = pd.read_csv(path)
            except Exception:
                continue
            if metrics.empty:
                continue
            values = []
            for column in ("MAE", "test_MAE"):
                if column in metrics and pd.notna(metrics.iloc[0][column]):
                    values.append(float(metrics.iloc[0][column]))
            if values:
                output[target] = max(values)
        return output

    def _prediction_samples(
        self,
        smiles: str,
        prediction: dict[str, float],
        mc_predictions: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        scale = float(getattr(self.cfg, "prediction_noise_scale", 0.0))
        sample_count = int(getattr(self.cfg, "prediction_noise_samples", 1))
        base_samples = [dict(row) for row in mc_predictions] if mc_predictions else []
        desired_samples = max(sample_count, len(base_samples), 1)
        if base_samples:
            base_samples = [
                dict(base_samples[idx % len(base_samples)])
                for idx in range(desired_samples)
            ]
        else:
            base_samples = [dict(prediction) for _ in range(desired_samples)]
        if scale <= 0.0:
            return base_samples
        seed = int(getattr(self.cfg, "prediction_noise_seed", 0))
        noisy_rows = []
        for sample_idx, sample in enumerate(base_samples):
            row = dict(sample)
            for target, mae in self.prediction_mae.items():
                if target not in row:
                    continue
                digest = hashlib.sha256(
                    f"{seed}|{smiles}|{target}|{sample_idx}".encode("utf-8")
                ).hexdigest()
                rng = random.Random(int(digest[:16], 16))
                row[target] = float(row[target]) + rng.gauss(0.0, scale * mae)
            noisy_rows.append(row)
        return noisy_rows

    def _sample_score(
        self,
        smiles: str,
        prediction: dict[str, float],
        learned_score: float,
        similarity_score: float,
        uncertainty_score: float,
        structural_penalty: float,
    ) -> float:
        score = score_candidate(
            smiles,
            prediction,
            self.cfg.targets,
            self.cfg,
            motif_weights=self.motif_profile.motif_weights,
        )
        score += getattr(self.cfg, "learned_substructure_match_weight", 0.0) * learned_score
        score += getattr(self.cfg, "similarity_match_weight", 0.0) * similarity_score
        score -= getattr(self.cfg, "applicability_penalty_weight", 0.0) * uncertainty_score
        score -= structural_penalty
        return float(score)

    def _make_candidates(self, smiles_list: list[str], stage: str = "candidates") -> list[Candidate]:
        if self._runtime_exceeded():
            self._log(f"{stage}: skipped scoring because {self.stop_reason}")
            return []
        unique_all = sorted(
            {
                smiles
                for smiles in smiles_list
                if graph_constraints_ok(
                    smiles,
                    self.cfg.max_atoms,
                    self.cfg.max_ring_rank,
                    getattr(self.cfg, "allowed_heavy_atoms", None),
                    cfg=self.cfg,
                )
            }
        )
        self._log(f"{stage}: {len(unique_all)} unique valid graphs from {len(smiles_list)} inputs")
        structural_by_smiles = {}
        unique = []
        for smiles in unique_all:
            structural_result = evaluate_structural_filters(smiles, self.cfg)
            if (
                getattr(self.cfg, "enforce_structural_filters", True)
                and not structural_result.passed
            ):
                continue
            structural_by_smiles[smiles] = structural_result
            unique.append(smiles)
        if not unique:
            return []
        self._log(f"{stage}: {len(unique)} candidates after structural filters")
        candidates: list[Candidate] = []
        batch_size = max(1, int(getattr(self.cfg, "prediction_batch_size", len(unique))))
        for batch_start in range(0, len(unique), batch_size):
            if self._runtime_exceeded():
                self._log(f"{stage}: stopped after {len(candidates)} scored candidates; {self.stop_reason}")
                break
            batch = unique[batch_start : batch_start + batch_size]
            self._log(
                f"{stage}: scoring {batch_start + 1}-{batch_start + len(batch)} of {len(unique)}"
            )
            predictions = self.predictor.predict(batch)
            mc_by_smiles: dict[str, list[dict[str, float]]] = {smiles: [] for smiles in batch}
            if getattr(self.cfg, "mc_dropout_enabled", False):
                mc_samples = self.predictor.predict_mc(
                    batch,
                    samples=int(getattr(self.cfg, "mc_dropout_samples", 8)),
                )
                for sample_rows in mc_samples:
                    for smiles, sample in zip(batch, sample_rows, strict=True):
                        mc_by_smiles[smiles].append(sample)
            for smiles, predicted in zip(batch, predictions, strict=True):
                structural_result = structural_by_smiles[smiles]
                motif_score = motif_match_score(
                    smiles,
                    self.cfg.target_motifs,
                    self.motif_profile.motif_weights,
                )
                learned_score = learned_substructure_score(
                    smiles,
                    self.motif_profile.learned_substructures,
                )
                similarity_score = self.similarity_scorer.score(smiles)
                training_similarity = self.training_similarity_scorer.score(smiles)
                uncertainty_score = 1.0 - training_similarity
                min_similarity = getattr(self.cfg, "applicability_min_similarity", 0.0)
                if training_similarity < min_similarity:
                    continue
                structural_penalty = (
                    0.0
                    if structural_result.passed
                    else float(getattr(self.cfg, "structural_filter_penalty", 0.0))
                )
                prediction_samples = self._prediction_samples(
                    smiles,
                    predicted,
                    mc_predictions=mc_by_smiles.get(smiles),
                )
                sample_scores = [
                    self._sample_score(
                        smiles,
                        sample,
                        learned_score,
                        similarity_score,
                        uncertainty_score,
                        structural_penalty,
                    )
                    for sample in prediction_samples
                ]
                sample_misses = [target_miss(sample, self.cfg) for sample in prediction_samples]
                sample_hits = [
                    satisfies_electrolyte_targets(sample, self.cfg)
                    for sample in prediction_samples
                ]
                robust_score_mean = float(sum(sample_scores) / len(sample_scores))
                robust_score_std = (
                    float(pd.Series(sample_scores).std(ddof=0)) if len(sample_scores) > 1 else 0.0
                )
                hit_probability = float(sum(sample_hits) / len(sample_hits))
                (
                    failure_probs,
                    normalized_violations,
                    normalized_margins,
                    failure_probability_mean,
                    normalized_violation_mean,
                    normalized_margin_min,
                ) = aggregate_target_statistics(prediction_samples, self.cfg, self.prediction_mae)
                mean_prediction = numeric_mean(prediction_samples, predicted)
                robust_score = robust_score_mean - (
                    float(getattr(self.cfg, "robust_score_std_penalty", 0.0)) * robust_score_std
                )
                candidates.append(
                    Candidate(
                        smiles=smiles,
                        score=robust_score,
                        predicted=mean_prediction,
                        rdkit={
                            k.removeprefix("rdkit_"): v
                            for k, v in predicted.items()
                            if k.startswith("rdkit_")
                        },
                        satisfied=hit_probability >= float(
                            getattr(self.cfg, "robust_min_hit_probability", 0.5)
                        ),
                        motif_match_score=motif_score,
                        matched_motifs=matched_motif_names(smiles, self.cfg.target_motifs),
                        learned_substructure_score=learned_score,
                        matched_learned_substructures=matched_learned_substructure_names(
                            smiles,
                            self.motif_profile.learned_substructures,
                        ),
                        similarity_score=similarity_score,
                        training_similarity=training_similarity,
                        uncertainty_score=uncertainty_score,
                        target_miss=float(sum(sample_misses) / len(sample_misses)),
                        robust_score_mean=robust_score_mean,
                        robust_score_std=robust_score_std,
                        target_hit_probability=hit_probability,
                        structural_filter_passed=structural_result.passed,
                        structural_filter_failures=structural_result.failures,
                        synthesis_proxy=structural_result.synthesis_proxy,
                        failure_probability_mean=failure_probability_mean,
                        normalized_violation_mean=normalized_violation_mean,
                        normalized_margin_min=normalized_margin_min,
                        target_failure_probabilities=failure_probs,
                        target_normalized_violations=normalized_violations,
                        target_normalized_margins=normalized_margins,
                    )
                )
        return pareto_rank_candidates(candidates)

    def search(self) -> list[Candidate]:
        self.started_at = time.monotonic()
        self._log(
            "starting search "
            f"beam_width={self.cfg.beam_width}, max_steps={self.cfg.max_steps}, "
            f"top_k={self.cfg.top_k}, max_runtime_seconds={getattr(self.cfg, 'max_runtime_seconds', None)}"
        )
        frontier = self._make_candidates(list(self.cfg.seed_smiles), stage="seed")
        self.archive = {candidate.smiles: candidate for candidate in frontier}
        try:
            for step_idx in range(self.cfg.max_steps):
                if self._runtime_exceeded():
                    self._log(f"stopping before step {step_idx + 1}; {self.stop_reason}")
                    break
                expanded: set[str] = set()
                parent_count = len(frontier[: self.cfg.beam_width])
                self._log(f"step {step_idx + 1}: expanding {parent_count} parents")
                for candidate in frontier[: self.cfg.beam_width]:
                    expanded.update(expand_molecule(candidate.smiles, self.cfg))
                raw_expanded = len(expanded)
                max_expanded = getattr(self.cfg, "max_expanded_per_step", None)
                if max_expanded is not None and len(expanded) > max_expanded:
                    expanded = set(sorted(expanded)[:max_expanded])
                self._log(
                    f"step {step_idx + 1}: raw_expanded={raw_expanded}, "
                    f"expanded_after_cap={len(expanded)}"
                )
                new_candidates = self._make_candidates(list(expanded), stage=f"step {step_idx + 1}")
                accepted = 0
                improved = 0
                for candidate in new_candidates:
                    existing = self.archive.get(candidate.smiles)
                    if existing is None or candidate.score > existing.score:
                        if existing is None:
                            accepted += 1
                        else:
                            improved += 1
                        self.archive[candidate.smiles] = candidate
                frontier = pareto_rank_candidates(
                    list(self.archive.values()),
                    max_ranked=max(200, self.cfg.beam_width * 8),
                )
                frontier = select_diverse_candidates(
                    frontier,
                    self.cfg.beam_width,
                    getattr(self.cfg, "beam_diversity_max_similarity", 1.0),
                )
                self.step_summaries.append(
                    {
                        "step": len(self.step_summaries) + 1,
                        "parents": parent_count,
                        "raw_expanded": raw_expanded,
                        "expanded_after_cap": len(expanded),
                        "candidate_count": len(new_candidates),
                        "new_archive_entries": accepted,
                        "improved_archive_entries": improved,
                        "archive_size": len(self.archive),
                        "selected_frontier": len(frontier),
                        "full_hit_frontier": sum(1 for candidate in frontier if candidate.satisfied),
                        "elapsed_seconds": self._elapsed_seconds(),
                        "stopped_early": self.stopped_early,
                        "stop_reason": self.stop_reason,
                    }
                )
                self._log(
                    f"step {step_idx + 1}: accepted={accepted}, improved={improved}, "
                    f"archive_size={len(self.archive)}, selected_frontier={len(frontier)}, "
                    f"full_hit_frontier={self.step_summaries[-1]['full_hit_frontier']}"
                )
                if self._runtime_exceeded():
                    self._log(f"stopping after step {step_idx + 1}; {self.stop_reason}")
                    break
        except KeyboardInterrupt:
            self.stopped_early = True
            self.stop_reason = "keyboard_interrupt"
            self._log("received keyboard interrupt; writing best candidates found so far")
        ranked = pareto_rank_candidates(
            list(self.archive.values()),
            max_ranked=max(200, self.cfg.top_k * 10),
        )
        selected = select_diverse_candidates(
            ranked,
            self.cfg.top_k,
            getattr(self.cfg, "final_diversity_max_similarity", 1.0),
        )
        self._log(f"finished search with {len(selected)} selected candidates")
        return selected


def candidate_rows(candidates: list[Candidate]) -> list[dict]:
    rows: list[dict] = []
    for rank, candidate in enumerate(candidates, start=1):
        row = {
            "rank": rank,
            "smiles": candidate.smiles,
            "score": candidate.score,
            "satisfied_predicted_electrolyte_targets": candidate.satisfied,
        }
        for target in ELECTROLYTE_TARGETS:
            row[f"pred_{target}"] = candidate.predicted.get(target)
        for name in RDKIT_DESCRIPTOR_NAMES:
            row[f"rdkit_{name}"] = candidate.predicted.get(name)
        row["motif_match_score"] = candidate.motif_match_score
        row["matched_motifs"] = ",".join(candidate.matched_motifs)
        row["learned_substructure_score"] = candidate.learned_substructure_score
        row["matched_learned_substructures"] = ",".join(candidate.matched_learned_substructures)
        row["similarity_score"] = candidate.similarity_score
        row["training_similarity"] = candidate.training_similarity
        row["uncertainty_score"] = candidate.uncertainty_score
        row["target_miss"] = candidate.target_miss
        row["pareto_rank"] = candidate.pareto_rank
        row["robust_score_mean"] = candidate.robust_score_mean
        row["robust_score_std"] = candidate.robust_score_std
        row["target_hit_probability"] = candidate.target_hit_probability
        row["structural_filter_passed"] = candidate.structural_filter_passed
        row["structural_filter_failures"] = ",".join(candidate.structural_filter_failures)
        row["synthesis_proxy"] = candidate.synthesis_proxy
        row["failure_probability_mean"] = candidate.failure_probability_mean
        row["normalized_violation_mean"] = candidate.normalized_violation_mean
        row["normalized_margin_min"] = candidate.normalized_margin_min
        row["target_failure_probabilities"] = json.dumps(
            candidate.target_failure_probabilities,
            sort_keys=True,
        )
        row["target_normalized_violations"] = json.dumps(
            candidate.target_normalized_violations,
            sort_keys=True,
        )
        row["target_normalized_margins"] = json.dumps(
            candidate.target_normalized_margins,
            sort_keys=True,
        )
        rows.append(row)
    return rows


def _generation_output_paths(
    output_csv: Path | None,
    output_dir: Path | None,
    run_name: str,
) -> dict[str, Path]:
    """Resolve primary run artifacts without running the search."""

    if output_dir is not None:
        run_dir = Path(output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        final_csv = run_dir / (Path(output_csv).name if output_csv is not None else DEFAULT_FINAL_CSV_NAME)
        return {
            "run_dir": run_dir,
            "final_csv": final_csv,
            "all_generated_csv": run_dir / DEFAULT_ALL_GENERATED_CSV_NAME,
            "step_summary": run_dir / "step_summary.csv",
            "resolved_config": run_dir / "resolved_config.json",
            "run_summary": run_dir / "run_summary.json",
            "log": run_dir / "run.log",
        }

    if output_csv is not None:
        final_csv = Path(output_csv).expanduser().resolve()
        run_dir = final_csv.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = final_csv.stem
        return {
            "run_dir": run_dir,
            "final_csv": final_csv,
            "all_generated_csv": final_csv.with_name(f"{stem}_all_generated_molecules.csv"),
            "step_summary": final_csv.with_name(f"{stem}_step_summary.csv"),
            "resolved_config": final_csv.with_name(f"{stem}_resolved_config.json"),
            "run_summary": final_csv.with_name(f"{stem}_run_summary.json"),
            "log": final_csv.with_name(f"{stem}_run.log"),
        }

    run_dir = timestamped_run_dir(GENERATION_RUNS_DIR, run_name)
    return {
        "run_dir": run_dir,
        "final_csv": run_dir / DEFAULT_FINAL_CSV_NAME,
        "all_generated_csv": run_dir / DEFAULT_ALL_GENERATED_CSV_NAME,
        "step_summary": run_dir / "step_summary.csv",
        "resolved_config": run_dir / "resolved_config.json",
        "run_summary": run_dir / "run_summary.json",
        "log": run_dir / "run.log",
    }


def generate(
    output_csv: Path | None = None,
    beam_width: int | None = None,
    max_steps: int | None = None,
    top_k: int | None = None,
    *,
    output_dir: Path | None = None,
    run_name: str = "electrolyte_generation",
    max_runtime_seconds: int | None = None,
    prediction_batch_size: int | None = None,
    disable_mc_dropout: bool = False,
    quiet: bool = False,
    strict_preflight: bool = True,
    config_path: Path | None = None,
    legacy_outputs: bool = True,
    log_level: str = "INFO",
) -> pd.DataFrame:
    paths = _generation_output_paths(output_csv, output_dir, run_name)
    logger = setup_run_logger(
        "targetmolgraph.electrolyte_generation",
        paths["log"],
        level=log_level,
        console=not quiet,
    )

    logger.info("Result directory: %s", paths["run_dir"])
    logger.info("Starting electrolyte generation")
    try:
        logger.info("Stage 1/5 | Resolve configuration")
        overrides = load_config_overrides(config_path) if config_path is not None else {}
        cfg = replace(
            ELECTROLYTE_GENERATION_CONFIG,
            **overrides,
        )
        cli_overrides = {}
        if beam_width is not None:
            cli_overrides["beam_width"] = beam_width
        if max_steps is not None:
            cli_overrides["max_steps"] = max_steps
        if top_k is not None:
            cli_overrides["top_k"] = top_k
        if max_runtime_seconds is not None:
            cli_overrides["max_runtime_seconds"] = max_runtime_seconds
        if prediction_batch_size is not None:
            cli_overrides["prediction_batch_size"] = prediction_batch_size
        if disable_mc_dropout:
            cli_overrides["mc_dropout_enabled"] = False
        if cli_overrides:
            cfg = replace(cfg, **cli_overrides)

        write_json(paths["resolved_config"], config_to_jsonable(cfg))
        logger.info(
            "Resolved run settings: beam_width=%d | max_steps=%d | top_k=%d | mc_dropout=%s",
            int(cfg.beam_width),
            int(cfg.max_steps),
            int(cfg.top_k),
            bool(cfg.mc_dropout_enabled),
        )
        logger.info("Resolved config written: %s", paths["resolved_config"])

        logger.info("Stage 2/5 | Preflight inputs")
        if strict_preflight:
            preflight_generation_inputs()
            logger.info("Preflight passed")
        else:
            logger.warning("Preflight skipped by user request")

        logger.info("Stage 3/5 | Search and score candidates")
        generator = ElectrolyteBeamSearchGenerator(cfg=cfg, logger=logger)
        candidates = generator.search()
        table = pd.DataFrame(candidate_rows(candidates))
        all_generated_candidates = sorted(
            generator.archive.values(),
            key=lambda candidate: (
                not candidate.satisfied,
                candidate.pareto_rank,
                -candidate.score,
                candidate.smiles,
            ),
        )
        all_generated_table = pd.DataFrame(candidate_rows(all_generated_candidates))

        logger.info("Stage 4/5 | Write run artifacts")
        write_table(paths["final_csv"], table)
        write_table(paths["all_generated_csv"], all_generated_table)
        write_table(paths["step_summary"], pd.DataFrame(generator.step_summaries))

        legacy_paths: dict[str, str] = {}
        if legacy_outputs:
            legacy_generated = GENERATED_DIR / "topk_electrolytes.csv"
            legacy_results = RESULTS_DIR / "electrolyte" / "generated_electrolytes.csv"
            for label, destination in (
                ("generated_csv", legacy_generated),
                ("results_csv", legacy_results),
            ):
                try:
                    if destination.resolve() != paths["final_csv"].resolve():
                        copy_file(paths["final_csv"], destination)
                    legacy_paths[label] = str(destination)
                except PermissionError as exc:
                    logger.warning("Could not update legacy output %s: %s", destination, exc)

        logger.info("Stage 5/5 | Write run summary")
        run_summary = {
            "candidate_count": int(len(table)),
            "elapsed_seconds": generator._elapsed_seconds(),
            "stopped_early": generator.stopped_early,
            "stop_reason": generator.stop_reason,
            "run_dir": str(paths["run_dir"]),
            "final_csv": str(paths["final_csv"]),
            "all_generated_csv": str(paths["all_generated_csv"]),
            "step_summary_csv": str(paths["step_summary"]),
            "resolved_config_json": str(paths["resolved_config"]),
            "log_file": str(paths["log"]),
            "all_generated_candidate_count": int(len(all_generated_table)),
            "legacy_outputs": legacy_paths,
        }
        write_json(paths["run_summary"], run_summary)

        table.attrs["run_dir"] = str(paths["run_dir"])
        table.attrs["final_csv"] = str(paths["final_csv"])
        table.attrs["all_generated_csv"] = str(paths["all_generated_csv"])
        table.attrs["run_summary"] = str(paths["run_summary"])
        logger.info("Final candidates written: %s", paths["final_csv"])
        logger.info("All generated molecules written: %s", paths["all_generated_csv"])
        logger.info("Step summary written: %s", paths["step_summary"])
        logger.info("Run summary written: %s", paths["run_summary"])
        logger.info(
            "Generation finished successfully. Final candidates: %d | all generated molecules: %d",
            len(table),
            len(all_generated_table),
        )
    except Exception:
        logger.exception("Electrolyte generation failed")
        raise
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional fixed final CSV path. If omitted, a timestamped run folder "
            f"is created under {GENERATION_RUNS_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional run directory. Writes run.log, final_candidates.csv, step_summary.csv, config, and summary there.",
    )
    parser.add_argument("--run-name", default="electrolyte_generation", help="Name suffix for timestamped run directories.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config override.")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=None,
        help="Stop after this many seconds and write the best candidates found so far.",
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=None,
        help="Number of candidates scored per GNN prediction batch.",
    )
    parser.add_argument(
        "--disable-mc-dropout",
        action="store_true",
        help="Disable MC-dropout robustness scoring for faster interactive runs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console logs; run.log is still written.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip strict checkpoint/data checks before generation.",
    )
    parser.add_argument(
        "--no-legacy-copy",
        action="store_true",
        help="Do not update data/generated/topk_electrolytes.csv or results/electrolyte/generated_electrolytes.csv.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level for run.log and console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = generate(
        output_csv=args.output,
        output_dir=args.output_dir,
        run_name=args.run_name,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        top_k=args.top_k,
        max_runtime_seconds=args.max_runtime_seconds,
        prediction_batch_size=args.prediction_batch_size,
        disable_mc_dropout=args.disable_mc_dropout,
        quiet=args.quiet,
        strict_preflight=not args.skip_preflight,
        config_path=args.config,
        legacy_outputs=not args.no_legacy_copy,
        log_level=args.log_level,
    )
    hit_rate = (
        float(table["satisfied_predicted_electrolyte_targets"].mean()) if len(table) else 0.0
    )
    if not args.quiet:
        print(f"Wrote {len(table)} electrolyte candidates to {table.attrs.get('final_csv')}")
        print(f"Run directory: {table.attrs.get('run_dir')}")
        print(f"Predicted electrolyte target hit rate: {hit_rate:.2%}")


if __name__ == "__main__":
    main()

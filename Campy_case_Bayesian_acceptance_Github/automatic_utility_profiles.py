"""Generate private utility profiles from public ordinal priors.

The LLM is not asked to invent cardinal scores.  It has already converted public
stakeholder statements into ordinal constraints in
``public_priors_Campylobacter.json``.
This module samples the corresponding maximum-entropy utility distribution,
converts valid draws to integer 100-point tables, reports score ranges, and
selects one reproducible, valid profile nearest to the componentwise medians for
the later negotiation run.

No original human score table is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from semantic_bayesian_inference import load_prior_bundle, sample_public_prior


def stable_seed(*parts: Any) -> int:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edges(prior: dict, key: str, left: str, right: str) -> list[tuple[str, str]]:
    return [(str(row[left]).upper(), str(row[right]).upper()) for row in prior.get(key, [])]


def integer_simplex_round(values: list[float], total: int) -> list[int]:
    """Round a non-negative vector to integer values that sum exactly to total.

    Every positive component remains at least one point.  The residual points are
    assigned by the largest-remainder rule with deterministic index tie-breaking.
    """
    array = np.asarray(values, dtype=float)
    if np.any(array < 0) or not np.any(array > 0):
        raise ValueError("Issue weights must contain at least one positive value")
    active = array > 1e-12
    if int(active.sum()) > total:
        raise ValueError("Integer budget is smaller than the number of active issues")
    scaled = array / float(array.sum()) * total
    result = np.floor(scaled).astype(int)
    result[active & (result == 0)] = 1
    fractions = scaled - np.floor(scaled)

    while int(result.sum()) < total:
        order = sorted(np.flatnonzero(active), key=lambda i: (-fractions[i], i))
        for index in order:
            if int(result.sum()) == total:
                break
            result[index] += 1

    while int(result.sum()) > total:
        candidates = [i for i in np.flatnonzero(active) if result[i] > 1]
        if not candidates:
            raise ValueError("Cannot reduce rounded weights while preserving active issues")
        index = min(candidates, key=lambda i: (fractions[i], -result[i], i))
        result[index] -= 1

    return [int(value) for value in result]


def quantize_one_sample(samples: dict, prior: dict, index: int, total: int) -> dict | None:
    """Convert one continuous prior draw to an integer table.

    A draw is rejected if integer rounding would reverse or tie an extracted
    strict relation.  Rejection is preferable to silently changing the public
    evidence constraint.
    """
    issues = list(samples["issues"])
    continuous_weights = [float(samples["issue_weights"][issue][index]) for issue in issues]
    weights_list = integer_simplex_round(continuous_weights, total)
    weights = dict(zip(issues, weights_list))

    relevant = {str(issue).upper() for issue in prior.get("relevant_issues", [])}
    if any(weights.get(issue, 0) <= 0 for issue in relevant):
        return None
    for high, low in _edges(prior, "issue_relations", "more_important", "less_important"):
        if weights[high] <= weights[low]:
            return None

    table: dict[str, dict[str, int]] = {}
    for issue in issues:
        weight = weights[issue]
        option_arrays = samples["scores"][issue]
        if weight == 0:
            option_scores = {option: 0 for option in sorted(option_arrays)}
        else:
            continuous_weight = continuous_weights[issues.index(issue)]
            if continuous_weight <= 0:
                return None
            option_scores = {}
            for option, array in sorted(option_arrays.items()):
                ratio = min(1.0, max(0.0, float(array[index]) / continuous_weight))
                option_scores[option] = int(math.floor(weight * ratio + 0.5))
            # The continuous sampler normalizes one option to the issue maximum.
            if max(option_scores.values()) != weight:
                return None

        table[issue] = option_scores

    for preferred, less in _edges(prior, "option_relations", "preferred", "less_preferred"):
        issue = preferred[0]
        if table[issue][preferred] <= table[issue][less]:
            return None

    if sum(max(options.values()) for options in table.values()) != total:
        return None
    return {"issue_weights": weights, "utility_table": table}


def valid_integer_draws(prior: dict, count: int, seed: int, total: int) -> tuple[list[dict], dict]:
    """Collect ``count`` valid integer draws, reporting quantization rejection."""
    accepted: list[dict] = []
    attempted = 0
    batch_index = 0
    while len(accepted) < count:
        remaining = count - len(accepted)
        batch_size = max(1000, min(50000, remaining * 3))
        sampled = sample_public_prior(
            prior,
            batch_size,
            stable_seed(seed, prior["stakeholder"], prior["source_sha256"], batch_index),
        )
        for index in range(batch_size):
            attempted += 1
            row = quantize_one_sample(sampled, prior, index, total)
            if row is not None:
                accepted.append(row)
                if len(accepted) == count:
                    break
        batch_index += 1
        if batch_index >= 100 and len(accepted) < count:
            raise RuntimeError(
                f"Could not obtain {count} valid integer draws for {prior['stakeholder']}; "
                f"accepted {len(accepted)} of {attempted}. The ordinal constraints may be "
                "incompatible with integer scoring."
            )
    return accepted, {
        "attempted_continuous_draws": attempted,
        "accepted_integer_draws": len(accepted),
        "acceptance_fraction": round(len(accepted) / attempted, 6),
    }


def summarize(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": int(np.min(array)),
        "p05": round(float(np.quantile(array, 0.05)), 3),
        "median": round(float(np.quantile(array, 0.50)), 3),
        "mean": round(float(np.mean(array)), 3),
        "p95": round(float(np.quantile(array, 0.95)), 3),
        "maximum": int(np.max(array)),
    }


def summarize_draws(draws: list[dict], prior: dict) -> dict:
    issues = sorted(prior["issues"])
    return {
        "issue_weights": {
            issue: summarize([row["issue_weights"][issue] for row in draws])
            for issue in issues
        },
        "option_scores": {
            issue: {
                option: summarize([row["utility_table"][issue][option] for row in draws])
                for option in sorted(prior["issues"][issue]["options"])
            }
            for issue in issues
        },
    }


def median_representative(draws: list[dict], prior: dict) -> tuple[int, dict]:
    """Return the valid integer draw closest to all marginal score medians.

    A table made by rounding every marginal median independently need not retain
    a 100-point budget or the strict ordinal constraints.  We therefore compute
    the coordinatewise medians and select the *observed valid draw* with minimum
    mean absolute deviation from that median vector.  Squared distance and draw
    index are deterministic tie-breakers.
    """
    coordinates = [
        (issue, option)
        for issue in sorted(prior["issues"])
        for option in sorted(prior["issues"][issue]["options"])
    ]
    matrix = np.asarray([
        [row["utility_table"][issue][option] for issue, option in coordinates]
        for row in draws
    ], dtype=float)
    medians = np.median(matrix, axis=0)
    absolute = np.mean(np.abs(matrix - medians), axis=1)
    squared = np.mean(np.square(matrix - medians), axis=1)
    index = min(range(len(draws)), key=lambda i: (absolute[i], squared[i], i))
    metadata = {
        "criterion": "minimum mean absolute deviation to the componentwise option-score medians",
        "tie_breakers": ["minimum mean squared deviation", "lowest source draw index"],
        "mean_absolute_deviation": round(float(absolute[index]), 6),
        "root_mean_squared_deviation": round(float(np.sqrt(squared[index])), 6),
        "coordinatewise_medians": {
            issue: {
                option: round(float(medians[position]), 3)
                for position, (coordinate_issue, option) in enumerate(coordinates)
                if coordinate_issue == issue
            }
            for issue in sorted(prior["issues"])
        },
        "valid_draw_preserved": True,
        "reason_not_to_round_medians_independently": (
            "Independent marginal rounding can violate the 100-point budget or strict public ordinal constraints."
        ),
    }
    return index, metadata


def preferred_deal(table: dict[str, dict[str, int]]) -> dict:
    issue_order = sorted(table)
    maximizing = {
        issue: sorted(
            option for option, value in table[issue].items()
            if value == max(table[issue].values())
        )
        for issue in issue_order
    }
    selected = [maximizing[issue][0] for issue in issue_order]
    return {
        "issue_order": issue_order,
        "maximizing_options_by_issue": maximizing,
        "canonical_starting_deal": ", ".join(selected),
        "tie_policy": "lexicographically first maximum within each issue",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample automatic integer utility profiles from public ordinal priors"
    )
    parser.add_argument("--prior_file", default="public_priors_Campylobacter.json")
    parser.add_argument("--output_dir", default="automatic_scoring")
    parser.add_argument("--monte_carlo_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.monte_carlo_samples < 100:
        raise ValueError("--monte_carlo_samples must be at least 100")

    prior_path = Path(args.prior_file).resolve()
    output = Path(args.output_dir).resolve()
    profile_dir = output / "utility_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_prior_bundle(prior_path)
    stakeholders = list(bundle["stakeholders"])
    if not stakeholders:
        raise ValueError("Prior contains no stakeholders")

    totals = {float(bundle["stakeholders"][name]["total_points"]) for name in stakeholders}
    thresholds = {
        float(bundle["stakeholders"][name]["acceptance_threshold"])
        for name in stakeholders
    }
    if len(totals) != 1 or len(thresholds) != 1:
        raise ValueError("All stakeholder priors must use one total and one threshold")
    total_float = next(iter(totals))
    if not total_float.is_integer():
        raise ValueError("Automatic integer prompts require an integer total-point budget")
    total = int(total_float)
    threshold = next(iter(thresholds))
    if not 0 <= threshold <= total:
        raise ValueError("Acceptance threshold must lie within the score budget")

    draws_by_stakeholder: dict[str, list[dict]] = {}
    diagnostics = {}
    ranges = {}
    for name in stakeholders:
        prior = bundle["stakeholders"][name]
        draws, diagnostic = valid_integer_draws(
            prior,
            args.monte_carlo_samples,
            stable_seed(args.seed, name),
            total,
        )
        draws_by_stakeholder[name] = draws
        diagnostics[name] = diagnostic
        ranges[name] = {
            "stakeholder": name,
            "public_constraints": {
                "relevant_issues": prior.get("relevant_issues", []),
                "issue_relations": prior.get("issue_relations", []),
                "option_relations": prior.get("option_relations", []),
                "uncertainties": prior.get("uncertainties", []),
            },
            **summarize_draws(draws, prior),
        }

    ranges_payload = {
        "schema": "automatic_score_ranges_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Empirical score ranges under the maximum-entropy integer utility distribution "
            "conditioned only on public ordinal constraints. These are model-based uncertainty "
            "intervals, not population confidence intervals."
        ),
        "prior_file": prior_path.name,
        "prior_sha256": sha256_file(prior_path),
        "human_score_tables_used": False,
        "monte_carlo_samples_per_stakeholder": args.monte_carlo_samples,
        "integer_total_points": total,
        "common_acceptance_threshold": threshold,
        "quantization_diagnostics": diagnostics,
        "stakeholders": ranges,
    }
    (output / "automatic_score_ranges.json").write_text(
        json.dumps(ranges_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    profile_id = "profile_median_representative"
    rows = {}
    for name in stakeholders:
        draw_index, selection_metadata = median_representative(
            draws_by_stakeholder[name], bundle["stakeholders"][name]
        )
        draw = draws_by_stakeholder[name][draw_index]
        rows[name] = {
            "stakeholder": name,
            "source_draw_index": draw_index,
            "median_representative_selection": selection_metadata,
            "issue_weights": draw["issue_weights"],
            "utility_table": draw["utility_table"],
            "preferred_deal": preferred_deal(draw["utility_table"]),
            "acceptance_threshold": threshold,
        }
    payload = {
        "schema": "automatic_private_utility_profile_v1",
        "profile_id": profile_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_prior_sha256": sha256_file(prior_path),
        "selection_rule": (
            "For each stakeholder, select the valid integer ensemble draw closest in mean absolute "
            "option-score distance to the componentwise medians; use squared distance and draw index "
            "as deterministic tie-breakers."
        ),
        "human_score_tables_used": False,
        "total_points": total,
        "acceptance_threshold": threshold,
        "stakeholders": rows,
    }
    path = profile_dir / f"{profile_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    profile_records = [{
        "profile_id": profile_id,
        "path": str(path.relative_to(output)),
        "sha256": sha256_file(path),
    }]

    manifest = {
        "schema": "automatic_utility_generation_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "maximum-entropy constrained Monte Carlo followed by relation-preserving integer quantization",
        "prior_file": str(prior_path),
        "prior_sha256": sha256_file(prior_path),
        "human_score_tables_used": False,
        "monte_carlo_samples_per_stakeholder": args.monte_carlo_samples,
        "generated_profiles": 1,
        "profile_selection": "valid draw nearest to componentwise medians",
        "seed": args.seed,
        "total_points": total,
        "acceptance_threshold": threshold,
        "profiles": profile_records,
    }
    (output / "automatic_scoring_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Generated one median-representative private utility profile in {profile_dir}")
    print(f"Score ranges: {output / 'automatic_score_ranges.json'}")


if __name__ == "__main__":
    main()

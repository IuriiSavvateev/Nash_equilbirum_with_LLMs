"""Integer-only balanced perturbation analysis (Figs. 6-7) for the ORIGINAL participant scores.

This is the post-hoc counterpart of ``balanced_perturbation_analysis.py``.  It applies
exactly the same adaptation rules to the workshop participants' own score tables:

* adaptation = bounded integer transfers among each stakeholder's active issue weights;
* every adapted option score is an integer, issue maxima still sum to 100, and
  within-issue rankings, ties and zeros are preserved by the same monotone integer
  re-quantization (``integer_rescale``); zero-weight issues stay inactive;
* "smallest genuine NBS" = exact finite search for the landscape in which some package
  gives every party >= 66 at disagreement point 65, minimizing the largest issue-weight
  change, then the summed squared change (identical lexicographic objective);
* "IR-rich" = sampled landscape with the most strictly IR packages at 65, then the
  smallest change (identical rule, not a certified global optimum).

Differences from the median-representative run are only those forced by the data:
the participant landscape has no IR package at 65 (maximin 55, gap 10), so a cap of
+/-5 points cannot produce an NBS.  The default cap grid is therefore 0-10 and the
threshold grid extends down to 54, where an unadapted strict-IR package first exists.
The exact search is vectorized so that larger caps remain tractable.

Participant scores are read only here; this script is never an input to the
automatic prior, distribution, profile or prompt-generation stages.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from balanced_perturbation_analysis import (
    Problem,
    adaptation_cost,
    deltas,
    equilibrium,
    integer_rescale,
    parse_floats,
    parse_ints,
    plot_grid,
    plot_selected,
    portable_path,
    weights_json,
)

PROFILE_ID = "profile_participant_original_scores"


# --------------------------------------------------------------------------- inputs

def participant_profile(csv_path: Path, order_reference: Path | None) -> dict:
    """Build a profile dict (same schema as the automatic profile) from the participant CSV."""
    frame = pd.read_csv(csv_path, sep=";")
    if "Stakeholder" not in frame:
        raise ValueError("Participant CSV needs a Stakeholder column")
    option_columns = [c for c in frame.columns if c != "Stakeholder"]
    issues = sorted({c[0] for c in option_columns})
    options = {issue: sorted(c for c in option_columns if c[0] == issue) for issue in issues}
    names = [str(x).strip() for x in frame["Stakeholder"]]
    # Use the automatic profile's stakeholder order so the figures align with Figs. 6-7.
    if order_reference is not None and order_reference.exists():
        reference = json.loads(order_reference.read_text(encoding="utf-8"))
        ref_names = list(reference["stakeholders"])
        if set(ref_names) == set(names):
            names = ref_names
    rows = {str(r["Stakeholder"]).strip(): r for _, r in frame.iterrows()}
    stakeholders = {}
    for name in names:
        row = rows[name]
        table = {issue: {o: int(row[o]) for o in options[issue]} for issue in issues}
        if any(v < 0 for t in table.values() for v in t.values()):
            raise ValueError(f"{name}: negative score")
        weights = {issue: max(table[issue].values()) for issue in issues}
        if sum(weights.values()) != 100:
            raise ValueError(f"{name}: issue maxima sum to {sum(weights.values())}, not 100")
        stakeholders[name] = {"issue_weights": weights, "utility_table": table}
    return {
        "profile_id": PROFILE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "post-hoc participant score table (diagnostic only)",
        "source_csv": portable_path(csv_path),
        "stakeholders": stakeholders,
    }


# ----------------------------------------------------------- vectorized weight banks

class Bank:
    """All valid integer issue-weight vectors for one stakeholder within +/-cap."""

    def __init__(self, problem: Problem, name: str, cap: int):
        original, active = problem.weights[name], problem.active[name]
        lower = np.maximum(problem.minimums[name][active], original[active] - cap)
        upper = original[active] + cap
        grids = [np.arange(int(lo), int(hi) + 1) for lo, hi in zip(lower[:-1], upper[:-1])]
        prefix = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(grids), -1).T
        last = int(original[active].sum()) - prefix.sum(axis=1)
        keep = (last >= lower[-1]) & (last <= upper[-1])
        weights = np.tile(original, (int(keep.sum()), 1))
        weights[:, active[:-1]] = prefix[keep]
        weights[:, active[-1]] = last[keep]
        if len(weights) == 0:
            raise RuntimeError(f"No valid integer transfers for {name}, cap={cap}")
        delta = (weights - original)[:, active]
        self.name, self.weights = name, weights
        self.max_change = np.abs(delta).max(axis=1)
        self.sumsq = np.square(delta).sum(axis=1)
        # Lexicographic preference used by the original exact search: (max_change, sumsq, weights).
        keys = [weights[:, j] for j in reversed(range(weights.shape[1]))] + [self.sumsq, self.max_change]
        self.order = np.lexsort(keys)
        # Per-issue lookup: re-quantized option scores for every attainable weight value.
        self.lookup = []
        for i, issue in enumerate(problem.issues):
            values = np.unique(weights[:, i])
            arr = np.zeros((int(values.max()) + 1, len(problem.options[issue])), dtype=np.int16)
            for v in values:
                table = integer_rescale(problem.tables[name][issue], int(v))
                arr[int(v)] = [table[o] for o in problem.options[issue]]
            self.lookup.append(arr)
        self.option_index = [
            np.asarray([problem.options[issue].index(p[i]) for p in problem.packages])
            for i, issue in enumerate(problem.issues)
        ]

    def package_value(self, package_index: int, rows=None) -> np.ndarray:
        w = self.weights if rows is None else self.weights[rows]
        return sum(self.lookup[i][w[:, i], self.option_index[i][package_index]] for i in range(w.shape[1])).astype(int)

    def all_package_values(self, rows) -> np.ndarray:
        w = self.weights[rows]
        out = np.zeros((len(rows), len(self.option_index[0])), dtype=np.int16)
        for i in range(w.shape[1]):
            out += self.lookup[i][w[:, i][:, None], self.option_index[i][None, :]]
        return out

    def table(self, problem: Problem, row: int) -> dict:
        return problem.table_for(self.name, self.weights[row])


# --------------------------------------------------------------- exact smallest NBS

def exact_smallest_witness(problem: Problem, banks: dict, threshold: float) -> dict:
    target = int(np.floor(threshold) + 1)
    best = None
    per_package = []
    for package_i, label in enumerate(problem.labels):
        choice, needed = {}, {}
        for name in problem.stakeholders:
            bank = banks[name]
            feasible = bank.package_value(package_i)[bank.order] >= target
            if not feasible.any():
                choice = None
                break
            row = int(bank.order[int(np.argmax(feasible))])
            choice[name] = row
            needed[name] = int(bank.max_change[row])
        if choice is None:
            continue
        per_package.append({"package": label, "required_max_issue_change": max(needed.values()), **{f"required_change_{k}": v for k, v in needed.items()}})
        if best is not None and max(needed.values()) > best[0][0]:
            continue  # cannot beat the incumbent on the primary criterion
        weights = {name: banks[name].weights[choice[name]] for name in problem.stakeholders}
        tables = {name: banks[name].table(problem, choice[name]) for name in problem.stakeholders}
        result = equilibrium(problem, tables, threshold)
        if not result["positive_gain_nbs_exists"]:
            continue
        cost = adaptation_cost(problem, weights, tables)
        key = (cost["maximum_absolute_issue_change"], cost["sum_squared_issue_change"],
               cost["maximum_absolute_option_score_change"], -result["strict_ir_package_count"], label)
        if best is None or key < best[0]:
            best = (key, {"conditioning_package": label, "weights": weights, "tables": tables,
                          "equilibrium": result, "cost": cost})
    if best is None:
        raise RuntimeError("No strict-IR integer score landscape within the largest cap; increase --caps")
    requirement = pd.DataFrame(per_package).sort_values(["required_max_issue_change", "package"])
    return best[1], requirement


# ------------------------------------------------------------- Monte Carlo grid

def sampled_grid(problem: Problem, banks: dict, cap: int, samples: int, thresholds, rng):
    picks = {name: rng.integers(0, len(banks[name].weights), size=samples) for name in problem.stakeholders}
    utility = np.stack([banks[name].all_package_values(picks[name]) for name in problem.stakeholders], axis=2).astype(np.int32)
    summaries, rows = [], []
    costs = []
    for s in range(samples):
        weights = {name: banks[name].weights[int(picks[name][s])] for name in problem.stakeholders}
        delta = np.concatenate([(weights[n] - problem.weights[n])[problem.active[n]] for n in problem.stakeholders])
        costs.append({
            "maximum_absolute_issue_change": int(np.abs(delta).max()),
            "rms_issue_change": float(np.sqrt(np.mean(delta.astype(float) ** 2))),
            "changed_stakeholders": int(sum(np.any(weights[n] != problem.weights[n]) for n in problem.stakeholders)),
        })
    for threshold in thresholds:
        weak, strict = np.all(utility >= threshold, axis=2), np.all(utility > threshold, axis=2)
        nash = np.where(strict, np.prod((utility - threshold).astype(float), axis=2), -np.inf)
        maximum = nash.max(axis=1)
        positive = np.isfinite(maximum) & (maximum > 0)
        winners = np.where(positive, nash.argmax(axis=1), -1)
        summaries.append({
            "absolute_integer_issue_weight_cap": cap, "threshold_and_disagreement_point": threshold, "samples": samples,
            "probability_positive_gain_nbs": float(positive.mean()),
            "probability_any_weak_ir_package": float((weak.sum(axis=1) > 0).mean()),
            "mean_weak_ir_package_count": float(weak.sum(axis=1).mean()),
            "mean_strict_ir_package_count": float(strict.sum(axis=1).mean()),
            "p95_strict_ir_package_count": float(np.quantile(strict.sum(axis=1), .95)),
            "maximum_strict_ir_package_count_found": int(strict.sum(axis=1).max()),
            "maximum_weak_ir_package_count_found": int(weak.sum(axis=1).max()),
        })
        if threshold == 65:
            for s in range(samples):
                # Option-score change is only needed for the selected profile; omitted here for speed.
                rows.append({"cap": cap, "sample": s, **costs[s],
                             "weak_ir_package_count": int(weak[s].sum()), "strict_ir_package_count": int(strict[s].sum()),
                             "maximum_nash_product": int(maximum[s]) if positive[s] else 0,
                             "nbs_package": problem.labels[int(winners[s])] if winners[s] >= 0 else ""})
    return summaries, rows, picks


# ------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant_scores", default="reference_participant_scores/Case1_Campy_Workshop_Alex_final_scores.csv")
    parser.add_argument("--stakeholder_order_profile", default="automatic_scoring/utility_profiles/profile_median_representative.json")
    parser.add_argument("--output_dir", default="participant_perturbation_outputs")
    parser.add_argument("--caps", default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--thresholds", default="65,64,63,62,60,58,56,54")
    parser.add_argument("--samples_per_cap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.participant_scores).resolve()
    profile = participant_profile(csv_path, Path(args.stakeholder_order_profile))
    profile_path = output / "participant_original_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    problem = Problem(profile, expected_profile_id=PROFILE_ID)

    caps, thresholds = parse_ints(args.caps), parse_floats(args.thresholds)
    if 65.0 not in thresholds:
        raise ValueError("Threshold grid must include 65")
    baseline = equilibrium(problem, problem.tables, 65.0)
    baseline_by_threshold = {f"{t:g}": equilibrium(problem, problem.tables, t) for t in thresholds}

    max_banks = {name: Bank(problem, name, max(caps)) for name in problem.stakeholders}
    exact, requirement = exact_smallest_witness(problem, max_banks, 65.0)
    requirement.to_csv(output / "package_minimum_required_issue_change.csv", index=False)

    rng = np.random.default_rng(args.seed)
    summaries, rows, picks_by_cap, banks_by_cap = [], [], {}, {}
    for cap in caps:
        banks = max_banks if cap == max(caps) else {name: Bank(problem, name, cap) for name in problem.stakeholders}
        n = 1 if cap == 0 else args.samples_per_cap
        a, b, picks = sampled_grid(problem, banks, cap, n, thresholds, rng)
        summaries.extend(a); rows.extend(b); picks_by_cap[cap] = picks; banks_by_cap[cap] = banks
        print(f"cap {cap}: {n} samples, NBS frequency at 65 = {a[thresholds.index(65.0)]['probability_positive_gain_nbs']:.4%}")
    summary, frame = pd.DataFrame(summaries), pd.DataFrame(rows)
    summary.to_csv(output / "perturbation_threshold_grid.csv", index=False)
    frame.to_csv(output / "threshold65_sample_metrics.csv", index=False)

    positive = frame[frame.maximum_nash_product > 0]
    if positive.empty:
        ir_rich = dict(exact, search_cap=int(exact["cost"]["maximum_absolute_issue_change"]), sample_index_within_cap=None,
                       selection_rule="no sampled landscape reached a positive-gain NBS; falls back to the exact smallest witness")
    else:
        best = positive.sort_values(["strict_ir_package_count", "maximum_absolute_issue_change", "rms_issue_change", "maximum_nash_product"],
                                    ascending=[False, True, True, False]).iloc[0]
        cap, s = int(best["cap"]), int(best["sample"])
        banks = banks_by_cap[cap]
        weights = {n: banks[n].weights[int(picks_by_cap[cap][n][s])] for n in problem.stakeholders}
        tables = {n: banks[n].table(problem, int(picks_by_cap[cap][n][s])) for n in problem.stakeholders}
        ir_rich = {"weights": weights, "tables": tables, "cost": adaptation_cost(problem, weights, tables),
                   "equilibrium": equilibrium(problem, tables, 65.0), "search_cap": cap, "sample_index_within_cap": s,
                   "selection_rule": "maximize strict-IR package count among sampled positive-gain integer landscapes; then minimize issue-weight change; then maximize Nash product"}

    deltas(problem, exact["weights"]).assign(profile="participant_smallest_genuine_nbs_integer").to_csv(output / "smallest_genuine_nbs_weight_changes.csv", index=False)
    deltas(problem, ir_rich["weights"]).assign(profile="participant_ir_rich_integer_monte_carlo").to_csv(output / "ir_rich_weight_changes.csv", index=False)

    report = {
        "schema": "integer_balanced_participant_score_perturbation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": portable_path(profile_path),
        "source_csv": portable_path(csv_path),
        "post_hoc_diagnostic_only": True,
        "baseline": baseline,
        "baseline_by_threshold": baseline_by_threshold,
        "method": {
            "adapted_quantity": "integer issue weights and integer option scores",
            "integer_score_rule": "all option values and issue maxima are integers; issue maxima sum to 100 for every stakeholder",
            "within_issue_rule": "nearest monotone integer rescaling preserves option rankings, ties and zeros",
            "inactive_issue_rule": "zero-weight issues remain frozen at zero",
            "stakeholder_budget": 100,
            "caps_are_absolute_integer_points": caps,
            "threshold_grid": thresholds,
            "disagreement_point_rule": "equal to tested threshold",
            "sampling_rule": "uniform draws over enumerated bounded integer issue-weight vectors, independently by stakeholder",
            "samples_per_nonzero_cap": args.samples_per_cap,
            "balanced_cost": "lexicographically minimize maximum integer issue-weight change then summed squared changes; this favors several small adjustments",
            "participant_scores_used": True,
            "relation_to_automatic_analysis": "identical adaptation rules as balanced_perturbation_analysis.py; cap and threshold grids extended because the participant landscape has a 10-point infeasibility gap at 65",
        },
        "smallest_practical_genuine_nbs_profile": {
            "conditioning_package": exact["conditioning_package"],
            "selection_rule": "exact finite search over valid integer score tables, requiring every stakeholder utility >=66 at disagreement point 65; select minimum maximum issue change then minimum squared change",
            "issue_weights": weights_json(problem, exact["weights"]),
            "utility_table": exact["tables"],
            "adaptation_cost": exact["cost"],
            "equilibrium": exact["equilibrium"],
        },
        "ir_rich_profile": {
            "selection_rule": ir_rich["selection_rule"], "global_optimum_certified": False,
            "search_cap": ir_rich["search_cap"], "sample_index_within_cap": ir_rich["sample_index_within_cap"],
            "issue_weights": weights_json(problem, ir_rich["weights"]), "utility_table": ir_rich["tables"],
            "adaptation_cost": ir_rich["cost"], "equilibrium": ir_rich["equilibrium"],
        },
        "interpretation": {
            "counterfactual": "Adapted landscapes are counterfactual preference landscapes, not claims that participants changed their scores.",
            "maximize_ir_count_as_primary_objective": False,
        },
    }
    (output / "balanced_perturbation_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_grid(summary, frame, output / "Fig6P_participant_balanced_perturbation_grid",
              suptitle="Fig. 6P — Integer-only balanced perturbation and threshold grid: original participant scores",
              exact_point=(exact["cost"]["maximum_absolute_issue_change"], exact["equilibrium"]["strict_ir_package_count"]))
    plot_selected(problem, [("Smallest genuine NBS", exact), ("IR-rich sampled profile", ir_rich)],
                  output / "Fig7P_participant_selected_adapted_landscapes",
                  title="Fig. 7P — Participant scores: smallest-NBS adaptation versus IR-rich adaptation")
    print(json.dumps({"baseline": baseline, "smallest_practical_genuine_nbs": report["smallest_practical_genuine_nbs_profile"]["equilibrium"],
                      "smallest_cost": exact["cost"], "ir_rich": ir_rich["equilibrium"], "ir_rich_cost": ir_rich["cost"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

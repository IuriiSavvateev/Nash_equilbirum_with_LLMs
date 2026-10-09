"""Integer-only balanced perturbation and threshold-grid analysis.

All adapted utility tables obey the original integer scoring rule: option scores
and issue maxima are integers, issue maxima sum to 100, and zero-weight issues
remain zero.  Adaptation is implemented as bounded integer transfers among
active issue weights.  Option tables are then re-quantized by the nearest
monotone integer rescaling, preserving option order, ties and zeros.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def package_label(package):
    return ", ".join(package)


def parse_ints(text):
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if any(x < 0 for x in values):
        raise ValueError("Integer caps must be non-negative")
    return values


def parse_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def integer_rescale(source, target_maximum):
    """Closest monotone integer rescaling; all original ties and zeros remain."""
    target_maximum = int(target_maximum)
    old_maximum = int(max(source.values()))
    if old_maximum == 0:
        if target_maximum != 0:
            raise ValueError("Inactive issue cannot be activated")
        return {option: 0 for option in source}
    levels = sorted(set(int(x) for x in source.values()))
    if target_maximum < len(levels) - 1:
        raise ValueError("Issue maximum too small to preserve score ranking")
    mapped = [int(round(level * target_maximum / old_maximum)) for level in levels]
    if levels[0] == 0:
        mapped[0] = 0
    mapped[-1] = target_maximum
    for i in range(1, len(mapped)):
        mapped[i] = max(mapped[i], mapped[i - 1] + 1)
    mapped[-1] = target_maximum
    for i in range(len(mapped) - 2, -1, -1):
        mapped[i] = min(mapped[i], mapped[i + 1] - 1)
    if levels[0] == 0:
        mapped[0] = 0
    if min(mapped) < 0 or mapped[-1] != target_maximum or any(b <= a for a, b in zip(mapped, mapped[1:])):
        raise RuntimeError("Could not make an order-preserving integer rescaling")
    lookup = dict(zip(levels, mapped))
    return {option: int(lookup[int(value)]) for option, value in source.items()}


class Problem:
    def __init__(self, profile, expected_profile_id="profile_median_representative"):
        if expected_profile_id is not None and profile.get("profile_id") != expected_profile_id:
            raise ValueError(f"Expected {expected_profile_id}")
        self.profile = profile
        self.stakeholders = list(profile["stakeholders"])
        first = next(iter(profile["stakeholders"].values()))["utility_table"]
        self.issues = sorted(first)
        self.options = {issue: sorted(first[issue]) for issue in self.issues}
        self.packages = list(itertools.product(*(self.options[issue] for issue in self.issues)))
        self.labels = [package_label(x) for x in self.packages]
        self.weights, self.tables, self.active, self.minimums = {}, {}, {}, {}
        for name, row in profile["stakeholders"].items():
            weights = np.asarray([int(row["issue_weights"][issue]) for issue in self.issues], dtype=int)
            if int(weights.sum()) != 100:
                raise ValueError(f"{name}: issue maxima do not sum to 100")
            table = {issue: {option: int(value) for option, value in row["utility_table"][issue].items()} for issue in self.issues}
            self.weights[name], self.tables[name] = weights, table
            self.active[name] = np.flatnonzero(weights > 0)
            minimum = np.zeros(len(self.issues), dtype=int)
            for i, issue in enumerate(self.issues):
                if weights[i] > 0:
                    minimum[i] = max(1, len(set(table[issue].values())) - 1)
            self.minimums[name] = minimum

    def table_for(self, name, weights):
        return {issue: integer_rescale(self.tables[name][issue], int(weights[i])) for i, issue in enumerate(self.issues)}

    def package_values(self, table):
        return np.asarray([sum(table[issue][option] for issue, option in zip(self.issues, package)) for package in self.packages], dtype=int)

    def utilities(self, tables):
        return np.column_stack([self.package_values(tables[name]) for name in self.stakeholders])


def enumerate_weights(problem, name, cap):
    original, active = problem.weights[name], problem.active[name]
    lower = np.maximum(problem.minimums[name][active], original[active] - cap)
    upper = original[active] + cap
    answer = []
    for prefix in itertools.product(*[range(int(lo), int(hi) + 1) for lo, hi in zip(lower[:-1], upper[:-1])]):
        last = int(original[active].sum() - sum(prefix))
        if int(lower[-1]) <= last <= int(upper[-1]):
            candidate = original.copy()
            candidate[active] = np.asarray([*prefix, last], dtype=int)
            answer.append(candidate)
    if not answer:
        raise RuntimeError(f"No valid integer transfers for {name}, cap={cap}")
    return answer


def candidate_bank(problem, name, cap):
    rows = []
    for weights in enumerate_weights(problem, name, cap):
        table = problem.table_for(name, weights)
        delta = weights - problem.weights[name]
        rows.append({
            "weights": weights, "table": table, "values": problem.package_values(table),
            "max_change": int(np.max(np.abs(delta[problem.active[name]]))),
            "sumsq": int(np.sum(np.square(delta[problem.active[name]]))),
        })
    return rows


def adaptation_cost(problem, weights, tables):
    all_issue, all_option, per_party = [], [], {}
    changed = 0
    for name in problem.stakeholders:
        delta = weights[name] - problem.weights[name]
        active_delta = delta[problem.active[name]]
        all_issue.extend(active_delta.tolist())
        per_party[name] = float(np.linalg.norm(active_delta))
        changed += int(np.max(np.abs(active_delta)) > 0)
        for issue in problem.issues:
            for option in problem.options[issue]:
                all_option.append(tables[name][issue][option] - problem.tables[name][issue][option])
    issue, option = np.asarray(all_issue, float), np.asarray(all_option, float)
    return {
        "maximum_absolute_issue_change": int(np.max(np.abs(issue))),
        "rms_issue_change": float(np.sqrt(np.mean(issue ** 2))),
        "sum_squared_issue_change": int(np.sum(issue ** 2)),
        "maximum_absolute_option_score_change": int(np.max(np.abs(option))),
        "rms_option_score_change": float(np.sqrt(np.mean(option ** 2))),
        "changed_stakeholders": changed,
        "stakeholder_l2_change": per_party,
    }


def equilibrium(problem, tables, threshold):
    utilities = problem.utilities(tables)
    weak, strict = np.all(utilities >= threshold, axis=1), np.all(utilities > threshold, axis=1)
    products = np.where(strict, np.prod(utilities - threshold, axis=1), -np.inf)
    if strict.any():
        peak = float(products.max()); candidates = np.flatnonzero(np.isclose(products, peak))
        index = min(candidates, key=lambda i: (-int(utilities[i].sum()), problem.labels[i]))
        kind, product = "positive-gain NBS", int(peak)
    elif weak.any():
        candidates = np.flatnonzero(weak); index = min(candidates, key=lambda i: (-int(utilities[i].sum()), problem.labels[i]))
        kind, product = "weak-IR boundary benchmark", 0
    else:
        minimum = utilities.min(axis=1); candidates = np.flatnonzero(minimum == minimum.max())
        index = min(candidates, key=lambda i: (-int(utilities[i].sum()), problem.labels[i]))
        kind, product = "maximin fallback", None
    return {
        "threshold_and_disagreement_point": threshold,
        "weak_ir_package_count": int(weak.sum()), "strict_ir_package_count": int(strict.sum()),
        "positive_gain_nbs_exists": bool(strict.any()), "benchmark_type": kind,
        "benchmark_package": problem.labels[index],
        "benchmark_utilities": {name: int(utilities[index, j]) for j, name in enumerate(problem.stakeholders)},
        "benchmark_minimum_utility": int(utilities[index].min()), "nash_product_above_threshold": product,
    }


def exact_smallest_witness(problem, cap, threshold):
    """Exact finite search over adapted integer score tables (not a relaxation)."""
    target = int(math.floor(threshold) + 1)
    banks = {name: candidate_bank(problem, name, cap) for name in problem.stakeholders}
    best = None
    for package_i, label in enumerate(problem.labels):
        choice = {}
        for name in problem.stakeholders:
            feasible = [x for x in banks[name] if int(x["values"][package_i]) >= target]
            if not feasible:
                choice = None; break
            choice[name] = min(feasible, key=lambda x: (x["max_change"], x["sumsq"], tuple(x["weights"])))
        if choice is None:
            continue
        weights = {name: choice[name]["weights"] for name in problem.stakeholders}
        tables = {name: choice[name]["table"] for name in problem.stakeholders}
        result = equilibrium(problem, tables, threshold)
        if not result["positive_gain_nbs_exists"]:
            continue
        record = {"conditioning_package": label, "weights": weights, "tables": tables, "equilibrium": result, "cost": adaptation_cost(problem, weights, tables)}
        key = (record["cost"]["maximum_absolute_issue_change"], record["cost"]["sum_squared_issue_change"], record["cost"]["maximum_absolute_option_score_change"], -result["strict_ir_package_count"], label)
        if best is None or key < best[0]:
            best = (key, record)
    if best is None:
        raise RuntimeError("No strict-IR integer score landscape in the requested cap")
    return best[1]


def sampled_grid(problem, banks, cap, samples, thresholds, rng):
    selected = {name: rng.integers(0, len(banks[name]), size=samples) for name in problem.stakeholders}
    utility = np.stack([np.vstack([banks[name][int(i)]["values"] for i in selected[name]]) for name in problem.stakeholders], axis=2)
    profiles, costs = [], []
    for s in range(samples):
        weights = {name: banks[name][int(selected[name][s])]["weights"] for name in problem.stakeholders}
        tables = {name: banks[name][int(selected[name][s])]["table"] for name in problem.stakeholders}
        profiles.append({"weights": weights, "tables": tables}); costs.append(adaptation_cost(problem, weights, tables))
    summaries, rows = [], []
    for threshold in thresholds:
        weak, strict = np.all(utility >= threshold, axis=2), np.all(utility > threshold, axis=2)
        nash = np.where(strict, np.prod(utility - threshold, axis=2), -np.inf)
        maximum = nash.max(axis=1); positive = np.isfinite(maximum) & (maximum > 0); winners = np.where(positive, nash.argmax(axis=1), -1)
        summaries.append({
            "absolute_integer_issue_weight_cap": cap, "threshold_and_disagreement_point": threshold, "samples": samples,
            "probability_positive_gain_nbs": float(positive.mean()), "probability_any_weak_ir_package": float((weak.sum(axis=1) > 0).mean()),
            "mean_weak_ir_package_count": float(weak.sum(axis=1).mean()), "mean_strict_ir_package_count": float(strict.sum(axis=1).mean()),
            "p95_strict_ir_package_count": float(np.quantile(strict.sum(axis=1), .95)), "maximum_strict_ir_package_count_found": int(strict.sum(axis=1).max()),
            "maximum_weak_ir_package_count_found": int(weak.sum(axis=1).max()),
        })
        if threshold == 65:
            for s in range(samples):
                c = costs[s]
                rows.append({"cap": cap, "sample": s, "maximum_absolute_issue_change": c["maximum_absolute_issue_change"], "rms_issue_change": c["rms_issue_change"], "maximum_absolute_option_score_change": c["maximum_absolute_option_score_change"], "changed_stakeholders": c["changed_stakeholders"], "weak_ir_package_count": int(weak[s].sum()), "strict_ir_package_count": int(strict[s].sum()), "maximum_nash_product": int(maximum[s]) if positive[s] else 0, "nbs_package": problem.labels[int(winners[s])] if winners[s] >= 0 else ""})
    return summaries, rows, profiles


# ------------------------------------------------------------------ vectorized engine
# Same weight enumeration order, objective and sampling stream as candidate_bank /
# exact_smallest_witness / sampled_grid above (verified to reproduce the cached results),
# but without materializing a score table per weight vector, so larger caps are tractable.

class Bank:
    def __init__(self, problem, name, cap):
        original, active = problem.weights[name], problem.active[name]
        lower = np.maximum(problem.minimums[name][active], original[active] - cap)
        upper = original[active] + cap
        grids = [np.arange(int(lo), int(hi) + 1) for lo, hi in zip(lower[:-1], upper[:-1])]
        prefix = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(grids), -1).T
        last = int(original[active].sum()) - prefix.sum(axis=1)
        keep = (last >= lower[-1]) & (last <= upper[-1])
        if not keep.any():
            raise RuntimeError(f"No valid integer transfers for {name}, cap={cap}")
        weights = np.tile(original, (int(keep.sum()), 1))
        weights[:, active[:-1]] = prefix[keep]
        weights[:, active[-1]] = last[keep]
        delta = (weights - original)[:, active]
        self.name, self.weights = name, weights
        self.max_change = np.abs(delta).max(axis=1)
        self.sumsq = np.square(delta).sum(axis=1)
        self.order = np.lexsort([weights[:, j] for j in reversed(range(weights.shape[1]))] + [self.sumsq, self.max_change])
        self.lookup, self.option_change = [], []
        for i, issue in enumerate(problem.issues):
            opts = problem.options[issue]
            base = np.asarray([problem.tables[name][issue][o] for o in opts])
            size = int(weights[:, i].max()) + 1
            arr = np.zeros((size, len(opts)), dtype=np.int16)
            change = np.zeros(size, dtype=int)
            for v in np.unique(weights[:, i]):
                table = integer_rescale(problem.tables[name][issue], int(v))
                arr[int(v)] = [table[o] for o in opts]
                change[int(v)] = int(np.max(np.abs(arr[int(v)] - base)))
            self.lookup.append(arr); self.option_change.append(change)
        self.option_index = [np.asarray([problem.options[issue].index(p[i]) for p in problem.packages]) for i, issue in enumerate(problem.issues)]

    def package_value(self, package_index):
        return sum(self.lookup[i][self.weights[:, i], self.option_index[i][package_index]] for i in range(self.weights.shape[1])).astype(int)

    def all_package_values(self, rows):
        w = self.weights[rows]
        out = np.zeros((len(rows), len(self.option_index[0])), dtype=np.int32)
        for i in range(w.shape[1]):
            out += self.lookup[i][w[:, i][:, None], self.option_index[i][None, :]]
        return out

    def max_option_change(self, rows):
        w = self.weights[rows]
        return np.max(np.column_stack([self.option_change[i][w[:, i]] for i in range(w.shape[1])]), axis=1)

    def table(self, problem, row):
        return problem.table_for(self.name, self.weights[row])


def fast_exact_smallest_witness(problem, banks, threshold):
    """Exact search (same objective as exact_smallest_witness); returns None if infeasible."""
    target = int(math.floor(threshold) + 1)
    best = None
    for package_i, label in enumerate(problem.labels):
        choice = {}
        for name in problem.stakeholders:
            bank = banks[name]
            feasible = bank.package_value(package_i)[bank.order] >= target
            if not feasible.any():
                choice = None; break
            choice[name] = int(bank.order[int(np.argmax(feasible))])
        if choice is None:
            continue
        if best is not None and max(int(banks[n].max_change[r]) for n, r in choice.items()) > best[0][0]:
            continue
        weights = {n: banks[n].weights[r] for n, r in choice.items()}
        tables = {n: banks[n].table(problem, r) for n, r in choice.items()}
        result = equilibrium(problem, tables, threshold)
        if not result["positive_gain_nbs_exists"]:
            continue
        cost = adaptation_cost(problem, weights, tables)
        key = (cost["maximum_absolute_issue_change"], cost["sum_squared_issue_change"], cost["maximum_absolute_option_score_change"], -result["strict_ir_package_count"], label)
        if best is None or key < best[0]:
            best = (key, {"conditioning_package": label, "weights": weights, "tables": tables, "equilibrium": result, "cost": cost})
    return None if best is None else best[1]


def fast_sampled_grid(problem, banks, cap, samples, thresholds, rng):
    picks = {name: rng.integers(0, len(banks[name].weights), size=samples) for name in problem.stakeholders}
    utility = np.stack([banks[name].all_package_values(picks[name]) for name in problem.stakeholders], axis=2)
    deltas_all = np.concatenate([(banks[n].weights[picks[n]] - problem.weights[n])[:, problem.active[n]] for n in problem.stakeholders], axis=1)
    max_issue = np.abs(deltas_all).max(axis=1)
    rms_issue = np.sqrt(np.mean(deltas_all.astype(float) ** 2, axis=1))
    max_option = np.max(np.column_stack([banks[n].max_option_change(picks[n]) for n in problem.stakeholders]), axis=1)
    changed = np.sum(np.column_stack([np.abs((banks[n].weights[picks[n]] - problem.weights[n])).max(axis=1) > 0 for n in problem.stakeholders]), axis=1)
    summaries, rows = [], []
    for threshold in thresholds:
        weak, strict = np.all(utility >= threshold, axis=2), np.all(utility > threshold, axis=2)
        nash = np.where(strict, np.prod(utility - threshold, axis=2), -np.inf)
        maximum = nash.max(axis=1); positive = np.isfinite(maximum) & (maximum > 0); winners = np.where(positive, nash.argmax(axis=1), -1)
        summaries.append({
            "absolute_integer_issue_weight_cap": cap, "threshold_and_disagreement_point": threshold, "samples": samples,
            "probability_positive_gain_nbs": float(positive.mean()), "probability_any_weak_ir_package": float((weak.sum(axis=1) > 0).mean()),
            "mean_weak_ir_package_count": float(weak.sum(axis=1).mean()), "mean_strict_ir_package_count": float(strict.sum(axis=1).mean()),
            "p95_strict_ir_package_count": float(np.quantile(strict.sum(axis=1), .95)), "maximum_strict_ir_package_count_found": int(strict.sum(axis=1).max()),
            "maximum_weak_ir_package_count_found": int(weak.sum(axis=1).max()),
        })
        if threshold == 65:
            for s in range(samples):
                rows.append({"cap": cap, "sample": s, "maximum_absolute_issue_change": int(max_issue[s]), "rms_issue_change": float(rms_issue[s]), "maximum_absolute_option_score_change": int(max_option[s]), "changed_stakeholders": int(changed[s]), "weak_ir_package_count": int(weak[s].sum()), "strict_ir_package_count": int(strict[s].sum()), "maximum_nash_product": int(maximum[s]) if positive[s] else 0, "nbs_package": problem.labels[int(winners[s])] if winners[s] >= 0 else ""})
    return summaries, rows, picks


def weights_json(problem, weights):
    return {name: {issue: int(weights[name][i]) for i, issue in enumerate(problem.issues)} for name in problem.stakeholders}


def deltas(problem, weights):
    return pd.DataFrame([{"stakeholder": name, "issue": issue, "original_weight": int(problem.weights[name][i]), "adapted_weight": int(weights[name][i]), "change": int(weights[name][i] - problem.weights[name][i])} for name in problem.stakeholders for i, issue in enumerate(problem.issues)])


def portable_path(path):
    """Store project-relative paths so cached reports survive moving the bundle (e.g. to Colab/Drive)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def percent_label(value):
    if value == 0:
        return "0%"
    if value >= 99.95:
        return "100%"
    if value < 0.1:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def plot_grid(summary, rows, output, suptitle="Fig. 6 — Integer-only balanced perturbation and threshold grid", exact_point=None):
    caps = sorted(summary.absolute_integer_issue_weight_cap.unique()); thresholds = sorted(summary.threshold_and_disagreement_point.unique(), reverse=True)
    matrices = [summary.pivot(index="threshold_and_disagreement_point", columns="absolute_integer_issue_weight_cap", values=field).reindex(index=thresholds, columns=caps) for field in ["probability_positive_gain_nbs", "maximum_strict_ir_package_count_found"]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for axis, matrix, title, cmap, percent in [(axes[0], matrices[0] * 100, "A  Positive-gain NBS frequency", "Blues", True), (axes[1], matrices[1], "B  Maximum strict-IR packages found", "YlGn", False)]:
        im = axis.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap); axis.set_xticks(range(len(caps)), caps); axis.set_yticks(range(len(thresholds)), [f"{x:g}" for x in thresholds]); axis.set_xlabel("Maximum integer issue-weight transfer (points)"); axis.set_ylabel("Threshold = disagreement point"); axis.set_title(title, loc="left", fontweight="bold")
        for i in range(len(thresholds)):
            for j in range(len(caps)): axis.text(j, i, percent_label(matrix.iloc[i,j]) if percent else f"{matrix.iloc[i,j]:.0f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=axis, fraction=.046, pad=.04)
    positive = rows[rows.maximum_nash_product > 0]; axes[2].scatter(positive.maximum_absolute_issue_change, positive.strict_ir_package_count, s=12, alpha=.18, color="#2563a6", label="integer landscapes")
    if len(positive):
        frontier = []
        for _, row in positive.sort_values(["maximum_absolute_issue_change", "rms_issue_change"]).iterrows():
            if not frontier or row.strict_ir_package_count > frontier[-1][1]: frontier.append((row.maximum_absolute_issue_change, row.strict_ir_package_count))
        x, y = zip(*frontier); axes[2].step(x, y, where="post", color="#b45309", linewidth=2.3, label="observed IR frontier")
    if exact_point is not None: axes[2].scatter([exact_point[0]], [exact_point[1]], marker="*", s=260, color="#b91c1c", zorder=5, label="exact smallest genuine NBS")
    axes[2].set_xlabel("Maximum absolute integer issue-weight change"); axes[2].set_ylabel("Strictly IR packages at 65"); axes[2].set_title("C  Optionality versus integer adaptation", loc="left", fontweight="bold"); axes[2].grid(alpha=.3); axes[2].legend(frameon=False)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    for ext in ("png", "pdf"): fig.savefig(output.with_suffix("." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selected(problem, selected, output, title="Fig. 7 — Integer smallest-NBS adaptation versus integer IR-rich adaptation", threshold=65):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    limit = max(1, max(abs(x) for _, row in selected for x in deltas(problem, row["weights"]).change))
    for col, (label, row) in enumerate(selected):
        matrix = deltas(problem, row["weights"]).pivot(index="stakeholder", columns="issue", values="change").reindex(index=problem.stakeholders, columns=problem.issues)
        im = axes[0,col].imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"); axes[0,col].set_xticks(range(len(problem.issues)), problem.issues); axes[0,col].set_yticks(range(len(problem.stakeholders)), problem.stakeholders); axes[0,col].set_title(label + ": integer issue-weight changes", fontweight="bold")
        for i in range(len(problem.stakeholders)):
            for j in range(len(problem.issues)): axes[0,col].text(j, i, f"{int(matrix.iloc[i,j]):+d}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes[0,col], fraction=.046, pad=.04, label="integer points")
        eq = row["equilibrium"]; utility = [eq["benchmark_utilities"][name] for name in problem.stakeholders]
        axes[1,col].barh(problem.stakeholders, utility, color="#3478b8" if col == 0 else "#4d9461"); axes[1,col].axvline(threshold, color="#b91c1c", linestyle="--", linewidth=2); axes[1,col].set_xlim(min(60, min(utility)-2), max(72,max(utility)+2)); axes[1,col].set_xlabel("Integer utility"); axes[1,col].set_title(f"NBS {eq['benchmark_package']} | strict IR: {eq['strict_ir_package_count']}", fontweight="bold")
        for y, value in enumerate(utility): axes[1,col].text(value+.15, y, str(value), va="center", fontsize=9)
        axes[1,col].spines[["top","right"]].set_visible(False); axes[1,col].grid(axis="x", alpha=.25)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ext in ("png", "pdf"): fig.savefig(output.with_suffix("." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="automatic_scoring/utility_profiles/profile_median_representative.json")
    parser.add_argument("--output_dir", default="perturbation_outputs")
    parser.add_argument("--caps", default="0,1,2,3,4,5")
    parser.add_argument("--thresholds", default="65,64,63,62,60")
    parser.add_argument("--samples_per_cap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max_auto_cap", type=int, default=15, help="Widen the exact search up to this cap if --caps is too small for a genuine NBS.")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile).resolve(); problem = Problem(json.loads(profile_path.read_text(encoding="utf-8")))
    caps, thresholds = parse_ints(args.caps), parse_floats(args.thresholds)
    if 65.0 not in thresholds: raise ValueError("Threshold grid must include 65")
    baseline_weights = {name: problem.weights[name].copy() for name in problem.stakeholders}; baseline = equilibrium(problem, problem.tables, 65.0)
    # Exact smallest genuine NBS; widen the cap automatically if the requested grid is too small.
    requested_max_cap, exact, search_cap = max(caps), None, max(caps)
    while True:
        exact_banks = {name: Bank(problem, name, search_cap) for name in problem.stakeholders}
        exact = fast_exact_smallest_witness(problem, exact_banks, 65.0)
        if exact is not None or search_cap >= args.max_auto_cap:
            break
        search_cap += 1
        print(f"No genuine NBS within +/-{search_cap - 1}; widening the exact search to +/-{search_cap}", flush=True)
    if exact is None:
        raise RuntimeError(
            f"No integer landscape gives every party >=66 at 65 within +/-{args.max_auto_cap} issue-weight points. "
            f"Unadapted maximin package {baseline['benchmark_package']} has minimum utility {baseline['benchmark_minimum_utility']}. "
            "Raise --max_auto_cap or lower the threshold.")
    if search_cap > requested_max_cap:
        caps = caps + list(range(requested_max_cap + 1, search_cap + 1))
        print(f"Cap grid extended to {caps} so it contains the smallest genuine NBS.", flush=True)
    rng = np.random.default_rng(args.seed); summaries, rows, picks_by_cap, banks_by_cap = [], [], {}, {}
    for cap in caps:
        banks = exact_banks if cap == search_cap else {name: Bank(problem, name, cap) for name in problem.stakeholders}; n = 1 if cap == 0 else args.samples_per_cap
        a, b, c = fast_sampled_grid(problem, banks, cap, n, thresholds, rng); summaries.extend(a); rows.extend(b); picks_by_cap[cap] = c; banks_by_cap[cap] = banks
    summary, frame = pd.DataFrame(summaries), pd.DataFrame(rows); summary.to_csv(output / "perturbation_threshold_grid.csv", index=False); frame.to_csv(output / "threshold65_sample_metrics.csv", index=False)
    positive = frame[frame.maximum_nash_product > 0]
    if positive.empty:
        print("No sampled landscape reached a positive-gain NBS; the IR-rich profile falls back to the exact smallest witness.", flush=True)
        ir_rich = {"weights": exact["weights"], "tables": exact["tables"], "cost": exact["cost"], "equilibrium": exact["equilibrium"], "search_cap": search_cap, "sample_index_within_cap": None, "selection_rule": "no sampled landscape reached a positive-gain NBS; fallback to the exact smallest genuine-NBS landscape", "global_optimum_certified": False}
    else:
        best = positive.sort_values(["strict_ir_package_count", "maximum_absolute_issue_change", "rms_issue_change", "maximum_nash_product"], ascending=[False, True, True, False]).iloc[0]
        cap_b, s_b = int(best["cap"]), int(best["sample"])
        rows_b = {n: int(picks_by_cap[cap_b][n][s_b]) for n in problem.stakeholders}
        w_b = {n: banks_by_cap[cap_b][n].weights[r] for n, r in rows_b.items()}
        t_b = {n: banks_by_cap[cap_b][n].table(problem, r) for n, r in rows_b.items()}
        ir_rich = {"weights": w_b, "tables": t_b, "cost": adaptation_cost(problem, w_b, t_b), "equilibrium": equilibrium(problem, t_b, 65.0), "search_cap": cap_b, "sample_index_within_cap": s_b, "selection_rule": "maximize strict-IR package count among sampled positive-gain integer landscapes; then minimize issue-weight change; then maximize Nash product", "global_optimum_certified": False}
    deltas(problem,exact["weights"]).assign(profile="smallest_genuine_nbs_integer").to_csv(output / "smallest_genuine_nbs_weight_changes.csv",index=False); deltas(problem,ir_rich["weights"]).assign(profile="ir_rich_integer_monte_carlo").to_csv(output / "ir_rich_weight_changes.csv",index=False)
    report = {"schema":"integer_balanced_median_profile_perturbation_v2","created_utc":datetime.now(timezone.utc).isoformat(),"source_profile":portable_path(profile_path),"baseline":baseline,"method":{"adapted_quantity":"integer issue weights and integer option scores","integer_score_rule":"all option values and issue maxima are integers; issue maxima sum to 100 for every stakeholder","within_issue_rule":"nearest monotone integer rescaling preserves option rankings, ties and zeros","inactive_issue_rule":"zero-weight issues remain frozen at zero","stakeholder_budget":100,"caps_are_absolute_integer_points":caps,"threshold_grid":thresholds,"disagreement_point_rule":"equal to tested threshold","sampling_rule":"uniform draws over enumerated bounded integer issue-weight vectors, independently by stakeholder","samples_per_nonzero_cap":args.samples_per_cap,"requested_maximum_cap":requested_max_cap,"exact_search_cap_used":search_cap,"caps_extended_automatically":search_cap > requested_max_cap,"balanced_cost":"lexicographically minimize maximum integer issue-weight change then summed squared changes; this favors several small adjustments","participant_scores_used":False},"smallest_practical_genuine_nbs_profile":{"conditioning_package":exact["conditioning_package"],"selection_rule":"exact finite search over valid integer score tables, requiring every stakeholder utility >=66 at disagreement point 65; select minimum maximum issue change then minimum squared change","issue_weights":weights_json(problem,exact["weights"]),"utility_table":exact["tables"],"adaptation_cost":exact["cost"],"equilibrium":exact["equilibrium"]},"ir_rich_profile":{"selection_rule":ir_rich["selection_rule"],"global_optimum_certified":False,"search_cap":ir_rich["search_cap"],"sample_index_within_cap":ir_rich["sample_index_within_cap"],"issue_weights":weights_json(problem,ir_rich["weights"]),"utility_table":ir_rich["tables"],"adaptation_cost":ir_rich["cost"],"equilibrium":ir_rich["equilibrium"]},"interpretation":{"maximize_ir_count_as_primary_objective":False,"reason":"IR count measures optionality but can reward larger preference distortions and barely acceptable packages; it is reported separately after the smallest genuine-NBS criterion."}}
    (output / "balanced_perturbation_results.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    plot_grid(summary,frame,output / "Fig6_balanced_perturbation_grid",exact_point=(exact["cost"]["maximum_absolute_issue_change"],exact["equilibrium"]["strict_ir_package_count"])); plot_selected(problem,[("Smallest genuine NBS",exact),("IR-rich sampled profile",ir_rich)],output / "Fig7_selected_adapted_landscapes")
    print(json.dumps({"baseline":baseline,"smallest_practical_genuine_nbs":report["smallest_practical_genuine_nbs_profile"],"ir_rich_profile":report["ir_rich_profile"]},indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()

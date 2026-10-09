"""Post-hoc interval checks and equilibrium characterization at baseline 65.

The participant score file is read only here, after the automatic distribution
and median-representative profile have been generated.  It is never an input to
the prior, distribution, representative-profile, or prompt-generation stages.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_package(package: tuple[str, ...]) -> str:
    return ", ".join(package)


def participant_tables(path: Path, issue_options: dict[str, list[str]]) -> dict:
    frame = pd.read_csv(path, sep=";")
    if "Stakeholder" not in frame:
        raise ValueError("Participant CSV needs a Stakeholder column")
    expected = [option for issue in sorted(issue_options) for option in issue_options[issue]]
    missing = [option for option in expected if option not in frame]
    if missing:
        raise ValueError(f"Participant CSV is missing options: {missing}")
    tables = {}
    for _, row in frame.iterrows():
        name = str(row["Stakeholder"]).strip()
        tables[name] = {
            issue: {option: int(row[option]) for option in options}
            for issue, options in issue_options.items()
        }
    return tables


def enumerate_packages(
    tables: dict[str, dict[str, dict[str, int]]], threshold: float
) -> tuple[pd.DataFrame, dict]:
    stakeholders = list(tables)
    issue_order = sorted(next(iter(tables.values())))
    options_by_issue = [sorted(next(iter(tables.values()))[issue]) for issue in issue_order]
    packages = list(itertools.product(*options_by_issue))
    utilities = np.asarray([
        [sum(tables[name][issue][option] for issue, option in zip(issue_order, package)) for name in stakeholders]
        for package in packages
    ], dtype=float)
    minimum = utilities.min(axis=1)
    total = utilities.sum(axis=1)
    ir = np.all(utilities >= threshold, axis=1)
    strict_ir = np.all(utilities > threshold, axis=1)
    gains = np.maximum(utilities - threshold, 0.0)
    nash_product = np.where(ir, np.prod(gains, axis=1), np.nan)

    dominated = np.zeros(len(packages), dtype=bool)
    for index in range(len(packages)):
        dominated[index] = bool(np.any(
            np.all(utilities >= utilities[index], axis=1)
            & np.any(utilities > utilities[index], axis=1)
        ))
    pareto = ~dominated

    if ir.any():
        candidates = np.flatnonzero(ir)
        benchmark_index = min(
            candidates,
            key=lambda i: (-float(nash_product[i]), -float(total[i]), canonical_package(packages[i])),
        )
        benchmark_type = (
            "Nash bargaining solution"
            if float(nash_product[benchmark_index]) > 0
            else "weak-IR Nash-product tie (no package gives every party a strictly positive gain)"
        )
    else:
        candidates = np.flatnonzero(minimum == minimum.max())
        benchmark_index = min(
            candidates,
            key=lambda i: (-float(total[i]), canonical_package(packages[i])),
        )
        benchmark_type = "maximin fallback (no individually rational package)"

    frame = pd.DataFrame({
        "package": [canonical_package(package) for package in packages],
        **{name: utilities[:, position] for position, name in enumerate(stakeholders)},
        "minimum_utility": minimum,
        "total_utility": total,
        "individually_rational_at_65": ir,
        "nash_product": nash_product,
        "pareto_efficient": pareto,
    })
    benchmark_utilities = {
        name: int(utilities[benchmark_index, position])
        for position, name in enumerate(stakeholders)
    }
    summary = {
        "threshold": threshold,
        "stakeholders": stakeholders,
        "issues": issue_order,
        "package_count": len(packages),
        "individually_rational_package_count": int(ir.sum()),
        "strictly_individually_rational_package_count": int(strict_ir.sum()),
        "equilibrium_exists_at_threshold": bool(ir.any()),
        "strict_positive_gain_equilibrium_exists": bool(strict_ir.any()),
        "pareto_efficient_package_count": int(pareto.sum()),
        "benchmark_type": benchmark_type,
        "benchmark_package": canonical_package(packages[benchmark_index]),
        "benchmark_utilities": benchmark_utilities,
        "benchmark_minimum_utility": int(minimum[benchmark_index]),
        "infeasibility_gap_to_65": round(max(0.0, threshold - float(minimum.max())), 3),
        "nash_product": None if not ir.any() else float(nash_product[benchmark_index]),
    }
    return frame, summary


def interval_rows(ranges: dict, old_tables: dict) -> pd.DataFrame:
    rows = []
    auto_names = set(ranges["stakeholders"])
    if set(old_tables) != auto_names:
        raise ValueError(
            f"Stakeholder mismatch: automatic={sorted(auto_names)}, participant={sorted(old_tables)}"
        )
    for name, stakeholder in ranges["stakeholders"].items():
        for issue, options in stakeholder["option_scores"].items():
            for option, stats in options.items():
                participant = old_tables[name][issue][option]
                rows.append({
                    "stakeholder": name,
                    "issue": issue,
                    "option": option,
                    "participant_score": participant,
                    "automatic_minimum": stats["minimum"],
                    "automatic_p05": stats["p05"],
                    "automatic_median": stats["median"],
                    "automatic_mean": stats["mean"],
                    "automatic_p95": stats["p95"],
                    "automatic_maximum": stats["maximum"],
                    "inside_central_90_interval": bool(stats["p05"] <= participant <= stats["p95"]),
                    "inside_full_simulated_range": bool(stats["minimum"] <= participant <= stats["maximum"]),
                })
    return pd.DataFrame(rows)


def style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d9e0e8", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_equilibrium(
    median_frame: pd.DataFrame,
    median_summary: dict,
    old_frame: pd.DataFrame,
    old_summary: dict,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    cases = [
        ("Median-representative automatic scores", median_frame, median_summary),
        ("Participant-provided scores", old_frame, old_summary),
    ]
    colors = ["#2463a2", "#b45f06"]
    for row, (label, frame, summary) in enumerate(cases):
        left, right = axes[row]
        values = frame["minimum_utility"].to_numpy()
        left.hist(values, bins=np.arange(values.min() - 0.5, values.max() + 1.5), color=colors[row], alpha=0.82)
        left.axvline(65, color="#b91c1c", linestyle="--", linewidth=2, label="65 baseline")
        if summary["benchmark_minimum_utility"] != 65:
            left.axvline(summary["benchmark_minimum_utility"], color="#111827", linewidth=2, label="benchmark minimum")
        left.set_title(f"{label}: worst-party utility across all packages")
        left.set_xlabel("Minimum stakeholder utility")
        left.set_ylabel("Packages")
        left.legend(frameon=False)
        style_axes(left)

        names = summary["stakeholders"]
        scores = [summary["benchmark_utilities"][name] for name in names]
        right.barh(names, scores, color=colors[row], alpha=0.88)
        right.axvline(65, color="#b91c1c", linestyle="--", linewidth=2)
        right.set_xlim(0, max(100, max(scores) + 8))
        right.set_xlabel("Utility")
        short_type = (
            "Boundary benchmark (zero Nash product)"
            if summary["equilibrium_exists_at_threshold"]
            else "Maximin fallback (no IR package)"
        )
        status = (
            f"IR packages: {summary['individually_rational_package_count']} | "
            f"Pareto: {summary['pareto_efficient_package_count']}\n"
            f"{short_type}\n{summary['benchmark_package']}"
        )
        right.set_title(status, fontsize=10.5)
        for y, score in enumerate(scores):
            right.text(score + 1, y, str(score), va="center", fontsize=9)
        style_axes(right)
    fig.suptitle("Fig. 4 — Equilibrium characterization (disagreement baseline = 65)", fontsize=16, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_intervals(coverage: pd.DataFrame, output: Path) -> None:
    stakeholders = list(dict.fromkeys(coverage["stakeholder"]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for axis, name in zip(axes.flat, stakeholders):
        rows = coverage[coverage["stakeholder"] == name].reset_index(drop=True)
        x = np.arange(len(rows))
        for i, row in rows.iterrows():
            axis.vlines(i, row["automatic_minimum"], row["automatic_maximum"], color="#cbd5e1", linewidth=4, zorder=1)
            axis.vlines(i, row["automatic_p05"], row["automatic_p95"], color="#2563a6", linewidth=7, zorder=2)
        axis.scatter(x, rows["automatic_median"], color="white", edgecolor="#0f3d66", s=34, zorder=3, label="automatic median")
        inside = rows["inside_central_90_interval"].to_numpy(dtype=bool)
        axis.scatter(x[inside], rows.loc[inside, "participant_score"], marker="x", color="#15803d", s=55, linewidth=2, zorder=4, label="participant: inside p05–p95")
        axis.scatter(x[~inside], rows.loc[~inside, "participant_score"], marker="x", color="#c2410c", s=60, linewidth=2.2, zorder=4, label="participant: outside p05–p95")
        for boundary in np.flatnonzero(rows["issue"].to_numpy()[1:] != rows["issue"].to_numpy()[:-1]) + 0.5:
            axis.axvline(boundary, color="#94a3b8", linewidth=0.8)
        count = int(inside.sum())
        axis.set_title(f"{name} — {count}/{len(rows)} inside central 90% interval")
        axis.set_xticks(x)
        axis.set_xticklabels(rows["option"], rotation=60, ha="right", fontsize=8)
        axis.set_ylim(-2, max(45, float(coverage[["automatic_maximum", "participant_score"]].to_numpy().max()) + 5))
        axis.set_ylabel("Option score")
        style_axes(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle(
        "Fig. 5 — Participant scores versus automatic step-5 intervals\n"
        "thin gray: simulated min–max | thick blue: p05–p95 | circle: median | ×: participant",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.88, hspace=0.27, wspace=0.05)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze automatic and participant Campylobacter scores")
    parser.add_argument("--profile", default="automatic_scoring/utility_profiles/profile_median_representative.json")
    parser.add_argument("--ranges", default="automatic_scoring/automatic_score_ranges.json")
    parser.add_argument("--participant_scores", default="reference_participant_scores/Case1_Campy_Workshop_Alex_final_scores.csv")
    parser.add_argument("--output_dir", default="analysis_outputs")
    parser.add_argument("--threshold", type=float, default=65.0)
    args = parser.parse_args()

    profile_path = Path(args.profile).resolve()
    ranges_path = Path(args.ranges).resolve()
    participant_path = Path(args.participant_scores).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    profile = load_json(profile_path)
    ranges = load_json(ranges_path)
    median_tables = {name: row["utility_table"] for name, row in profile["stakeholders"].items()}
    issue_options = {
        issue: sorted(options)
        for issue, options in next(iter(median_tables.values())).items()
    }
    old_tables = participant_tables(participant_path, issue_options)

    median_packages, median_summary = enumerate_packages(median_tables, args.threshold)
    old_packages, old_summary = enumerate_packages(old_tables, args.threshold)
    median_packages.to_csv(output / "equilibrium_packages_median_representative.csv", index=False)
    old_packages.to_csv(output / "equilibrium_packages_participant_scores.csv", index=False)
    equilibrium = {
        "schema": "campylobacter_equilibrium_characterization_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disagreement_baseline": args.threshold,
        "automatic_median_representative": median_summary,
        "participant_scores": old_summary,
    }
    (output / "equilibrium_summary.json").write_text(
        json.dumps(equilibrium, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_equilibrium(
        median_packages,
        median_summary,
        old_packages,
        old_summary,
        output / "Fig4_equilibrium_characterization",
    )

    coverage = interval_rows(ranges, old_tables)
    coverage.to_csv(output / "participant_score_interval_coverage.csv", index=False)
    overall_inside = int(coverage["inside_central_90_interval"].sum())
    full_inside = int(coverage["inside_full_simulated_range"].sum())
    coverage_summary = {
        "schema": "participant_score_interval_coverage_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interval_definition": "empirical p05-p95 of step-5 automatic integer-score distribution",
        "interpretation": "model-based central intervals, not population confidence intervals",
        "post_hoc_only": True,
        "participant_scores_used_to_generate_automatic_distribution": False,
        "overall": {
            "scores_checked": len(coverage),
            "inside_central_90_interval": overall_inside,
            "inside_central_90_fraction": round(overall_inside / len(coverage), 4),
            "inside_full_simulated_range": full_inside,
            "inside_full_simulated_fraction": round(full_inside / len(coverage), 4),
        },
        "by_stakeholder": {
            name: {
                "scores_checked": int(len(group)),
                "inside_central_90_interval": int(group["inside_central_90_interval"].sum()),
                "inside_full_simulated_range": int(group["inside_full_simulated_range"].sum()),
            }
            for name, group in coverage.groupby("stakeholder", sort=False)
        },
    }
    (output / "participant_score_interval_coverage_summary.json").write_text(
        json.dumps(coverage_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_intervals(coverage, output / "Fig5_participant_scores_vs_automatic_intervals")

    print(json.dumps({
        "automatic_equilibrium": median_summary,
        "participant_equilibrium": old_summary,
        "interval_coverage": coverage_summary["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

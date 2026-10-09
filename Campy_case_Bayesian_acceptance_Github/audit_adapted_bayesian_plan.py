"""Audit the integer-adapted all-moderator Bayesian negotiation plan.

Besides plan-level checks, every private prompt that a run will actually load (via its
moderator configuration) is parsed with the same regular expressions the negotiation
script uses.  The audit fails unless the parsed tables equal the adapted profile and
the parsed tables admit a positive-gain Nash bargaining solution at the prompts'
minimum acceptable score.  With --baseline_profile it also reports every option score
that differs from the unadapted prompts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path


UTILITY_RE = re.compile(r"Issue\s+([A-Z])\s*\(max\s+score\s+\d+\)\s*:\s*(.+)", re.I)
OPTION_RE = re.compile(r"\b([A-Z]\d+)\s*\((-?\d+)\)")
# Same patterns as main_campy_bayesian_negotiation.py (copied to avoid importing the agents SDK).
PROMPT_UTILITY_RE = re.compile(r"Issue\s+([A-Z])\s*\(max\s+score\s+([0-9]+(?:\.[0-9]+)?)\)\s*:\s*(.+)", re.I)
PROMPT_OPTION_RE = re.compile(r"\b([A-Z]\d+)\s*\((-?[0-9]+(?:\.[0-9]+)?)\)")
THRESHOLD_RE = re.compile(r"score less than\s+([0-9]+(?:\.[0-9]+)?)", re.I)


def parse_prompt(text: str) -> tuple[dict, float | None]:
    table = {}
    for line in text.splitlines():
        match = PROMPT_UTILITY_RE.search(line.strip())
        if match:
            table[match.group(1).upper()] = {o.upper(): int(float(v)) for o, v in PROMPT_OPTION_RE.findall(match.group(3))}
    threshold = THRESHOLD_RE.search(text)
    return table, (float(threshold.group(1)) if threshold else None)


def nash_solution(tables: dict, threshold: float) -> dict:
    names = list(tables)
    issues = sorted(next(iter(tables.values())))
    options = [sorted(next(iter(tables.values()))[i]) for i in issues]
    best, strict_count = None, 0
    for package in itertools.product(*options):
        utilities = [sum(tables[n][i][o] for i, o in zip(issues, package)) for n in names]
        if min(utilities) > threshold:
            strict_count += 1
            product = 1
            for u in utilities:
                product *= u - threshold
            key = (product, sum(utilities))
            if best is None or key > best[0]:
                best = (key, ", ".join(package), dict(zip(names, utilities)))
    return {
        "threshold_and_disagreement_point": threshold,
        "strict_ir_package_count": strict_count,
        "positive_gain_nbs_exists": best is not None,
        "nbs_package": best[1] if best else None,
        "nbs_utilities": best[2] if best else None,
        "nash_product_above_threshold": best[0][0] if best else None,
    }


def audit_prompts(plan: dict, profile: dict, project: Path, baseline: dict | None, errors: list) -> dict:
    expected = {n: {i: {o: int(v) for o, v in opts.items()} for i, opts in row["utility_table"].items()} for n, row in profile["stakeholders"].items()}
    loaded, thresholds, checked = {}, set(), 0
    for run in plan.get("runs", []):
        config = project / run["configuration"]
        for line in config.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, prompt_path = [x.strip() for x in line.split(",")[:2]]
            table, threshold = parse_prompt((project / prompt_path).read_text(encoding="utf-8"))
            checked += 1
            thresholds.add(threshold)
            if table != expected.get(name):
                errors.append(f"{run['run_id']}:{name}: prompt score table differs from the adapted profile")
            loaded.setdefault(name, table)
    if len(thresholds) != 1 or None in thresholds:
        errors.append(f"Prompts disagree on, or omit, the minimum acceptable score: {sorted(map(str, thresholds))}")
        return {"prompts_checked": checked}
    threshold = thresholds.pop()
    result = nash_solution(loaded, threshold)
    if not result["positive_gain_nbs_exists"]:
        errors.append(f"Final prompt score tables admit no positive-gain NBS at {threshold:g}")
    summary = {"prompts_checked": checked, "prompt_equilibrium": result}
    if baseline is not None:
        changes = []
        for name, row in baseline["stakeholders"].items():
            for issue, opts in row["utility_table"].items():
                for option, value in opts.items():
                    new = loaded[name][issue][option]
                    if int(value) != new:
                        changes.append({"stakeholder": name, "option": option, "baseline": int(value), "final_prompt": new, "change": new - int(value)})
        summary["option_score_changes_vs_baseline"] = changes
        summary["stakeholders_changed"] = sorted({c["stakeholder"] for c in changes})
        summary["baseline_equilibrium"] = nash_solution(
            {n: {i: {o: int(v) for o, v in opts.items()} for i, opts in r["utility_table"].items()} for n, r in baseline["stakeholders"].items()},
            threshold)
    return summary


def portable(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected_runs_per_moderator", type=int, default=2)
    parser.add_argument("--expected_required_acceptances", type=int, default=3,
                        help="Acceptance rule the plan must use (3 = OPM-style; 2 = supplied Campylobacter protocol).")
    parser.add_argument("--baseline_profile", default=None,
                        help="Optional unadapted profile; the audit then lists every changed option score.")
    parser.add_argument("--project_dir", default=".")
    args = parser.parse_args()
    profile_path, plan_path = Path(args.profile).resolve(), Path(args.plan).resolve()
    profile, plan = json.loads(profile_path.read_text()), json.loads(plan_path.read_text())
    errors = []
    names = list(profile["stakeholders"])
    for name, row in profile["stakeholders"].items():
        table = row["utility_table"]
        if sum(max(values.values()) for values in table.values()) != 100:
            errors.append(f"{name}: issue maxima do not sum to 100")
        if any(not isinstance(value, int) or value < 0 for values in table.values() for value in values.values()):
            errors.append(f"{name}: table contains a non-integer or negative score")
    if plan.get("required_acceptances") != args.expected_required_acceptances:
        errors.append(
            f"Plan uses required_acceptances={plan.get('required_acceptances')}, "
            f"expected {args.expected_required_acceptances}"
        )
    if plan.get("total_planned_runs") != len(names) * args.expected_runs_per_moderator:
        errors.append("Unexpected all-moderator run count")
    counts = {name: 0 for name in names}
    for run in plan.get("runs", []):
        counts[run.get("moderator")] = counts.get(run.get("moderator"), 0) + 1
        if run.get("profile_id") != profile.get("profile_id"):
            errors.append(f"{run.get('run_id')}: wrong profile ID")
        for key in ("profile_file", "configuration", "initial_deal_file"):
            if "Prompts_main" in str(run.get(key, "")):
                errors.append(f"{run.get('run_id')}: references an original participant prompt")
    for name, count in counts.items():
        if count != args.expected_runs_per_moderator:
            errors.append(f"{name}: expected {args.expected_runs_per_moderator} moderator runs, found {count}")
    for path in [profile_path, plan_path]:
        if "Case1_Campy_Workshop_Alex_final_scores" in path.read_text():
            errors.append(f"{path.name}: references participant score CSV")
    baseline = json.loads(Path(args.baseline_profile).read_text()) if args.baseline_profile else None
    prompt_check = audit_prompts(plan, profile, Path(args.project_dir).resolve(), baseline, errors)
    report = {"status": "passed" if not errors else "failed", "profile": portable(profile_path), "plan": portable(plan_path), "required_acceptances": plan.get("required_acceptances"), "integer_score_tables": not any("non-integer" in x for x in errors), "runs_by_moderator": counts, "final_prompt_check": prompt_check, "errors": errors}
    output = plan_path.parent / "adapted_bayesian_plan_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()

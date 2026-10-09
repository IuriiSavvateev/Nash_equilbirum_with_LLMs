"""Materialize one integer-adapted score profile for Bayesian negotiations."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def preferred_deal(table: dict[str, dict[str, int]]) -> dict:
    issue_order = sorted(table)
    maxima = {
        issue: sorted(option for option, value in table[issue].items() if value == max(table[issue].values()))
        for issue in issue_order
    }
    return {
        "issue_order": issue_order,
        "maximizing_options_by_issue": maxima,
        "canonical_starting_deal": ", ".join(maxima[issue][0] for issue in issue_order),
        "tie_policy": "lexicographically first maximum within each issue",
    }


def portable(path: Path) -> str:
    """Project-relative path when possible, so outputs stay valid after moving the bundle."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_source_profile(recorded: str, report_path: Path, override: str | None) -> Path:
    """Locate the source profile even when the report stores a path from another machine.

    Earlier cached reports stored an absolute path from the machine that produced them
    (e.g. /workspace/.../automatic_scoring/...), which does not exist in Colab/Drive.
    Try, in order: an explicit override, the recorded path as-is, the recorded path
    relative to the working directory and to the report's project folder, and finally
    every trailing sub-path of the recorded path (so .../Bundle/automatic_scoring/x.json
    resolves to ./automatic_scoring/x.json; shortest match first).
    """
    if override:
        candidate = Path(override)
        if not candidate.exists():
            raise FileNotFoundError(f"--source_profile not found: {candidate}")
        return candidate.resolve()
    recorded_path = Path(recorded)
    roots = [Path.cwd(), report_path.parent.parent, report_path.parent]
    candidates = [recorded_path] + [root / recorded_path for root in roots if not recorded_path.is_absolute()]
    parts = recorded_path.parts
    # Shortest trailing sub-path first, so the bundle's own copy wins over nested/stale copies.
    for start in range(len(parts) - 1, 0, -1):
        tail = Path(*parts[start:])
        candidates.extend(root / tail for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate source profile '{recorded}' recorded in {report_path}. "
        "Pass --source_profile explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="perturbation_outputs/balanced_perturbation_results.json")
    parser.add_argument("--selection", choices=["smallest", "ir_rich"], default="smallest")
    parser.add_argument("--output", default="adapted_profiles/profile_integer_adapted_smallest_genuine_nbs.json")
    parser.add_argument("--source_profile", default=None,
                        help="Optional explicit path to the unperturbed profile; overrides the path stored in the report.")
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_path = resolve_source_profile(report["source_profile"], report_path, args.source_profile)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    key = "smallest_practical_genuine_nbs_profile" if args.selection == "smallest" else "ir_rich_profile"
    selected = report[key]
    profile = copy.deepcopy(source)
    profile["profile_id"] = f"profile_integer_adapted_{args.selection}"
    profile["created_utc"] = datetime.now(timezone.utc).isoformat()
    profile["selection_rule"] = f"integer adaptation selected from {key}"
    profile["integer_adaptation"] = {
        "source_report": portable(report_path),
        "source_profile": portable(source_path),
        "selection": key,
        "adaptation_cost": selected["adaptation_cost"],
        "equilibrium": selected["equilibrium"],
    }
    for name, row in profile["stakeholders"].items():
        table = selected["utility_table"][name]
        weights = selected["issue_weights"][name]
        if any(not isinstance(value, int) for options in table.values() for value in options.values()):
            raise ValueError(f"{name}: non-integer option score in integer adaptation")
        if sum(max(options.values()) for options in table.values()) != 100:
            raise ValueError(f"{name}: adapted issue maxima do not sum to 100")
        row["utility_table"] = table
        row["issue_weights"] = weights
        row["preferred_deal"] = preferred_deal(table)
        row["median_representative_selection"] = {
            "valid_draw_preserved": False,
            "note": "Integer-adapted profile derived from the valid median-representative draw.",
        }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote integer-adapted profile: {output}")


if __name__ == "__main__":
    main()

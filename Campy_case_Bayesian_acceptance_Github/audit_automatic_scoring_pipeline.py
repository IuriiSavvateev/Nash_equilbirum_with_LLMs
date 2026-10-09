"""Audit automatic score generation, prompts, and the prepared run plan."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


UTILITY_LINE_RE = re.compile(
    r"Issue\s+([A-Z])\s*\(max\s+score\s+([0-9]+(?:\.[0-9]+)?)\)\s*:\s*(.+)",
    re.IGNORECASE,
)
OPTION_SCORE_RE = re.compile(r"\b([A-Z]\d+)\s*\((-?[0-9]+(?:\.[0-9]+)?)\)")


def parse_utility_table_from_prompt(prompt: str) -> dict[str, dict[str, float]]:
    result = {}
    for line in prompt.splitlines():
        match = UTILITY_LINE_RE.search(line.strip())
        if match:
            result[match.group(1).upper()] = {
                option.upper(): float(score)
                for option, score in OPTION_SCORE_RE.findall(match.group(3))
            }
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_profile(profile: dict, prior: dict) -> list[str]:
    errors: list[str] = []
    total = int(profile["total_points"])
    for name, row in profile["stakeholders"].items():
        if name not in prior["stakeholders"]:
            errors.append(f"{profile['profile_id']}: unknown stakeholder {name}")
            continue
        stakeholder_prior = prior["stakeholders"][name]
        selection = row.get("median_representative_selection", {})
        if selection.get("valid_draw_preserved") is not True:
            errors.append(f"{profile['profile_id']}:{name}: missing valid median-representative selection record")
        table = row["utility_table"]
        public_catalogue = {
            issue: set(issue_row["options"])
            for issue, issue_row in stakeholder_prior["issues"].items()
        }
        private_catalogue = {issue: set(options) for issue, options in table.items()}
        if public_catalogue != private_catalogue:
            errors.append(f"{profile['profile_id']}:{name}: catalogue mismatch")
        if sum(max(options.values()) for options in table.values()) != total:
            errors.append(f"{profile['profile_id']}:{name}: issue maxima do not sum to {total}")
        for issue, options in table.items():
            weight = int(row["issue_weights"][issue])
            if min(options.values()) < 0 or max(options.values()) != weight:
                errors.append(f"{profile['profile_id']}:{name}:{issue}: invalid option range/max")
        for issue in stakeholder_prior.get("relevant_issues", []):
            if row["issue_weights"][issue] <= 0:
                errors.append(f"{profile['profile_id']}:{name}: relevant issue {issue} inactive")
        for relation in stakeholder_prior.get("issue_relations", []):
            high, low = relation["more_important"], relation["less_important"]
            if row["issue_weights"][high] <= row["issue_weights"][low]:
                errors.append(f"{profile['profile_id']}:{name}: violated issue relation {high}>{low}")
        for relation in stakeholder_prior.get("option_relations", []):
            high, low = relation["preferred"], relation["less_preferred"]
            if table[high[0]][high] <= table[low[0]][low]:
                errors.append(f"{profile['profile_id']}:{name}: violated option relation {high}>{low}")
        deal = row["preferred_deal"]["canonical_starting_deal"].split(", ")
        for option in deal:
            issue = option[0]
            if table[issue][option] != max(table[issue].values()):
                errors.append(f"{profile['profile_id']}:{name}: starting deal is not maximizing")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the prepared Campylobacter single-run pipeline")
    parser.add_argument("--project_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--scoring_dir", default="automatic_scoring")
    parser.add_argument("--prepared_dir", default="automatic_prepared_runs")
    parser.add_argument("--output", default="automatic_pipeline_audit.json")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    scoring = project / args.scoring_dir
    prepared = project / args.prepared_dir
    prior_path = project / "public_priors_Campylobacter.json"
    prior = load(prior_path)
    scoring_manifest = load(scoring / "automatic_scoring_manifest.json")
    plan = load(prepared / "experiment_plan.json")
    errors: list[str] = []
    warnings: list[str] = []

    if prior.get("private_files_used") is not False or prior.get("private_scores_used") is not False:
        errors.append("Public prior does not certify exclusion of private score files")
    if scoring_manifest.get("human_score_tables_used") is not False:
        errors.append("Scoring manifest does not certify exclusion of human score tables")
    if plan.get("human_score_tables_used") is not False:
        errors.append("Run plan does not certify exclusion of human score tables")
    if int(scoring_manifest.get("generated_profiles", 0)) != 1:
        errors.append("Exactly one automatic utility profile is required")
    if int(plan.get("total_planned_runs", 0)) != 1:
        errors.append("Exactly one negotiation run must be planned")
    for filename in (
        "generate_discussion_priors.py",
        "automatic_utility_profiles.py",
        "prepare_automatic_negotiation_runs.py",
        "balanced_perturbation_analysis.py",
    ):
        source = (project / filename).read_text(encoding="utf-8")
        if "reference_participant_scores" in source or "Case1_Campy_Workshop_Alex_final_scores" in source:
            errors.append(f"Generation-stage source references the participant score file: {filename}")

    perturbation_path = project / "perturbation_outputs" / "balanced_perturbation_results.json"
    if not perturbation_path.exists():
        errors.append("Balanced perturbation results are missing")
    else:
        perturbation = load(perturbation_path)
        smallest = perturbation.get("smallest_practical_genuine_nbs_profile", {})
        equilibrium = smallest.get("equilibrium", {})
        if equilibrium.get("positive_gain_nbs_exists") is not True:
            errors.append("Smallest perturbation profile does not produce a positive-gain NBS")
        if int(equilibrium.get("strict_ir_package_count", 0)) < 1:
            errors.append("Smallest perturbation profile has no strictly IR package")
        if perturbation.get("method", {}).get("participant_scores_used") is not False:
            errors.append("Perturbation report does not certify participant-score exclusion")
        if perturbation.get("method", {}).get("integer_score_rule") is None:
            errors.append("Perturbation report does not certify integer-only adapted scores")
        for label in ("smallest_practical_genuine_nbs_profile", "ir_rich_profile"):
            adapted = perturbation.get(label, {})
            for name, table in adapted.get("utility_table", {}).items():
                if any(not isinstance(value, int) or value < 0 for options in table.values() for value in options.values()):
                    errors.append(f"{label}:{name}: contains non-integer or negative adapted score")
                if sum(max(options.values()) for options in table.values()) != 100:
                    errors.append(f"{label}:{name}: adapted issue maxima do not sum to 100")

    profile_count = 0
    for record in scoring_manifest["profiles"]:
        profile_path = scoring / record["path"]
        profile = load(profile_path)
        errors.extend(check_profile(profile, prior))
        profile_count += 1
        prompt_root = prepared / profile["profile_id"] / "private_prompts"
        for name, row in profile["stakeholders"].items():
            candidates = list(prompt_root.glob("*.txt"))
            matches = []
            for path in candidates:
                prompt = path.read_text(encoding="utf-8")
                if f'You represent the "{name}"' in prompt:
                    matches.append((path, prompt))
            if len(matches) != 1:
                errors.append(f"{profile['profile_id']}:{name}: expected one generated prompt")
                continue
            path, prompt = matches[0]
            parsed = parse_utility_table_from_prompt(prompt)
            expected = {
                issue: {option: float(score) for option, score in options.items()}
                for issue, options in row["utility_table"].items()
            }
            if parsed != expected:
                errors.append(f"{profile['profile_id']}:{name}: prompt score parse mismatch in {path}")

    expected_runs = profile_count
    if len(plan["runs"]) != expected_runs or plan["total_planned_runs"] != expected_runs:
        errors.append(f"Run-plan count mismatch: expected {expected_runs}")
    if any(row.get("status") != "prepared_not_run" for row in plan["runs"]):
        warnings.append("At least one plan row is not marked prepared_not_run")
    if any("Prompts_main" in json.dumps(row) for row in plan["runs"]):
        errors.append("A prepared run references the original human-score prompt directory")

    report = {
        "schema": "automatic_pipeline_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "human_score_inputs_used": False if not any("human" in error.casefold() for error in errors) else None,
        "profiles_checked": profile_count,
        "planned_runs_checked": len(plan["runs"]),
        "checks": [
            "public/private option-catalogue equality",
            "integer issue maxima sum to the fixed total",
            "relevant issues remain active",
            "issue and option ordinal relations are preserved",
            "moderator starting deals maximize the sampled private utility",
            "generated prompts round-trip through the negotiation parser",
            "prepared run plan does not reference original human-score prompts",
            "exactly one median-representative profile and one run are prepared",
            "generation-stage source files do not reference the participant score CSV",
            "balanced perturbation output contains a genuine positive-gain NBS and excludes participant scores",
            "selected perturbation landscapes preserve the integer 100-point scoring rule",
        ],
        "warnings": warnings,
        "errors": errors,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = project / output
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit {report['status']}: {output}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()

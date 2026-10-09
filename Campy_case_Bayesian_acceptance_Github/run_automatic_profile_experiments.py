"""Execute a prepared automatic-scoring negotiation plan.

This is deliberately separate from prompt preparation.  It requires the user's
API credentials and is not executed when the bundle is built or validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prepared automatic utility-profile negotiations")
    parser.add_argument("--plan", default="automatic_prepared_runs/experiment_plan.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--trace_sdk_tool_calls_to_langsmith", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "automatic_negotiation_run_plan_v1":
        raise ValueError("Unsupported experiment-plan schema")
    project_dir = plan_path.parent.parent
    runs = list(plan["runs"])
    if args.max_runs is not None:
        if args.max_runs < 1:
            raise ValueError("--max_runs must be positive")
        runs = runs[: args.max_runs]

    execution_rows = []
    result_rows = []
    for row in runs:
        output_dir = Path(row["output_dir"])
        if not output_dir.is_absolute():
            output_dir = project_dir / output_dir
        summary_path = output_dir / "run_summary.json"
        command = list(row["command"])
        command[0] = sys.executable
        if args.trace_sdk_tool_calls_to_langsmith:
            command.append("--trace_sdk_tool_calls_to_langsmith")

        if args.resume and summary_path.exists():
            status = "skipped_completed"
            print(f"SKIP {row['run_id']}")
        elif args.dry_run:
            status = "dry_run"
            print("DRY RUN:", " ".join(command))
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                # A previous attempt crashed before writing run_summary.json: move its
                # partial files aside so they are not mixed with the new attempt.
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                aborted = output_dir.parent / "_aborted_attempts" / f"{output_dir.name}_{stamp}"
                aborted.parent.mkdir(parents=True, exist_ok=True)
                output_dir.rename(aborted)
                print(f"MOVED partial attempt to {aborted}")
            print(f"RUN {row['run_id']}", flush=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, cwd=project_dir)
            if completed.returncode == 0 and summary_path.exists():
                status = "completed"
            else:
                status = "failed"
                print(f"FAILED {row['run_id']} (exit {completed.returncode}); continuing with the next run", flush=True)

        execution_rows.append({**row, "execution_status": status})
        if summary_path.exists() and not args.dry_run:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result = {
                "run_id": row["run_id"],
                "profile_id": row["profile_id"],
                "moderator": row["moderator"],
                "repetition": row["repetition"],
                "seed": row["seed"],
                "initial_deal": row["initial_deal"],
                "outcome": summary.get("outcome", "agreement" if summary.get("proposal_passes") else "no_agreement"),
                "agreement_reached": summary.get("agreement_reached", summary.get("proposal_passes")),
                "main_blocker": (summary.get("main_disagreement") or {}).get("stakeholder", ""),
                "main_blocker_issue": ((summary.get("main_disagreement") or {}).get("largest_issue_losses") or [{}])[0].get("issue", ""),
                "main_blocker_shortfall": (summary.get("main_disagreement") or {}).get("shortfall_to_threshold"),
                "final_deal": summary.get("final_deal", ""),
                "primary_final_deal": summary.get("primary_final_deal", summary.get("final_deal", "")),
                "final_deals": json.dumps(summary.get("final_deals", [summary.get("final_deal", "")])),
                "final_deal_count": summary.get("final_deal_count", 1 if summary.get("final_deal") else 0),
                "accepting_stakeholders": summary.get("accepting_stakeholders"),
                "required_acceptances": summary.get("required_acceptances"),
                "proposal_passes": summary.get("proposal_passes"),
                "majority_rule_passes": summary.get("majority_rule_passes"),
                "unanimous": summary.get("unanimous"),
            }
            for stakeholder_row in summary.get("stakeholders", []):
                key = slug(stakeholder_row["stakeholder"])
                result[f"{key}_score"] = stakeholder_row.get("true_score_evaluation_only")
                result[f"{key}_accepts"] = stakeholder_row.get("accepts_at_known_threshold")
            result_rows.append(result)

    execution_manifest = {
        "schema": "automatic_negotiation_execution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan_path),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "requested_runs": len(runs),
        "runs": execution_rows,
    }
    execution_path = plan_path.parent / "execution_manifest.json"
    execution_path.write_text(
        json.dumps(execution_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if result_rows:
        result_json = plan_path.parent / "automatic_experiment_results.json"
        result_csv = plan_path.parent / "automatic_experiment_results.csv"
        result_json.write_text(json.dumps(result_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        fields = list(dict.fromkeys(key for result in result_rows for key in result))
        with result_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result_rows)
        print(f"Results: {result_csv}")
    failed = [r["run_id"] for r in execution_rows if r["execution_status"] == "failed"]
    if failed:
        print(f"{len(failed)} run(s) failed; rerun with --resume to retry them:", *failed, sep="\n  ")
    print(f"Execution manifest: {execution_path}")


if __name__ == "__main__":
    main()

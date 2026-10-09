# Campylobacter case — automatic scoring, adaptation analysis and adapted Bayesian negotiations

Copy this folder to Google Drive as `MyDrive/MicRisk_agents_SDK/Campy_case_Bayesian` (or change `BUNDLE_NAME` / the Drive path in Step 1) and open `Campylobacter_Fully_Automated_Scoring_Colab.ipynb` in Colab. The bundle contains code and inputs only; every output folder is created when the notebook runs.

## Notebook steps

| Step | What it does | API? |
|---|---|---|
| 1–2 | Mount Drive, locate the bundle, install requirements | no |
| 3 | Check the public case inputs (`public_info_Campylobacter/`) | no |
| 4 | Load the public qualitative priors; `REGENERATE_PRIORS=True` re-codes them with GPT-5.5 | only if regenerating |
| 5 | 20,000 valid integer score tables per stakeholder; median-representative profile | no |
| 6 | Baseline single-run plan (control prompts; input to the Step 9 audit) | no |
| 7 | Equilibrium at 65 and participant-score interval check (old Figs. 4–5) | no |
| 8 | Integer adaptation of the automatic profile (smallest genuine NBS; widens the cap automatically if ±5 is not enough) | no |
| 8b | Same adaptation analysis for the original participant scores | no |
| 9 | Audit of the automatic pipeline | no |
| 10 | Materialise the adapted profile | no |
| 11 | Prepare the all-moderator plan: `RUNS_PER_MODERATOR=7`, `ADAPTED_REQUIRED_ACCEPTANCES=3`, `MAX_TOOL_TURNS=20`; audits the final prompts | no |
| 12 | Load OpenAI / LangSmith keys (`CONFIGURE_API=True`) | — |
| 13 | Run the negotiations (`RUN_ADAPTED_BAYESIAN_NEGOTIATIONS=True`) | yes |

Steps 1–11 run in about 2–3 minutes without any API call. Step 13 runs 4 × `RUNS_PER_MODERATOR` negotiations; uncomment `--max_runs 1` in Step 13 for a single test run first.

Colab secrets used: `api-key-MicRisk` (OpenAI; set `OPENAI_SECRET_NAME` in Step 4 to change it) and `langsmith`.

## Priors — important for reproducibility

`public_priors_Campylobacter.json` in this bundle is the **quote-verified cache** (fixed coding, `build_verified_campy_priors.py`). The results in the current manuscript figures were produced with **GPT-5.5-regenerated priors**. To reproduce those results exactly, copy the `public_priors_Campylobacter.json` from your previous Drive folder over the one in this bundle and keep `REGENERATE_PRIORS=False`. Setting `REGENERATE_PRIORS=True` calls the model again and may give different priors.

## Negotiation outputs

Each run writes `adapted_bayesian_negotiation_outputs/<run_id>/` with the conversation, `bayesian_inference_calls.jsonl`, `semantic_evidence_cache.json` and `run_summary.json`. The summary records:

- `outcome`: `agreement` or `no_agreement` (final proposal below the required acceptances);
- `realized_payoffs`: true scores if agreed, the disagreement payoff (65) for every party otherwise;
- `main_disagreement` and `parties_below_threshold`: the blocking party, its shortfall to 65 and the issues on which the rejected proposal costs it most;
- `voting_history` (interim and final votes) and `turn_events` (speakers that exhausted their tool steps).

A speaker that exhausts `MAX_TOOL_TURNS` answers once more without tools instead of aborting the negotiation. A failed run is logged and the runner continues; `--resume` skips finished runs and moves partial folders of crashed attempts to `_aborted_attempts/`. `adapted_bayesian_prepared_runs/automatic_experiment_results.csv` collects all runs with outcome and blocker columns.

## Figures

`Campylobacter_Figures_1_2.ipynb` builds manuscript Figures 1 and 2 from the outputs (`analysis_outputs/`, `perturbation_outputs/`, `participant_perturbation_outputs/`, `adapted_bayesian_negotiation_outputs/` and a LangSmith `trace-*.json` export of one run placed in this folder). Runs that ended without agreement are shown as "NO AGREEMENT" rows in Fig. 2a.

## Main scripts

- `generate_discussion_priors.py`, `build_verified_campy_priors.py` — public qualitative priors
- `automatic_utility_profiles.py` — score distributions and median-representative profile
- `analyze_scores_and_equilibrium.py` — equilibrium and interval diagnostics
- `balanced_perturbation_analysis.py`, `participant_perturbation_analysis.py`, `create_adapted_integer_profile.py` — adaptation analysis
- `prepare_automatic_negotiation_runs.py`, `audit_adapted_bayesian_plan.py` — prompts, run plan and audits
- `main_campy_bayesian_negotiation.py`, `semantic_bayesian_inference.py`, `build_prompts_cooperative.py` — negotiation and Bayesian opponent model
- `run_automatic_profile_experiments.py` — executes the plan

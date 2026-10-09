# Nash bargaining with LLM agents — *Campylobacter* case

Code, data and LLM negotiation runs for the paper *[title and reference to be added]*.

The pipeline turns plain-text stakeholder statements into a game-theoretic negotiation:

1. **Automatic scoring** — public preliminary-discussion statements → qualitative (ordinal) preferences → score-table distributions and a representative score profile per stakeholder.
2. **Adaptation analysis** — the smallest integer redistribution of issue weights that makes a Nash bargaining solution (NBS) feasible.
3. **Agent negotiation** — four LLM agents negotiate over the adapted profile; each can query a Bayesian model of every other party, built only from public statements.

The case is the setting of a food safety criterion for *Campylobacter* spp. in broiler meat (four stakeholders, five issues, 1,200 possible packages).

---

## Repository structure

| Folder | Purpose | Start here |
|---|---|---|
| [`Campy_case_Bayesian_acceptance_Github/`](Campy_case_Bayesian_acceptance_Github) | Full pipeline: code, public inputs, intermediate outputs and the 28 negotiation runs | `Campylobacter_Fully_Automated_Scoring_Colab.ipynb` |
| [`Figure_1_2_replot/`](Figure_1_2_replot) | Re-plots manuscript Figures 1 and 2 from the stored data (no API key) | `Figures_1_2_from_bundle.ipynb` |
| `LICENSE` | GNU GPL v3.0 | — |

## Quick start

| I want to… | Open | API key? | Time |
|---|---|---|---|
| Re-plot Figures 1 and 2 | `Figure_1_2_replot/Figures_1_2_from_bundle.ipynb` in Colab → *Run all* → upload the zip in the same folder when prompted | no | a few minutes |
| Re-run scoring and adaptation analysis | `Campy_case_Bayesian_acceptance_Github/Campylobacter_Fully_Automated_Scoring_Colab.ipynb`, Steps 1–11 | no | a few minutes |
| Re-run the negotiations | same notebook, Steps 12–13 (`CONFIGURE_API=True`, `RUN_ADAPTED_BAYESIAN_NEGOTIATIONS=True`) | OpenAI (LangSmith optional) | ~10 min per run |
| Inspect a negotiation | unzip `adapted_bayesian_negotiation_outputs/Bayesian_inference_run.zip` in place; see [Run outputs](#run-outputs) | no | — |

Requirements: Python ≥ 3.10 and `Campy_case_Bayesian_acceptance_Github/requirements.txt` (`openai-agents`, `openai`, `langsmith`, `numpy`, `pandas`, `matplotlib`, `scipy`, `pydantic`). Colab secrets for Steps 12–13: `api-key-MicRisk` (OpenAI) and `langsmith` (optional).

---

## 1 · Pipeline folder `Campy_case_Bayesian_acceptance_Github/`

### Notebook steps and scripts

| Step | Script | What it does | Writes to | Paper |
|---|---|---|---|---|
| 1–2 | — | Locate the folder (current directory or Google Drive), install requirements | — | — |
| 3 | — | Check the score-free public inputs (built once with `extract_campy_public_inputs.py`) | `public_info_Campylobacter/` | 2.1 |
| 4 | `generate_discussion_priors.py` | LLM coding of each stakeholder's preliminary statements into ordinal constraints; `REGENERATE_PRIORS=False` uses the stored file | `public_priors_Campylobacter.json` | 2.2 |
| 5 | `automatic_utility_profiles.py` | 20,000 valid integer score tables per stakeholder; representative profile | `automatic_scoring/` | 2.2 |
| 6 | `prepare_automatic_negotiation_runs.py` | Prompts and a single control run for the representative profile | `automatic_prepared_runs/` † | — |
| 7 | `analyze_scores_and_equilibrium.py` | NBS search at threshold 65; participant scores vs automatic p05–p95 intervals | `analysis_outputs/` | 3.1, Fig. 1a |
| 8 | `balanced_perturbation_analysis.py` | Smallest integer adaptation creating an NBS (automatic profile) | `perturbation_outputs/` † | 2.3, Fig. 1b–c |
| 8b | `participant_perturbation_analysis.py` | Same for the participant-derived profile | `participant_perturbation_outputs/` | 2.3, Fig. 1b–c |
| 9 | `audit_automatic_scoring_pipeline.py` | Checks that no participant score or private prompt entered Steps 4–6 | `automatic_pipeline_audit.json` | — |
| 10 | `create_adapted_integer_profile.py` | Adapted profile used in the negotiations | `adapted_profiles/` † | 2.3 |
| 11 | `prepare_automatic_negotiation_runs.py`, `audit_adapted_bayesian_plan.py` | Private prompts, moderator configurations, opening packages, 28-run plan; audit that every prompt reproduces the adapted profile and admits an NBS | `adapted_bayesian_prepared_runs/` † | 2.4 |
| 12 | — | Load API keys | — | — |
| 13 | `run_automatic_profile_experiments.py` → `main_campy_bayesian_negotiation.py` | Run the negotiations (`--resume` skips finished runs) | `adapted_bayesian_negotiation_outputs/` | 2.4–2.5, Fig. 2 |

† Created when the notebook runs; the stored versions are included in the data zip in `Figure_1_2_replot/`.

### Other files

| File | Role |
|---|---|
| `semantic_bayesian_inference.py` | Core module: prior sampler (`sample_public_prior`), evidence extractor (`SemanticEvidenceExtractor`), Bayesian opponent model (`infer_opponent_posterior`) |
| `build_prompts_cooperative.py` | Per-turn prompts with the recent negotiation history |
| `preliminary_discussion_prior_protocol.md` | System prompt for Step 4 |
| `build_verified_campy_priors.py` → `public_priors_Campylobacter_verified_cache.json` | Alternative conservative, quote-verified prior coding; **not used** for the reported results |
| `public_priors_Campylobacter.json` | **Priors used in all reported runs** (GPT-5.5) |
| `requirements.txt` | Python packages |
| `Campylobacter_Figures_1_2.ipynb` | Legacy figure notebook, superseded by `Figure_1_2_replot/` |

### Where LLMs are used

All LLM calls use the OpenAI API with **fixed instructions and a structured output schema** (a Pydantic model passed to `client.responses.parse(..., text_format=…)`, i.e. OpenAI Structured Outputs). No agent "skills" or external prompt libraries are used. The LLMs never output scores, probabilities or confidences.

| Purpose | Model | Instructions | Output schema | Code | Validation |
|---|---|---|---|---|---|
| Ordinal priors from preliminary statements | GPT-5.5 | `preliminary_discussion_prior_protocol.md` | `PriorAssessment` | `generate_discussion_priors.py` | IDs exist; within-issue comparisons; duplicates and cycles dropped; quote and explanation required (`validate_assessment`) |
| Evidence from public negotiation statements | GPT-5.5 (`--semantic_model`) | `EVIDENCE_INSTRUCTIONS` | `PublicMessageEvidence` | `SemanticEvidenceExtractor.extract()` in `semantic_bayesian_inference.py` | Kept only if the verbatim quote occurs in the statement and the options exist; conditional items stored but not used |
| Negotiating agents | GPT-5.5, reasoning effort high | private prompts in `adapted_bayesian_prepared_runs/…/private_prompts/` + tool instructions in `NegotiationAgent.build_sdk_agent()` | tagged text (`<SCRATCHPAD>`, `<ANSWER>`, `<DEAL>`, `<QUESTION>`) | OpenAI Agents SDK in `main_campy_bayesian_negotiation.py` | Private tool results and scores must not appear in public answers |

### Methods as implemented

| Component | Implementation | Where |
|---|---|---|
| Case and NBS | 5 issues (4, 5, 5, 4, 3 options), 100-point integer budgets, utility = sum of option scores; disagreement point and acceptance threshold 65; NBS = package maximising Π(uᵢ − 65) with every uᵢ > 65 | `analyze_scores_and_equilibrium.py` |
| Score sampling | Active issues uniform over admissible subsets (relevant issues always active); Dirichlet(1) issue weights and Uniform(0, 1) option fractions, ordered to satisfy the extracted relations; rounded to integers, draws that break a relation rejected | `sample_public_prior`, `quantize_one_sample` |
| Representative profile | Valid draw with the smallest mean absolute deviation from the componentwise medians | `median_representative` |
| Adaptation | Integer transfers between active issue weights; order-preserving integer rescaling of option scores; zero-weight issues fixed; exact search for every party ≥ 66; minimise largest change, then sum of squared changes | `balanced_perturbation_analysis.py` |
| Negotiation protocol | 4 agents; rotating moderator (7 runs each, 28 runs, seeds 1729–1756); 12 public statements, each agent sees the last 6; interim package after statement 6, final package after statement 12 | `main_campy_bayesian_negotiation.py`; commands in `adapted_bayesian_prepared_runs/execution_manifest.json` † |
| Votes | Computed, not generated: a party accepts if its true score ≥ 65; a package passes with ≥ 3 acceptances | `conduct_deterministic_vote` |
| Agent tools | `calculate_my_deal_score` (own score), `infer_opponent_utility` (Bayesian opponent model) | `make_negotiation_tools` |
| Opponent-model prior | One model per party, shared by all agents; 12,000 continuous draws from the same sampler, seeded by run, party and prior hash | `infer_opponent_posterior` |
| Evidence and likelihood | Non-conditional comparisons and package stances from the party's own statements; deterministic truthfulness (a draw survives only if consistent with all evidence); recomputed from the prior at each query; if no draw survives, the prior is returned | `collect_live_evidence`, `_posterior_mask` |
| Returned to the agent | Posterior means and 90 % credible intervals; P(score ≥ 65) for a candidate package; one question with the largest expected information gain (binary entropy), chosen from "accept this package?" and all not-yet-compared option pairs | `infer_opponent_posterior` |

### Data folders

| Folder / file | Content |
|---|---|
| `public_info_Campylobacter/` | Case description, score-free issue catalogue, preliminary discussion |
| `automatic_scoring/` | Score ranges and the representative profile |
| `analysis_outputs/` | Equilibria at 65; participant-score interval coverage |
| `participant_perturbation_outputs/` | Adaptation of the participant-derived profile |
| `adapted_bayesian_negotiation_outputs/Bayesian_inference_run.zip` | The 28 negotiation runs (unzip in place) |
| `automatic_pipeline_audit.json` | Result of the Step 9 audit |

### Run outputs

Each run folder `adapted_bayesian_negotiation_outputs/profile_integer_adapted_smallest_<moderator>_moderator_run_<n>_seed_<seed>/` contains:

| File | Content |
|---|---|
| `run_summary.json` | Outcome, final package, votes, payoffs, blocking party and its shortfall |
| `answers*.json` | Public record as seen by the agents (statements, proposals, public votes) |
| `full_conversation*.json` | Complete record including each agent's private `<SCRATCHPAD>`, true scores and private vote details |
| `bayesian_inference_calls.jsonl` | Every opponent-model query with the evidence used and the result |
| `semantic_evidence_cache.json` | Extractor output for each public statement |

---

## 2 · Figure folder `Figure_1_2_replot/`

| File | Content |
|---|---|
| `Figures_1_2_from_bundle.ipynb` | Regenerates Figures 1 and 2; all analysis code is inside the notebook |
| `Campy_case_Bayesian_acceptance_Github-…zip` | Data bundle read by the notebook (complete copy of the pipeline folder, runs unzipped) |

| Notebook section | What happens |
|---|---|
| 0 Load the bundle | Upload or point to the zip; extract; check that all inputs exist |
| 1 Figure 1 | (a) participant scores vs automatic intervals; (b) smallest adaptation; (c) minimum total transfer per package (cap 30 points per stakeholder; cached) |
| 2 Figure 2 | Checks the priors reproduce every 7th logged opponent-model call exactly; (a) final deals vs NBS; (b) equilibrium landscape; (c) classification of statements and questions; (d) learning curve; (e–f) start vs final vote with paired t-tests |
| Download | Zips `figure_outputs/` (PNG 300 dpi, PDF, CSV tables) |

Settings (cell 0.1): `BUNDLE_ZIP`/`BUNDLE_DIR` (data source), `PRIORS_FILE`, `INCLUDE_LAST_STATEMENT`, `EXCLUDE_NBS_STANCES`, `VIEW_ELEV`/`VIEW_AZIM`. Colours and fonts can be changed in cells 2.2–2.3 (Fig. 2) and 1.1 (Fig. 1).

---

## Reproducibility

* All sampling is seeded (SHA-256 of seed, stakeholder and source hash), so prior draws and every opponent-model state are reproduced exactly from the stored files; the figure notebook verifies this before plotting.
* LLM outputs (priors with `REGENERATE_PRIORS=True`, negotiations) are not deterministic; the stored files are those reported in the paper.

## Software

OpenAI API (Structured Outputs), OpenAI Agents SDK, Pydantic, LangSmith (optional tracing), NumPy, pandas, SciPy, Matplotlib.

## License and citation

GNU GPL v3.0 (see `LICENSE`). Please cite: *[paper reference to be added]*.

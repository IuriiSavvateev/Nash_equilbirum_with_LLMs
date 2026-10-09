"""Ordinal-prior Bayesian inference for live negotiations.

The prior is a maximum-entropy distribution over structurally valid additive
utility tables, conditioned on qualitative relations extracted from preliminary
public stakeholder discussion. Live updating conditions that prior on explicit,
unconditional public comparisons and deal acceptances/rejections.

No proposal pseudo-counts, LLM confidence values, likelihood temperatures, or
private opponent scores are used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel


class ComparisonEvidence(BaseModel):
    preferred: str
    less_preferred: str
    conditional: bool
    evidence_quote: str


class DealEvidence(BaseModel):
    deal: list[str]
    stance: Literal["accept", "reject", "conditional"]
    evidence_quote: str


class PublicMessageEvidence(BaseModel):
    comparisons: list[ComparisonEvidence]
    deal_stances: list[DealEvidence]
    unresolved_conditions: list[str]


EVIDENCE_INSTRUCTIONS = """You extract preference evidence from one public
negotiation message. Return only the speaker's own explicit claims. Do not infer
preferences merely because the speaker proposes a package, repeats another
party's view, asks a question, or mentions an option. Extract a comparison only
when the speaker explicitly prefers one option to another option from the same
issue. Extract a deal stance only when the speaker explicitly accepts or rejects
a complete deal. Mark qualified statements as conditional. Copy a short verbatim
evidence quote for every item. Do not output scores, probabilities, or confidence.
"""


def _stable_seed(*parts: Any) -> int:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _entropy_binary(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _canonical_deal(options: list[str], issue_order: list[str]) -> tuple[str, ...] | None:
    mapping = {str(x).upper()[0]: str(x).upper() for x in options if str(x)}
    if set(mapping) != set(issue_order):
        return None
    return tuple(mapping[issue] for issue in issue_order)


class SemanticEvidenceExtractor:
    def __init__(self, model: str, cache_file: str | Path | None = None, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.cache_file = Path(cache_file) if cache_file else None
        self.cache: dict[str, Any] = {}
        if self.cache_file and self.cache_file.exists():
            try:
                self.cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        if enabled:
            from openai import OpenAI
            self.client = OpenAI()
        else:
            self.client = None

    def _save(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def extract(self, speaker: str, message: str, stakeholder_prior: dict) -> dict:
        catalog = {
            issue: sorted(row["options"])
            for issue, row in stakeholder_prior["issues"].items()
        }
        key = hashlib.sha256(
            json.dumps([self.model, speaker, message, catalog], sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if not self.enabled:
            result = {"comparisons": [], "deal_stances": [], "unresolved_conditions": [], "disabled": True}
            self.cache[key] = result
            self._save()
            return result

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": EVIDENCE_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        f"Speaker: {speaker}\nValid catalogue: {json.dumps(catalog)}\n"
                        f"Public message:\n{message}"
                    ),
                },
            ],
            text_format=PublicMessageEvidence,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"Semantic evidence extraction failed for {speaker}")

        all_options = {option for options in catalog.values() for option in options}
        comparisons = []
        for item in parsed.comparisons:
            preferred, less = item.preferred.upper(), item.less_preferred.upper()
            if (
                preferred in all_options
                and less in all_options
                and preferred[0] == less[0]
                and preferred != less
                and _norm(item.evidence_quote) in _norm(message)
            ):
                comparisons.append({**item.model_dump(), "preferred": preferred, "less_preferred": less})

        issues = sorted(catalog)
        deal_stances = []
        for item in parsed.deal_stances:
            deal = _canonical_deal(item.deal, issues)
            if deal and all(option in all_options for option in deal) and _norm(item.evidence_quote) in _norm(message):
                deal_stances.append({**item.model_dump(), "deal": list(deal)})

        result = {
            "comparisons": comparisons,
            "deal_stances": deal_stances,
            "unresolved_conditions": parsed.unresolved_conditions,
        }
        self.cache[key] = result
        self._save()
        return result


def load_prior_bundle(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "semantic_ordinal_prior_v1":
        raise ValueError("Unsupported or missing prior schema")
    if payload.get("private_files_used") is not False:
        raise ValueError("Prior manifest does not certify that private files were excluded")
    return payload


def _valid_active_sets(issues: list[str], relevant: set[str], edges: list[tuple[str, str]]) -> list[tuple[str, ...]]:
    valid = []
    for bits in itertools.product((False, True), repeat=len(issues)):
        active = {issue for issue, bit in zip(issues, bits) if bit}
        if not active or not relevant.issubset(active):
            continue
        # If a less-important issue is active while the more-important one is not,
        # the strict ordering cannot hold.
        if any(low in active and high not in active for high, low in edges):
            continue
        valid.append(tuple(issue for issue in issues if issue in active))
    if not valid:
        raise ValueError("Public prior constraints permit no active-issue configuration")
    return valid


def _valid_permutations(items: tuple[str, ...], edges: list[tuple[str, str]]) -> list[tuple[str, ...]]:
    result = []
    for permutation in itertools.permutations(items):
        position = {item: idx for idx, item in enumerate(permutation)}
        if all(high not in position or low not in position or position[high] < position[low] for high, low in edges):
            result.append(permutation)
    if not result:
        raise ValueError(f"Cyclic or impossible ordinal constraints for {items}")
    return result


def sample_public_prior(stakeholder_prior: dict, n_samples: int, seed: int) -> dict:
    """Sample the maximum-entropy structural prior under ordinal constraints.

    Unknown issue activity patterns are uniform over all non-empty valid subsets.
    This is equivalent to independent Bernoulli(0.5) activity before conditioning.
    Active issue weights have a symmetric Dirichlet(1) distribution conditioned
    on the extracted order relations. Option ratios are exchangeable Uniform(0,1)
    draws conditioned on extracted within-issue order relations and normalized so
    one option reaches the issue maximum.
    """
    issues = sorted(stakeholder_prior["issues"])
    relevant = set(stakeholder_prior.get("relevant_issues", []))
    issue_edges = [
        (row["more_important"], row["less_important"])
        for row in stakeholder_prior.get("issue_relations", [])
    ]
    valid_sets = _valid_active_sets(issues, relevant, issue_edges)
    rng = np.random.default_rng(seed)
    selected_sets = [valid_sets[i] for i in rng.integers(0, len(valid_sets), size=n_samples)]
    total = float(stakeholder_prior["total_points"])

    issue_weights = {issue: np.zeros(n_samples, dtype=float) for issue in issues}
    for active in sorted(set(selected_sets)):
        indices = np.array([i for i, value in enumerate(selected_sets) if value == active], dtype=int)
        k = len(active)
        raw = rng.dirichlet(np.ones(k), size=len(indices))
        permutations = _valid_permutations(active, issue_edges)
        chosen = rng.integers(0, len(permutations), size=len(indices))
        sorted_values = np.sort(raw, axis=1)[:, ::-1] * total
        for row_idx, sample_idx in enumerate(indices):
            for rank, issue in enumerate(permutations[chosen[row_idx]]):
                issue_weights[issue][sample_idx] = sorted_values[row_idx, rank]

    scores: dict[str, dict[str, np.ndarray]] = {}
    for issue in issues:
        options = tuple(sorted(stakeholder_prior["issues"][issue]["options"]))
        option_edges = [
            (row["preferred"], row["less_preferred"])
            for row in stakeholder_prior.get("option_relations", [])
            if row["preferred"].startswith(issue) and row["less_preferred"].startswith(issue)
        ]
        permutations = _valid_permutations(options, option_edges)
        chosen = rng.integers(0, len(permutations), size=n_samples)
        raw = np.sort(rng.random((n_samples, len(options))), axis=1)[:, ::-1]
        raw[:, 0] = 1.0
        scores[issue] = {option: np.zeros(n_samples, dtype=float) for option in options}
        for sample_idx in range(n_samples):
            for rank, option in enumerate(permutations[chosen[sample_idx]]):
                scores[issue][option][sample_idx] = issue_weights[issue][sample_idx] * raw[sample_idx, rank]

    return {"issues": issues, "issue_weights": issue_weights, "scores": scores, "n_samples": n_samples}


def _deal_scores(samples: dict, deal: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(samples["n_samples"], dtype=float)
    for issue, option in zip(samples["issues"], deal):
        result += samples["scores"][issue][option]
    return result


def collect_live_evidence(
    answers_history: dict,
    opponent_name: str,
    stakeholder_prior: dict,
    extractor: SemanticEvidenceExtractor,
) -> list[dict]:
    observations = []
    seen = set()
    for round_index, row in enumerate(answers_history.get("rounds", [])):
        if not isinstance(row, (list, tuple)) or len(row) < 2 or str(row[0]).casefold() != opponent_name.casefold():
            continue
        extracted = extractor.extract(opponent_name, str(row[1]), stakeholder_prior)
        for comparison in extracted["comparisons"]:
            if comparison.get("conditional"):
                continue
            key = ("comparison", comparison["preferred"], comparison["less_preferred"])
            if key not in seen:
                seen.add(key)
                observations.append({"type": "comparison", "round": round_index, **comparison})
        for stance in extracted["deal_stances"]:
            if stance["stance"] == "conditional":
                continue
            key = ("deal_stance", tuple(stance["deal"]), stance["stance"])
            if key not in seen:
                seen.add(key)
                observations.append({"type": "deal_stance", "round": round_index, **stance})
    return observations


def _posterior_mask(samples: dict, observations: list[dict], threshold: float) -> tuple[np.ndarray, list[dict]]:
    mask = np.ones(samples["n_samples"], dtype=bool)
    used = []
    for obs in observations:
        if obs["type"] == "comparison":
            issue = obs["preferred"][0]
            compatible = samples["scores"][issue][obs["preferred"]] > samples["scores"][issue][obs["less_preferred"]]
        else:
            deal = tuple(obs["deal"])
            values = _deal_scores(samples, deal)
            compatible = values >= threshold if obs["stance"] == "accept" else values < threshold
        mask &= compatible
        used.append(obs)
    return mask, used


def infer_opponent_posterior(
    *,
    opponent_name: str,
    answers_history: dict,
    prior_bundle: dict,
    extractor: SemanticEvidenceExtractor,
    candidate_deal: str = "",
    n_samples: int = 12000,
    seed: int = 0,
    prior_mode: Literal["semantic", "structural"] = "semantic",
) -> dict:
    stakeholders = prior_bundle["stakeholders"]
    match = next((name for name in stakeholders if name.casefold() == opponent_name.casefold()), None)
    if match is None:
        return {"status": "unknown_opponent", "opponent": opponent_name}
    prior = dict(stakeholders[match])
    if prior_mode == "structural":
        prior = {
            **prior,
            "relevant_issues": [],
            "issue_relations": [],
            "option_relations": [],
        }
    samples = sample_public_prior(prior, n_samples, _stable_seed(seed, match, prior["source_sha256"]))
    observations = collect_live_evidence(answers_history, match, prior, extractor)
    threshold = float(prior["acceptance_threshold"])
    mask, used = _posterior_mask(samples, observations, threshold)
    conflict = bool(observations and not mask.any())
    if conflict:
        # Under the declared deterministic-truthfulness likelihood, contradictory
        # observations imply model misspecification. Keep the prior and report the
        # conflict instead of silently inserting an arbitrary lapse probability.
        mask[:] = True
    selected = np.flatnonzero(mask)

    estimated = {}
    intervals = {}
    issue_weights = {}
    for issue in samples["issues"]:
        w = samples["issue_weights"][issue][selected]
        issue_weights[issue] = {
            "mean": round(float(np.mean(w)), 3),
            "p05": round(float(np.quantile(w, 0.05)), 3),
            "p95": round(float(np.quantile(w, 0.95)), 3),
            "probability_active": round(float(np.mean(w > 0)), 4),
        }
        estimated[issue] = {}
        intervals[issue] = {}
        for option, array in samples["scores"][issue].items():
            values = array[selected]
            estimated[issue][option] = round(float(np.mean(values)), 3)
            intervals[issue][option] = {
                "p05": round(float(np.quantile(values, 0.05)), 3),
                "p95": round(float(np.quantile(values, 0.95)), 3),
            }

    parsed_candidate = {token.strip().upper()[0]: token.strip().upper() for token in candidate_deal.replace("<DEAL>", "").replace("</DEAL>", "").split(",") if token.strip()}
    candidate_tuple = tuple(parsed_candidate.get(issue, "") for issue in samples["issues"])
    valid_candidate = all(candidate_tuple) and all(candidate_tuple[i] in samples["scores"][issue] for i, issue in enumerate(samples["issues"]))
    candidate_summary = None
    candidates_for_question = []
    if valid_candidate:
        values = _deal_scores(samples, candidate_tuple)[selected]
        p_accept = float(np.mean(values >= threshold))
        candidate_summary = {
            "deal": list(candidate_tuple),
            "posterior_mean": round(float(np.mean(values)), 3),
            "p05": round(float(np.quantile(values, 0.05)), 3),
            "p95": round(float(np.quantile(values, 0.95)), 3),
            "probability_meeting_threshold": round(p_accept, 4),
            "threshold": threshold,
        }
        candidates_for_question.append({
            "kind": "candidate_acceptance",
            "information_gain_bits": _entropy_binary(p_accept),
            "question_tag": (
                f'<QUESTION target="{match}">Would you accept the complete package '
                f'{", ".join(candidate_tuple)}?</QUESTION>'
            ),
        })

    already_compared = {
        frozenset((obs["preferred"], obs["less_preferred"]))
        for obs in observations if obs["type"] == "comparison"
    }
    for issue in samples["issues"]:
        options = sorted(samples["scores"][issue])
        for a, b in itertools.combinations(options, 2):
            if frozenset((a, b)) in already_compared:
                continue
            p = float(np.mean(samples["scores"][issue][a][selected] > samples["scores"][issue][b][selected]))
            candidates_for_question.append({
                "kind": "option_comparison",
                "issue": issue,
                "options": [a, b],
                "information_gain_bits": _entropy_binary(p),
                "question_tag": (
                    f'<QUESTION target="{match}" option="{a}">On issue {issue}, '
                    f'do you prefer {a} or {b}?</QUESTION>'
                ),
            })
    candidates_for_question.sort(key=lambda row: (-row["information_gain_bits"], row["question_tag"]))

    return {
        "status": "evidence_conflict_prior_returned" if conflict else "posterior_estimated",
        "opponent": match,
        "model": "maximum-entropy ordinal prior with deterministic truthful-statement conditioning",
        "proposal_likelihood_used": False,
        "llm_confidence_used": False,
        "private_opponent_scores_used": False,
        "prior_source": prior["source_file"],
        "prior_model": prior_bundle.get("model"),
        "prior_mode": prior_mode,
        "posterior_samples": n_samples,
        "surviving_samples": int(len(selected)),
        "surviving_fraction": round(float(len(selected) / n_samples), 6),
        "observations_used": used,
        "estimated_utility_function": estimated,
        "credible_intervals_90": intervals,
        "issue_weights": issue_weights,
        "candidate_deal_distribution": candidate_summary,
        "recommended_public_questions": candidates_for_question[:1],
        "caution": (
            "The prior comes from preliminary-discussion semantics rather than historical calibration. "
            "The posterior is conditional on the explicit-statement truthfulness assumption."
        ),
    }

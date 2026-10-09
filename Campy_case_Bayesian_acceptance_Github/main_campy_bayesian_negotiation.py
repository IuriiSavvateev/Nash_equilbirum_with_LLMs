"""OpenAI Agents SDK + LangSmith cooperative Campylobacter negotiation simulation.

This script keeps the original project file format for the output:
- initial_prompts_file: CSV-like lines: agent_name,path_to_prompt,role1&role2,model
- initial_deal_file: one line containing a deal such as A1,B2,C1,D1,E2

Environment variables:
- OPENAI_API_KEY=...
- LANGSMITH_TRACING=true                  # for LangSmith outer traces
- LANGSMITH_API_KEY=...                   # for LangSmith outer traces
- LANGSMITH_PROJECT=negotiation-sdk       # optional but recommended
- OPENAI_AGENTS_DISABLE_TRACING=1         # optional: disable OpenAI Agents SDK traces
- OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false  # recommended if prompts contain private data

Install:
    pip install openai-agents langsmith numpy

Run example:
    python main_cooperative_agents_sdk_langsmith.py \
      --agents_num 5 \
      --rounds 12 \
      --window_size 6 \
      --votes_interval 6 \
      --initial_prompts_file initial_prompts_base/initial_prompts.txt \
      --initial_deal_file initial_prompts_base/initial_deal.txt \
      --output_dir ./output/ \
      --sleep_seconds 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from agents import Agent, ModelSettings, Runner, flush_traces, function_tool, trace
from agents.exceptions import MaxTurnsExceeded
import dataclasses
try:
    from agents import set_trace_processors
    from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
except Exception:  # keep the script runnable if integration package/version is missing
    set_trace_processors = None
    OpenAIAgentsTracingProcessor = None
from langsmith import traceable

from build_prompts_cooperative import (
    build_deal_suggestion_prompt,
    build_first_slot,
    build_slot_prompts,
)
from semantic_bayesian_inference import (
    SemanticEvidenceExtractor,
    infer_opponent_posterior as infer_semantic_opponent_posterior,
    load_prior_bundle,
)


# -----------------------------
# Parsing helpers
# -----------------------------

_TAG_RE = {
    "scratchpad": re.compile(r"<SCRATCHPAD>(.*?)</SCRATCHPAD>", re.DOTALL | re.IGNORECASE),
    "answer": re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL | re.IGNORECASE),
    "plan": re.compile(r"<PLAN>(.*?)</PLAN>", re.DOTALL | re.IGNORECASE),
}

# Plot-service compatibility helpers.
# plot_service.create_deals_dataframe() splits deal strings with split(', '),
# so every saved <DEAL> block must use comma + single-space separation.
_PLOT_SERVICE_DEAL_TAG_RE = re.compile(r"<DEAL>(.*?)</DEAL>", re.DOTALL | re.IGNORECASE)
_PLOT_SERVICE_OPTION_RE = re.compile(r"\b([A-Z])(\d+)\b", re.IGNORECASE)
_PUBLIC_SCORE_CLAIM_RE = re.compile(
    r"\b(score|utility|threshold|points?)\s*(?:is|of|=|:)?\s*(-?[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_PUBLIC_POINTS_RE = re.compile(r"\b(-?[0-9]+(?:\.[0-9]+)?)\s+points?\b", re.IGNORECASE)
_PUBLIC_OPTION_SCORE_RE = re.compile(r"\b([A-Z]\d+)\s*\((-?[0-9]+(?:\.[0-9]+)?)\)", re.IGNORECASE)


def normalize_deal_options_for_plot_service(deal_content: str) -> str:
    """Normalize only the inside of a <DEAL>...</DEAL> block.

    Examples:
        A1,B1,C3 -> A1, B1, C3
        A1; B1; C3 -> A1, B1, C3
        A1 B1 C3 -> A1, B1, C3

    This keeps the existing plot_service.py unchanged.
    """
    options = [f"{issue.upper()}{num}" for issue, num in _PLOT_SERVICE_OPTION_RE.findall(deal_content or "")]
    if options:
        return ", ".join(options)

    # Fallback for unusual content: normalize comma spacing without deleting text.
    parts = [part.strip() for part in (deal_content or "").split(",") if part.strip()]
    return ", ".join(parts)


def normalize_deal_text_for_plot_service(text: str) -> str:
    """Normalize every <DEAL>...</DEAL> block inside text."""
    if not text:
        return text or ""

    def replace_deal(match: re.Match) -> str:
        return "<DEAL>" + normalize_deal_options_for_plot_service(match.group(1)) + "</DEAL>"

    return _PLOT_SERVICE_DEAL_TAG_RE.sub(replace_deal, text)


def normalize_plain_deal_for_plot_service(deal: str) -> str:
    """Normalize a plain deal string that may not have <DEAL> tags."""
    return normalize_deal_options_for_plot_service(deal)


def normalize_json_deals_for_plot_service(obj: Any) -> Any:
    """Recursively normalize <DEAL> blocks in JSON-like objects."""
    if isinstance(obj, str):
        return normalize_deal_text_for_plot_service(obj)
    if isinstance(obj, list):
        return [normalize_json_deals_for_plot_service(item) for item in obj]
    if isinstance(obj, dict):
        return {key: normalize_json_deals_for_plot_service(value) for key, value in obj.items()}
    return obj


def _first_tag(text: str, tag_name: str) -> str:
    match = _TAG_RE[tag_name].search(text or "")
    return match.group(1).strip() if match else ""


def extract_public_answer(raw_text: str, *, allow_safe_unwrapped: bool = False) -> str:
    """Return only the public answer shared with the other agents.

    ``allow_safe_unwrapped`` exists only for the opening turn.  That prompt has
    no private scratchpad, but older prompt versions forgot to request an
    ``<ANSWER>`` wrapper.  The fallback remains fail-closed if private tags are
    present and requires a syntactically valid deal.
    """
    answer = _first_tag(raw_text, "answer")
    if answer:
        answer = _TAG_RE["scratchpad"].sub("", answer)
        answer = _TAG_RE["plan"].sub("", answer)
        answer = answer.replace(
            "I believe this proposal balances the interests of all parties involved.", ""
        ).strip()
        answer = _PUBLIC_OPTION_SCORE_RE.sub(r"\1 ([REDACTED])", answer)
        answer = _PUBLIC_SCORE_CLAIM_RE.sub(r"\1 [REDACTED]", answer)
        answer = _PUBLIC_POINTS_RE.sub("[REDACTED] points", answer)
        return normalize_deal_text_for_plot_service(answer)

    if allow_safe_unwrapped:
        has_private_tag = bool(
            _TAG_RE["scratchpad"].search(raw_text or "")
            or _TAG_RE["plan"].search(raw_text or "")
        )
        has_deal = bool(_PLOT_SERVICE_DEAL_TAG_RE.search(raw_text or ""))
        if has_deal and not has_private_tag:
            answer = (raw_text or "").strip()
            answer = _PUBLIC_OPTION_SCORE_RE.sub(r"\1 ([REDACTED])", answer)
            answer = _PUBLIC_SCORE_CLAIM_RE.sub(r"\1 [REDACTED]", answer)
            answer = _PUBLIC_POINTS_RE.sub("[REDACTED] points", answer)
            return normalize_deal_text_for_plot_service(answer)

    # Fail closed. Returning the raw model output could expose an unclosed private
    # scratchpad or score calculation to the other negotiating agents.
    return "[FORMAT ERROR: no valid public <ANSWER> block was produced.]"


def extract_plan(raw_text: str) -> str:
    """Return the private plan used in that agent's next prompt."""
    return _first_tag(raw_text, "plan")


def safe_usage_from_result(result: Any) -> Dict[str, Any]:
    """Best-effort token usage extraction from an Agents SDK RunResult.

    The Agents SDK exposes final_output, raw_responses, new_items, etc. Usage fields
    may differ across SDK/model versions, so this avoids assuming one exact schema.
    """
    usage: Dict[str, Any] = {}
    raw_responses = getattr(result, "raw_responses", None) or []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for response in raw_responses:
        candidate = getattr(response, "usage", None)
        if candidate is None and isinstance(response, dict):
            candidate = response.get("usage")
        if candidate is None:
            continue
        if hasattr(candidate, "model_dump"):
            candidate = candidate.model_dump()
        if not isinstance(candidate, dict):
            continue

        for key in totals:
            value = candidate.get(key)
            if isinstance(value, int):
                totals[key] += value

    usage.update({k: v for k, v in totals.items() if v})
    return usage


# -----------------------------
# Model-settings helpers
# -----------------------------

def is_reasoning_model(model_name: str) -> bool:
    """Return True for model families where reasoning.effort is the main tuning knob."""
    lower = (model_name or "").lower()
    return lower.startswith(("gpt-5", "o1", "o3", "o4"))


def build_model_settings(
    *,
    model_name: str,
    temperature: float,
    reasoning_effort: str | None = "xhigh",
    verbosity: str | None = None,
) -> ModelSettings:
    """Create ModelSettings that work for both GPT-4-style and GPT-5-style models.

    GPT-5.5 uses reasoning.effort as the main quality/latency control. In reasoning
    mode, temperature may be rejected by the Responses API, so this helper only sends
    temperature when reasoning is disabled or when the model is not a reasoning model.

    reasoning_effort:
        - "low", "medium", "high", "xhigh" for GPT-5.5 reasoning
        - "none" to disable reasoning and allow temperature, where supported
        - "default" or empty string to omit reasoning settings entirely
    verbosity:
        - "low", "medium", or "high", if supported by your SDK/model
    """
    effort = (reasoning_effort or "").strip().lower()
    verbosity_value = (verbosity or "").strip().lower() or None
    if verbosity_value == "default":
        verbosity_value = None

    kwargs: Dict[str, Any] = {}

    if is_reasoning_model(model_name):
        if effort in {"low", "medium", "high", "xhigh"}:
            kwargs["reasoning"] = {"effort": effort}
            # Do not send temperature with active GPT-5.5 reasoning.
        elif effort == "none":
            # Some model/API combinations support explicit none; others prefer no reasoning field.
            # If this errors in your SDK version, use --reasoning_effort default instead.
            kwargs["reasoning"] = {"effort": "none"}
            kwargs["temperature"] = temperature
        else:
            # default/empty: omit reasoning and temperature for maximum compatibility.
            pass
    else:
        kwargs["temperature"] = temperature

    if verbosity_value in {"low", "medium", "high"}:
        kwargs["verbosity"] = verbosity_value

    return ModelSettings(**kwargs)


def configure_openai_agents_langsmith_tracing(enabled: bool = True) -> None:
    """Send native OpenAI Agents SDK spans to LangSmith.

    The @traceable decorators around the orchestrator show only your Python-level
    workflow functions. This processor forwards the SDK's internal spans too,
    including LLM generations, tool calls, tool inputs, and tool outputs.
    """
    if not enabled:
        return
    if set_trace_processors is None or OpenAIAgentsTracingProcessor is None:
        print(
            "[warning] LangSmith OpenAI Agents SDK tracing processor is not available. "
            "Upgrade langsmith/openai-agents if you want tool-call spans in LangSmith."
        )
        return
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    print("[info] OpenAI Agents SDK internal spans will be sent to LangSmith.")


# -----------------------------
# Negotiation utility helpers
# -----------------------------

_OPTION_RE = re.compile(r"\b([A-Z])(\d+)\b")
_DEAL_RE = re.compile(r"<DEAL>(.*?)</DEAL>", re.DOTALL | re.IGNORECASE)
_UTILITY_LINE_RE = re.compile(
    r"Issue\s+([A-Z])\s*\(max\s+score\s+([0-9]+(?:\.[0-9]+)?)\)\s*:\s*(.+)",
    re.IGNORECASE,
)
_OPTION_SCORE_RE = re.compile(r"\b([A-Z]\d+)\s*\((-?[0-9]+(?:\.[0-9]+)?)\)")
_THRESHOLD_RE = re.compile(r"score less than\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_PASS_RE = re.compile(r"proposal will pass if at least\s+([0-9]+)\s+part", re.IGNORECASE)


def parse_utility_table_from_prompt(prompt: str) -> Dict[str, Dict[str, float]]:
    """Extract the current stakeholder's hidden additive utility table from its prompt.

    Expected prompt lines look like:
        Issue A (max score 50): A1 (40), A2 (50), A3 (0)

    Returns:
        {"A": {"A1": 40.0, "A2": 50.0, ...}, ...}
    """
    utilities: Dict[str, Dict[str, float]] = {}
    for line in (prompt or "").splitlines():
        line = line.strip()
        match = _UTILITY_LINE_RE.search(line)
        if not match:
            continue
        issue = match.group(1).upper()
        options_blob = match.group(3)
        option_scores: Dict[str, float] = {}
        for option, score in _OPTION_SCORE_RE.findall(options_blob):
            option = option.upper()
            try:
                value = float(score)
            except ValueError:
                continue
            option_scores[option] = value
        if option_scores:
            utilities[issue] = option_scores
    return utilities


def build_option_catalog(utilities_by_agent: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, List[str]]:
    """Collect the known issue/option IDs without exposing any stakeholder scores."""
    catalog: Dict[str, set[str]] = {}
    for utilities in utilities_by_agent.values():
        for issue, option_scores in utilities.items():
            catalog.setdefault(issue, set()).update(option_scores.keys())
    return {issue: sorted(options) for issue, options in sorted(catalog.items())}


def parse_deal_options(deal: str) -> Dict[str, str]:
    """Parse a deal such as 'A1,B2,C1,D2,E2' into {'A':'A1', ...}."""
    parsed: Dict[str, str] = {}
    for issue, option_num in _OPTION_RE.findall(deal or ""):
        option = f"{issue.upper()}{option_num}"
        parsed[issue.upper()] = option
    return parsed


def extract_deals_from_text(text: str) -> List[str]:
    deals = [m.group(1).strip() for m in _DEAL_RE.finditer(text or "")]
    return [deal for deal in deals if deal]


def score_deal_for_utility_table(
    deal: str,
    utility_table: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Deterministically score a deal for one additive utility table."""
    parsed = parse_deal_options(deal)
    required_issues = sorted(utility_table.keys())
    missing_issues = [issue for issue in required_issues if issue not in parsed]
    unknown_options: List[str] = []
    per_issue: Dict[str, Dict[str, Any]] = {}
    total = 0.0

    for issue in required_issues:
        option = parsed.get(issue)
        if not option:
            continue
        if option not in utility_table.get(issue, {}):
            unknown_options.append(option)
            continue
        value = float(utility_table[issue][option])
        per_issue[issue] = {"option": option, "score": value}
        total += value

    return {
        "deal": deal,
        "parsed_deal": parsed,
        "total_score": total,
        "per_issue": per_issue,
        "missing_issues": missing_issues,
        "unknown_options": unknown_options,
        "valid_full_deal": not missing_issues and not unknown_options,
    }


def make_negotiation_tools(
    *,
    current_agent_name: str,
    own_utility_table: Dict[str, Dict[str, float]],
    answers_history_getter: Callable[[], Dict[str, Any]],
    prior_bundle: Dict[str, Any],
    semantic_extractor: SemanticEvidenceExtractor,
    posterior_samples: int,
    prior_seed: int,
    prior_mode: str,
    inference_logger: Callable[[Dict[str, Any]], None] | None = None,
):
    """Create per-agent tools. Each agent receives only its own true utility table."""

    @function_tool
    def calculate_my_deal_score(deal: str) -> str:
        """Calculate my exact additive utility score for a full or partial proposal.

        Input examples: 'A1,B2,C1,D2,E2' or '<DEAL>A1,B2,C1,D2,E2</DEAL>'.
        The tool returns the total score and a private per-issue breakdown according
        to my confidential utility table. Do not reveal numeric scores in the public
        negotiation answer.
        """
        result = score_deal_for_utility_table(deal, own_utility_table)
        result["stakeholder"] = current_agent_name
        result["interpretation"] = (
            "This is your own exact score. You may use it privately in your scratchpad, "
            "but you must not disclose numeric scores publicly."
        )
        return json.dumps(result, ensure_ascii=False)

    @function_tool
    def infer_opponent_utility(opponent_name: str, candidate_deal: str = "") -> str:
        """Infer an opponent's utility from its public prior and explicit statements.

        The prior was derived from that stakeholder's preliminary public discussion,
        with the general description used only as context and a score-free catalogue
        used only to map policies to option IDs. Live updating uses explicit
        comparisons and unconditional deal acceptances/rejections. Offers are not
        treated as likelihood evidence. Optionally provide candidate_deal to obtain
        its posterior score distribution and an information-gain question.
        """
        result = infer_semantic_opponent_posterior(
            opponent_name=opponent_name,
            answers_history=answers_history_getter(),
            prior_bundle=prior_bundle,
            extractor=semantic_extractor,
            candidate_deal=candidate_deal,
            n_samples=posterior_samples,
            seed=prior_seed,
            prior_mode=prior_mode,
        )
        result["requesting_agent"] = current_agent_name
        if inference_logger is not None:
            inference_logger(result)
        return json.dumps(result, ensure_ascii=False)

    return [calculate_my_deal_score, infer_opponent_utility]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class NegotiationAgent:
    name: str
    prompt_file: Path
    roles: List[str]
    model: str
    initial_prompt: str
    temperature: float
    reasoning_effort: str | None = None
    verbosity: str | None = None
    sdk_agent: Agent | None = field(default=None)

    def build_sdk_agent(self, tools: List[Any]) -> None:
        # In the original implementation, the initial prompt was the first message in
        # each agent's private message list. In the SDK, this is more naturally the
        # agent's system-level instructions.
        tool_instructions = """

You have two private tools available during your internal reasoning:
1. calculate_my_deal_score(deal): deterministically calculates your exact additive utility for a proposal using your confidential scoring table. Use it instead of mental arithmetic whenever evaluating a concrete proposal.
2. infer_opponent_utility(opponent_name, candidate_deal=''): starts from a prior derived from that stakeholder's preliminary public discussion and updates it with explicit public comparisons and accept/reject statements. The general description is context only, the issue catalogue contains no scores, and the tool never reads the opponent's private prompt or preferred-deal file. Offers alone are not likelihood evidence because this experiment has no historical data with which to calibrate proposal behaviour.

Before making a voluntary complete proposal, calculate your own score and call infer_opponent_utility for each opponent using the proposed package as candidate_deal. Treat the returned probabilities and means as uncertain estimates, not facts.

If infer_opponent_utility returns recommended_public_questions and the answer could affect your next package, include the returned question in your public <ANSWER>, after your proposed <DEAL>. You may include at most one question for each target opponent. Use this exact tag format so future agents can learn from it:
<QUESTION target="StakeholderName" option="A3">Would A3 be acceptable for you, or would another option work better?</QUESTION>

Important: tool results and numeric utility scores are private. Use them in <SCRATCHPAD>, but do not reveal scores or hidden calculations in <ANSWER>.
"""
        self.sdk_agent = Agent(
            name=self.name,
            instructions=self.initial_prompt + tool_instructions,
            model=self.model,
            model_settings=build_model_settings(
                model_name=self.model,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                verbosity=self.verbosity,
            ),
            tools=tools,
        )


@dataclass
class NegotiationState:
    full_history: Dict[str, Any]
    answers_history: Dict[str, Any]
    round_assign: List[str]
    output_file_full: Path
    output_file_answers: Path


# -----------------------------
# Orchestrator
# -----------------------------

class AgentsSDKNegotiationRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.agents, self.roles_to_players, self.initial_deal = self.load_setup()
        self.agent_utilities = {
            name: parse_utility_table_from_prompt(agent.initial_prompt)
            for name, agent in self.agents.items()
        }
        prompt_pass_rules = {
            int(match.group(1))
            for agent in self.agents.values()
            for match in [_PASS_RE.search(agent.initial_prompt)]
            if match
        }
        if len(prompt_pass_rules) > 1:
            raise ValueError(f"Inconsistent proposal-pass rules in private prompts: {sorted(prompt_pass_rules)}")
        if args.minimum_accepting_stakeholders is not None:
            self.required_acceptances = int(args.minimum_accepting_stakeholders)
        elif prompt_pass_rules:
            self.required_acceptances = next(iter(prompt_pass_rules))
        else:
            self.required_acceptances = len(self.agents) // 2 + 1
        if not 1 <= self.required_acceptances <= len(self.agents):
            raise ValueError("Required acceptances must be between 1 and the number of agents")
        self.option_catalog = build_option_catalog(self.agent_utilities)
        self.prior_bundle = load_prior_bundle(args.public_prior_file)
        prior_names = {name.casefold() for name in self.prior_bundle["stakeholders"]}
        missing_priors = [name for name in self.agents if name.casefold() not in prior_names]
        if missing_priors:
            raise ValueError(f"No public prior found for agents: {missing_priors}")
        for name, private_table in self.agent_utilities.items():
            prior_name = next(key for key in self.prior_bundle["stakeholders"] if key.casefold() == name.casefold())
            public = self.prior_bundle["stakeholders"][prior_name]
            public_catalog = {
                issue: set(row["options"])
                for issue, row in public["issues"].items()
            }
            private_catalog = {issue: set(options) for issue, options in private_table.items()}
            if public_catalog != private_catalog:
                raise ValueError(f"Public/private option catalogue mismatch for {name}")
            private_total = sum(max(options.values()) for options in private_table.values())
            if abs(private_total - float(public["total_points"])) > 1e-9:
                raise ValueError(f"Public/private maximum-score mismatch for {name}")
            threshold_match = _THRESHOLD_RE.search(self.agents[name].initial_prompt)
            if threshold_match and abs(float(threshold_match.group(1)) - float(public["acceptance_threshold"])) > 1e-9:
                raise ValueError(f"Public/private threshold mismatch for {name}")
        self.semantic_extractor = SemanticEvidenceExtractor(
            model=args.semantic_model,
            cache_file=self.output_dir / args.semantic_cache_file,
            enabled=not args.disable_semantic_evidence,
        )
        self.inference_log_file = self.output_dir / args.bayesian_log_file
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        self.state, self.round_start = self.initialize_state()
        self.attach_tools_to_agents()

    def attach_tools_to_agents(self) -> None:
        for name, negotiation_agent in self.agents.items():
            tools = make_negotiation_tools(
                current_agent_name=name,
                own_utility_table=self.agent_utilities.get(name, {}),
                answers_history_getter=lambda self=self: self.state.answers_history,
                prior_bundle=self.prior_bundle,
                semantic_extractor=self.semantic_extractor,
                posterior_samples=self.args.posterior_samples,
                prior_seed=self.args.random_seed,
                prior_mode=self.args.prior_mode,
                inference_logger=self.log_bayesian_inference,
            )
            negotiation_agent.build_sdk_agent(tools=tools)

    def log_bayesian_inference(self, record: Dict[str, Any]) -> None:
        with self.inference_log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_run_summary(self, final_public_answer: str) -> None:
        deals = extract_deals_from_text(final_public_answer)
        final_deal = deals[-1] if deals else ""
        stakeholder_rows = []
        for name, utility in self.agent_utilities.items():
            score = score_deal_for_utility_table(final_deal, utility) if final_deal else None
            prior_name = next(key for key in self.prior_bundle["stakeholders"] if key.casefold() == name.casefold())
            threshold = float(self.prior_bundle["stakeholders"][prior_name]["acceptance_threshold"])
            stakeholder_rows.append({
                "stakeholder": name,
                "true_score_evaluation_only": score["total_score"] if score else None,
                "accepts_at_known_threshold": bool(score and score["valid_full_deal"] and score["total_score"] >= threshold),
                "threshold": threshold,
            })
        accepts = sum(row["accepts_at_known_threshold"] for row in stakeholder_rows)
        agreement = bool(final_deal) and accepts >= self.required_acceptances
        blockers = []
        for row in stakeholder_rows:
            if row["accepts_at_known_threshold"]:
                continue
            name = row["stakeholder"]
            entry = {"stakeholder": name, "score": row["true_score_evaluation_only"],
                     "shortfall_to_threshold": (row["threshold"] - row["true_score_evaluation_only"])
                     if row["true_score_evaluation_only"] is not None else None}
            if final_deal:
                table = self.agent_utilities[name]
                chosen = parse_deal_options(final_deal)
                losses = []
                for issue, options in table.items():
                    best_option = max(options, key=options.get)
                    if issue in chosen and chosen[issue] in options:
                        losses.append({"issue": issue, "proposed_option": chosen[issue], "own_best_option": best_option,
                                       "points_lost": float(options[best_option] - options[chosen[issue]])})
                losses.sort(key=lambda x: -x["points_lost"])
                entry["largest_issue_losses"] = losses[:2]
            blockers.append(entry)
        blockers.sort(key=lambda b: -(b["shortfall_to_threshold"] or 0))
        voting_history = [
            {"session": key, **{k: v for k, v in value.get("deterministic_votes", {}).items() if k != "votes"}}
            for key, value in self.state.answers_history.get("voting_sessions", {}).items()
        ]
        payload = {
            "outcome": "agreement" if agreement else "no_agreement",
            "agreement_reached": agreement,
            "realized_payoffs": {row["stakeholder"]: (row["true_score_evaluation_only"] if agreement else row["threshold"])
                                 for row in stakeholder_rows},
            "no_agreement_note": None if agreement else (
                "The final proposal did not reach the required number of acceptances; every party receives "
                "its disagreement payoff. final_deal is the rejected final proposal, not an agreement."),
            # parties whose true score is below their threshold for the final proposal; when no
            # agreement is reached these are the blockers, and the first is the main disagreement
            "parties_below_threshold": blockers,
            "main_disagreement": None if agreement else (blockers[0] if blockers else None),
            "voting_history": voting_history,
            "turn_events": self.state.full_history.get("turn_events", []),
            "final_deal": final_deal,
            "primary_final_deal": deals[0] if deals else "",
            "final_deals": deals,
            "final_deal_count": len(deals),
            "stakeholders": stakeholder_rows,
            "accepting_stakeholders": accepts,
            "required_acceptances": self.required_acceptances,
            "proposal_passes": accepts >= self.required_acceptances,
            "majority_rule_passes": accepts > len(stakeholder_rows) / 2,
            "unanimous": accepts == len(stakeholder_rows),
            "random_seed": self.args.random_seed,
            "public_prior_file": str(self.args.public_prior_file),
            "private_scores_used_during_opponent_inference": False,
            "private_scores_used_here_for_retrospective_evaluation": True,
        }
        (self.output_dir / self.args.run_summary_file).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def conduct_deterministic_vote(self, public_deal_answer: str) -> Dict[str, Any]:
        """Apply the prompt's mandatory threshold rule without another LLM judgment."""
        deals = extract_deals_from_text(public_deal_answer)
        deal = deals[-1] if deals else ""
        private_votes = []
        public_votes = []
        for name, utility in self.agent_utilities.items():
            score = score_deal_for_utility_table(deal, utility) if deal else None
            prior_name = next(key for key in self.prior_bundle["stakeholders"] if key.casefold() == name.casefold())
            threshold = float(self.prior_bundle["stakeholders"][prior_name]["acceptance_threshold"])
            accepts = bool(score and score["valid_full_deal"] and score["total_score"] >= threshold)
            private_votes.append({
                "stakeholder": name,
                "vote": "ACCEPT" if accepts else "REJECT",
                "true_score_private": score["total_score"] if score else None,
                "threshold": threshold,
            })
            public_votes.append({"stakeholder": name, "vote": "ACCEPT" if accepts else "REJECT"})
        accepts = sum(row["vote"] == "ACCEPT" for row in public_votes)
        result = {
            "deal": deal,
            "votes": public_votes,
            "accepting_stakeholders": accepts,
            "required_acceptances": self.required_acceptances,
            "proposal_passes": accepts >= self.required_acceptances,
            "majority_rule_passes": accepts > len(public_votes) / 2,
            "unanimous": accepts == len(public_votes),
        }
        session_id = str(len(self.state.answers_history["voting_sessions"]) - 1)
        self.state.answers_history["voting_sessions"][session_id]["deterministic_votes"] = result
        self.state.full_history["voting_sessions"][session_id]["deterministic_votes_private"] = {
            **result,
            "votes": private_votes,
        }
        self.write_json()
        return result

    def load_setup(self) -> Tuple[Dict[str, NegotiationAgent], Dict[str, Any], str]:
        setup_path = Path(self.args.initial_prompts_file)
        lines = [
            line.strip()
            for line in setup_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != self.args.agents_num:
            raise ValueError(
                f"Expected {self.args.agents_num} agents in {setup_path}, found {len(lines)}."
            )

        agents: Dict[str, NegotiationAgent] = {}
        roles_to_players: Dict[str, List[str]] = {}

        for line in lines:
            # Supported formats:
            #   name,file,role,model
            #   name,file,role,model,reasoning_effort
            #   name,file,role,model,reasoning_effort,verbosity
            # If reasoning_effort/verbosity are omitted, global CLI values are used.
            parts = [part.strip() for part in line.split(",")]
            if len(parts) not in {4, 5, 6}:
                raise ValueError(
                    f"Bad setup line: {line!r}. Expected "
                    "name,file,role,model[,reasoning_effort[,verbosity]]."
                )

            name, prompt_file, role, model = parts[:4]
            reasoning_effort = parts[4] if len(parts) >= 5 and parts[4] else self.args.reasoning_effort
            verbosity = parts[5] if len(parts) >= 6 and parts[5] else self.args.verbosity
            roles = [r.strip() for r in role.split("&") if r.strip()]
            prompt_path = Path(prompt_file)
            initial_prompt = prompt_path.read_text(encoding="utf-8")

            agents[name] = NegotiationAgent(
                name=name,
                prompt_file=prompt_path,
                roles=roles,
                model=model,
                initial_prompt=initial_prompt,
                temperature=self.args.temp,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )
            for role_name in roles:
                roles_to_players.setdefault(role_name, []).append(name)

        if "voting_moderator" not in roles_to_players or not roles_to_players["voting_moderator"]:
            raise ValueError("One agent must have the role 'voting_moderator'.")

        # Keep the original convention: exactly one voting moderator.
        roles_to_players["voting_moderator"] = roles_to_players["voting_moderator"][0]

        initial_deal = (
            Path(self.args.initial_deal_file)
            .read_text(encoding="utf-8")
            .splitlines()[0]
            .strip()
        )
        initial_deal = normalize_plain_deal_for_plot_service(initial_deal)
        return agents, roles_to_players, initial_deal

    def initialize_state(self) -> Tuple[NegotiationState, int]:
        if self.args.restart:
            output_file_full = self.output_dir / self.args.output_file_full
            output_file_answers = self.output_dir / self.args.output_file_answers
            full_history = json.loads(output_file_full.read_text(encoding="utf-8"))
            answers_history = json.loads(output_file_answers.read_text(encoding="utf-8"))
            # Defensive normalization: if a restarted run was created by an older
            # version, make all already saved <DEAL> blocks plot_service-compatible.
            full_history = normalize_json_deals_for_plot_service(full_history)
            answers_history = normalize_json_deals_for_plot_service(answers_history)
            round_start = int(answers_history.get("finished_rounds", 0))
            round_assign = full_history.get("slot_assignment", [])
        else:
            time_str = time.strftime("%H_%M_%S", time.localtime())
            output_file_full = self.output_dir / self.args.output_file_full.replace(
                ".json", f"{time_str}.json"
            )
            output_file_answers = self.output_dir / self.args.output_file_answers.replace(
                ".json", f"{time_str}.json"
            )
            full_history = {
                "token_count": 0,
                "slot_assignment": [],
                "voting_sessions": {},
                "private_plans": {},
                "rounds": {"prompts": [], "answers": []},
            }
            answers_history = {
                "rounds": [],
                "plan": {},
                "voting_sessions": {},
                "finished_rounds": 0,
            }
            round_start = 0
            round_assign = []

        # Private plans are maintained in memory for the originating agent only.
        # They are stored in full_history for restart support, never in answers.json.
        answers_history["plan"] = full_history.get(
            "private_plans", answers_history.get("plan", {})
        )

        state = NegotiationState(
            full_history=full_history,
            answers_history=answers_history,
            round_assign=round_assign,
            output_file_full=output_file_full,
            output_file_answers=output_file_answers,
        )
        return state, round_start

    def write_json(self) -> None:
        self.state.full_history["private_plans"] = self.state.answers_history.get("plan", {})
        self.state.output_file_full.write_text(
            json.dumps(self.state.full_history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        public_answers = dict(self.state.answers_history)
        public_answers["plan"] = {}
        public_answers["private_plans_redacted"] = True
        self.state.output_file_answers.write_text(
            json.dumps(public_answers, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add_usage(self, usage: Dict[str, Any]) -> None:
        total = usage.get("total_tokens") or 0
        try:
            self.state.full_history["token_count"] += int(total)
        except Exception:
            pass

    def save_full_turn(
        self,
        *,
        prompt: str,
        raw_answer: str,
        answer_type: str,
        agent_name: str,
        usage: Dict[str, Any] | None = None,
    ) -> None:
        # Normalize before writing to full_conversation.json so downstream
        # plot_service functions can parse all saved <DEAL> blocks directly.
        prompt = normalize_deal_text_for_plot_service(prompt)
        raw_answer = normalize_deal_text_for_plot_service(raw_answer)

        if usage:
            self.add_usage(usage)
        if answer_type == "round":
            self.state.full_history["rounds"]["prompts"].append(prompt)
            self.state.full_history["rounds"]["answers"].append([agent_name, raw_answer])
        elif answer_type == "deal_suggestion":
            voting_session_num = str(len(self.state.full_history["voting_sessions"]))
            self.state.full_history["voting_sessions"][voting_session_num] = {
                "deal_suggestion": {
                    "agent": agent_name,
                    "prompt": prompt,
                    "answer": raw_answer,
                    "usage": usage or {},
                }
            }
        self.write_json()

    def save_public_answer(self, *, answer: str, answer_type: str, agent_name: str) -> None:
        # Normalize before writing to answers.json. This makes the final output
        # directly compatible with plot_service.extract_deals_from_json() and
        # plot_service.create_deals_dataframe(), which expects comma-space deals.
        answer = normalize_deal_text_for_plot_service(answer)

        if answer_type == "round":
            self.state.answers_history["finished_rounds"] += 1
            self.state.answers_history["rounds"].append([agent_name, answer])
        elif answer_type == "plan":
            self.state.answers_history["plan"].setdefault(agent_name, []).append(answer)
        elif answer_type == "deal_suggestion":
            voting_session_num = str(len(self.state.answers_history["voting_sessions"]))
            self.state.answers_history["voting_sessions"][voting_session_num] = {
                "deal_suggestion": answer,
                "agent": agent_name,
            }
        self.write_json()

    def generate_round_assignment(self) -> None:
        names = list(self.agents.keys())
        last_agent = self.roles_to_players["voting_moderator"]
        round_assign: List[str] = []

        for _ in range(int(np.ceil(self.args.rounds / len(names)))):
            shuffled = random.sample(names, len(names))
            while shuffled[0] == last_agent or shuffled[-1] == self.roles_to_players["voting_moderator"]:
                shuffled = random.sample(names, len(names))
            round_assign.extend(shuffled)
            last_agent = shuffled[-1]

        self.state.round_assign = round_assign[: self.args.rounds]
        self.state.full_history["slot_assignment"] = self.state.round_assign
        self.write_json()

    @traceable(name="agents_sdk_agent_turn", run_type="chain")
    def run_sdk_turn(self, agent_name: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """One LangSmith-traced call to one OpenAI Agents SDK Agent.

        The outer scheduler still selects exactly one speaker. max_turns controls only
        that speaker's internal tool-use loop before producing the public answer.
        """
        sdk_agent = self.agents[agent_name].sdk_agent
        if sdk_agent is None:
            raise RuntimeError(f"SDK agent for {agent_name!r} was not initialized.")

        try:
            result = Runner.run_sync(
                starting_agent=sdk_agent,
                input=prompt,
                # Tool use requires at least one model turn to request the tool and
                # another to produce the final answer after observing the tool result.
                max_turns=self.args.max_turns,
            )
            return str(result.final_output or ""), safe_usage_from_result(result)
        except MaxTurnsExceeded:
            # The speaker kept calling tools without answering. This is a limit of one
            # agent turn, not a negotiation outcome: ask once more for an answer with
            # tools disabled instead of aborting the whole negotiation.
            self.record_turn_event(agent_name, "max_turns_exceeded_fallback_without_tools")
            print(f"WARNING: {agent_name} exceeded {self.args.max_turns} internal turns; retrying without tools.")
            no_tools = sdk_agent.clone(
                tools=[],
                model_settings=dataclasses.replace(sdk_agent.model_settings, tool_choice=None),
            )
            fallback_prompt = (
                prompt
                + "\n\nYour tool-call budget for this turn is exhausted. Do not call tools. "
                "Use what you already know and give your answer now in the required format."
            )
            try:
                result = Runner.run_sync(starting_agent=no_tools, input=fallback_prompt, max_turns=2)
                return str(result.final_output or ""), safe_usage_from_result(result)
            except Exception as error:  # still no answer: the speaker passes this slot
                self.record_turn_event(agent_name, f"turn_skipped: {type(error).__name__}: {error}")
                print(f"WARNING: {agent_name} produced no answer; the turn is recorded as a pass.")
                return (
                    "<ANSWER>We need more time to consider the current proposals and pass this turn.</ANSWER>",
                    {},
                )

    def record_turn_event(self, agent_name: str, event: str) -> None:
        events = self.state.full_history.setdefault("turn_events", [])
        events.append({"agent": agent_name, "event": event,
                       "voting_sessions_so_far": len(self.state.answers_history.get("voting_sessions", {}))})
        self.write_json()

    @traceable(name="initial_negotiation_proposal_sdk", run_type="chain")
    def run_initial_turn(self) -> None:
        moderator = self.roles_to_players["voting_moderator"]
        prompt = build_first_slot(deal=self.initial_deal, name=moderator)
        raw_answer, usage = self.run_sdk_turn(moderator, prompt)

        self.save_full_turn(
            prompt=prompt,
            raw_answer=raw_answer,
            answer_type="round",
            agent_name=moderator,
            usage=usage,
        )
        public_answer = extract_public_answer(raw_answer, allow_safe_unwrapped=True)
        self.save_public_answer(answer=public_answer, answer_type="round", agent_name=moderator)
        plan = extract_plan(raw_answer)
        if plan:
            self.save_public_answer(answer=plan, answer_type="plan", agent_name=moderator)
        print(f"\nInitial proposal by {moderator}:\n{public_answer}\n")

    @traceable(name="negotiation_round_sdk", run_type="chain")
    def one_negotiation_round(self, round_idx: int, agent_name: str) -> None:
        if self.args.sleep_seconds > 0:
            time.sleep(self.args.sleep_seconds)

        prompt = build_slot_prompts(
            self.state.answers_history,
            agent_name,
            self.roles_to_players["voting_moderator"],
            window_size=self.args.window_size,
            final_round=(self.args.rounds - round_idx) <= self.args.agents_num,
        )
        raw_answer, usage = self.run_sdk_turn(agent_name, prompt)

        self.save_full_turn(
            prompt=prompt,
            raw_answer=raw_answer,
            answer_type="round",
            agent_name=agent_name,
            usage=usage,
        )

        public_answer = extract_public_answer(raw_answer)
        plan = extract_plan(raw_answer)
        if plan:
            self.save_public_answer(answer=plan, answer_type="plan", agent_name=agent_name)
        self.save_public_answer(answer=public_answer, answer_type="round", agent_name=agent_name)

        print(f"\n[{round_idx + 1}] {agent_name} public answer:\n{public_answer}\n")

    @traceable(name="deal_suggestion_session_sdk", run_type="chain")
    def suggest_deal(self, agent_name: str, final_voting: bool = False) -> str:
        prompt = build_deal_suggestion_prompt(
            self.state.answers_history,
            agent_name,
            final_vote=final_voting,
            window_size=self.args.window_size,
        )
        raw_answer, usage = self.run_sdk_turn(agent_name, prompt)

        self.save_full_turn(
            prompt=prompt,
            raw_answer=raw_answer,
            answer_type="deal_suggestion",
            agent_name=agent_name,
            usage=usage,
        )

        public_answer = extract_public_answer(raw_answer)
        self.save_public_answer(answer=public_answer, answer_type="deal_suggestion", agent_name=agent_name)
        plan = extract_plan(raw_answer)
        if plan:
            self.save_public_answer(answer=plan, answer_type="plan", agent_name=agent_name)
        return public_answer

    @traceable(name="full_agents_sdk_negotiation", run_type="chain")
    def run(self) -> None:
        # This trace groups all SDK spans in the OpenAI traces dashboard.
        # LangSmith tracing is provided by the @traceable decorators above.
        with trace(
            workflow_name="cooperative_llm_negotiation",
            group_id=f"negotiation-{int(time.time())}",
            metadata={
                # OpenAI Agents SDK trace metadata values must be strings.
                # Passing ints causes a non-fatal 400 tracing upload error.
                "rounds": str(self.args.rounds),
                "agents_num": str(self.args.agents_num),
                "votes_interval": str(self.args.votes_interval),
                "reasoning_effort": str(self.args.reasoning_effort),
                "verbosity": str(self.args.verbosity),
                "window_size": str(self.args.window_size),
                "max_turns": str(self.args.max_turns),
            },
        ):
            if not self.args.restart:
                self.generate_round_assignment()
                self.run_initial_turn()

            if self.round_start != 0 and self.round_start >= self.args.rounds:
                if not self.state.answers_history.get("voting_sessions"):
                    moderator = self.roles_to_players["voting_moderator"]
                    deal = self.suggest_deal(moderator, final_voting=True)
                    self.conduct_deterministic_vote(deal)
                    self.write_run_summary(deal)
                    print(deal)
                return

            for round_idx in range(self.round_start, self.args.rounds):
                agent_name = self.state.round_assign[round_idx]
                self.one_negotiation_round(round_idx, agent_name)

                if (
                    self.args.votes_interval > 0
                    and (round_idx + 1) % self.args.votes_interval == 0
                    and (round_idx + 1) < self.args.rounds
                ):
                    moderator = self.roles_to_players["voting_moderator"]
                    deal = self.suggest_deal(moderator, final_voting=False)
                    self.conduct_deterministic_vote(deal)
                    print(f"\nIntermediate deal suggestion by {moderator}:\n{deal}\n")

            moderator = self.roles_to_players["voting_moderator"]
            final_deal = self.suggest_deal(moderator, final_voting=True)
            self.conduct_deterministic_vote(final_deal)
            print(f"\nFinal deal suggestion by {moderator}:\n{final_deal}\n")
            self.write_run_summary(final_deal)

        # Makes SDK traces more likely to appear immediately after the script exits.
        flush_traces()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cooperative LLM negotiation with OpenAI Agents SDK + LangSmith"
    )
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--output_file_full", type=str, default="full_conversation.json")
    parser.add_argument("--output_file_answers", type=str, default="answers.json")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--agents_num", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--votes_interval", type=int, default=6)
    parser.add_argument("--initial_prompts_file", type=str, default="initial_prompts_base/initial_prompts.txt")
    parser.add_argument("--initial_deal_file", type=str, default="initial_prompts_base/initial_deal.txt")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum internal SDK turns per scheduled speaker; allow >1 so tools can be called.")
    parser.add_argument("--public_prior_file", default="public_priors.json")
    parser.add_argument("--semantic_model", default="gpt-5.5")
    parser.add_argument("--posterior_samples", type=int, default=12000)
    parser.add_argument(
        "--prior_mode",
        choices=["semantic", "structural"],
        default="semantic",
        help="Use LLM-derived ordinal constraints or an unconstrained maximum-entropy baseline.",
    )
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--semantic_cache_file", default="semantic_evidence_cache.json")
    parser.add_argument("--bayesian_log_file", default="bayesian_inference_calls.jsonl")
    parser.add_argument("--run_summary_file", default="run_summary.json")
    parser.add_argument(
        "--minimum_accepting_stakeholders",
        type=int,
        default=None,
        help=(
            "Override the common pass rule parsed from the stakeholder prompts. "
            "By default the Campylobacter prompts' 'at least 2 parties' rule is used."
        ),
    )
    parser.add_argument(
        "--disable_semantic_evidence",
        action="store_true",
        help="Ablation: use the public-information prior without live LLM evidence extraction.",
    )
    parser.add_argument(
        "--trace_sdk_tool_calls_to_langsmith",
        action="store_true",
        help="Forward native OpenAI Agents SDK spans, including tool-call input/output, to LangSmith.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=["default", "none", "low", "medium", "high", "xhigh"],
        help=(
            "Reasoning effort for GPT-5-style models. Use xhigh for maximum reasoning; "
            "use high/medium/low for cheaper or faster runs; use default to omit the setting."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="medium",
        choices=["default", "low", "medium", "high"],
        help="GPT-5-style text verbosity. Use default to omit the setting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    configure_openai_agents_langsmith_tracing(parsed_args.trace_sdk_tool_calls_to_langsmith)
    runner = AgentsSDKNegotiationRunner(parsed_args)
    runner.run()

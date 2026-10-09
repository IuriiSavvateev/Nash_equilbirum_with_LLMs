"""Build a conservative, quote-verified prior cache for the supplied case.

This cache lets the notebook run without paying for an additional semantic
extraction call.  It contains only qualitative relations that are directly
supported by the prepared public conversation.  ``generate_discussion_priors``
remains available when an independent LLM re-coding is desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def rel_issue(more: str, less: str, quote: str, explanation: str) -> dict:
    return {
        "more_important": more,
        "less_important": less,
        "basis": "semantic_mapping",
        "evidence_quote": quote,
        "explanation": explanation,
    }


def rel_option(preferred: str, less: str, quote: str, explanation: str) -> dict:
    return {
        "preferred": preferred,
        "less_preferred": less,
        "basis": "semantic_mapping",
        "evidence_quote": quote,
        "explanation": explanation,
    }


def relevant(issue: str, quote: str, explanation: str) -> dict:
    return {
        "issue": issue,
        "basis": "explicit",
        "evidence_quote": quote,
        "explanation": explanation,
    }


CODING = {
    "Consumers": {
        "relevant": [
            relevant("A", "Stricter microbiological criteria can play a critical role in enhancing food safety", "Directly addresses whether stricter criteria should be introduced."),
            relevant("B", "Clear labeling and consumer education on safe food handling are essential", "Directly endorses consumer-stage transparency and education as a control focus."),
            relevant("C", "should not result in substantial price increases that limit access to safe poultry products", "Directly addresses how regulatory costs should affect consumers."),
        ],
        "issue_relations": [],
        "option_relations": [
            rel_option("A3", "A1", "Stricter microbiological criteria can play a critical role in enhancing food safety", "A3 introduces a risk-based FSC, whereas A1 retains the PHC-only status quo."),
            rel_option("B4", "B2", "Clear labeling and consumer education on safe food handling are essential", "B4 centers consumer behavior and transparency; B2 centers processing controls."),
            rel_option("C2", "C3", "with comprehensive consumer education campaigns and subsidies to offset potential economic burdens", "C2 supplies public co-funding; C3 passes costs to consumers without subsidies."),
        ],
        "uncertainties": [
            "The discussion does not distinguish A2, A3, and A4 precisely enough for a full ranking.",
            "No clear preference among D1-D4 is stated.",
            "Issue E options have placeholder descriptions and the consumer discussion does not order them.",
        ],
    },
    "Food industry": {
        "relevant": [
            relevant("A", "Implement a phased approach to stricter microbiological criteria", "Directly addresses the criterion type and stringency."),
            relevant("B", "focus on interventions at the farm and processing levels", "Directly addresses where controls should be concentrated."),
            relevant("C", "supported by industry innovation and adaptation strategies", "Directly addresses support, compliance burden, and waste mitigation."),
            relevant("E", "Align strategies with the EU Green Deal and circular economy goals", "Directly makes environmental sustainability relevant."),
        ],
        "issue_relations": [],
        "option_relations": [
            rel_option("A2", "A4", "Implement a phased approach to stricter microbiological criteria", "A2 is phased and lenient; A4 imposes strict immediate enforcement."),
            rel_option("A2", "A1", "Implement a phased approach to stricter microbiological criteria", "The industry endorses a phased stricter criterion rather than the PHC-only status quo."),
            rel_option("B3", "B4", "focus on interventions at the farm and processing levels", "B3 is a multi-stage approach; B4 centers controls at the consumer stage."),
            rel_option("C4", "C1", "real-time monitoring and predictive risk algorithms, to prevent unnecessary batch withdrawals", "C4 funds waste-minimizing innovations; C1 uses standard disposal without support."),
            rel_option("C2", "C1", "with adequate support for adaptation", "C2 provides adaptation support; C1 places the full burden on producers."),
        ],
        "uncertainties": [
            "The discussion does not cleanly distinguish B1, B2, and B3 beyond favoring farm/processing controls over a consumer-only focus.",
            "No clear preference among D1-D4 is stated.",
            "Environmental concern is clear, but the placeholder E option descriptions do not support an E1-E3 ordering.",
        ],
    },
    "Food Safety Authorities": {
        "relevant": [
            relevant("A", "Stricter microbiological criteria are essential to effectively mitigate Campylobacter infections", "Directly addresses criterion adoption and stringency."),
            relevant("B", "comprehensive risk management strategies that encompass the entire poultry production chain", "Directly addresses the location and breadth of controls."),
            relevant("C", "dynamic shelf-life management to reduce unnecessary food waste", "Directly addresses a waste-mitigation mechanism."),
            relevant("E", "economic and environmental sustainability", "Directly makes environmental consequences relevant."),
        ],
        "issue_relations": [
            rel_issue("A", "C", "the overarching priority is public health", "Public-health criterion design is explicitly placed above economic and waste concerns."),
        ],
        "option_relations": [
            rel_option("A3", "A1", "Stricter microbiological criteria are essential to effectively mitigate Campylobacter infections", "A3 introduces a risk-based FSC; A1 retains PHC only."),
            rel_option("A3", "A4", "must be implemented with a balanced approach that also considers economic and environmental impacts", "A3 combines a moderate public-health target with risk-based sampling and waste mitigation; A4 is immediate strict enforcement."),
            rel_option("B3", "B2", "comprehensive risk management strategies that encompass the entire poultry production chain", "B3 spans the chain, unlike the processing-centered B2."),
            rel_option("B3", "B4", "comprehensive risk management strategies that encompass the entire poultry production chain", "B3 spans the chain, unlike the consumer-centered B4."),
            rel_option("C4", "C1", "dynamic shelf-life management to reduce unnecessary food waste", "C4 explicitly funds dynamic shelf-life and reprocessing tools; C1 uses standard disposal."),
        ],
        "uncertainties": [
            "No clear preference among D1-D4 is stated.",
            "The placeholder E option descriptions do not support an E1-E3 ordering.",
            "The relation between B3 and the environment-centered B5 is not explicit enough to order.",
        ],
    },
    "Environmental NGOs": {
        "relevant": [
            relevant("A", "Stricter criteria must focus on minimizing environmental transmission", "Directly addresses the form of a stricter criterion."),
            relevant("B", "targeting sources of contamination rather than solely tightening end-product standards", "Directly addresses where controls should be focused."),
            relevant("C", "waste mitigation strategies that valorize rejected meat through safe reprocessing and alternative uses", "Directly endorses a waste-mitigation mechanism."),
            relevant("E", "the ecological implications of stricter microbiological criteria cannot be overlooked", "Directly makes environmental footprint relevant."),
        ],
        "issue_relations": [],
        "option_relations": [
            rel_option("A3", "A4", "rather than solely tightening end-product standards", "A3 combines risk-based FSC actions with waste mitigation; A4 emphasizes strict immediate batch withdrawal."),
            rel_option("A3", "A1", "Implement stricter microbiological criteria that prioritize source-based interventions", "A3 introduces a risk-based FSC, whereas A1 retains PHC only."),
            rel_option("B5", "B1", "minimizing environmental transmission by targeting sources of contamination", "B5 explicitly centers environmental leakage, unlike the processing-and-retail focus of B1."),
            rel_option("C4", "C1", "valorize rejected meat through safe reprocessing and alternative uses", "C4 provides reprocessing and valorization tools; C1 sends failed batches through standard disposal routes."),
        ],
        "uncertainties": [
            "The discussion supports source control but does not fully order B2, B3, and B5.",
            "No clear preference among D1-D4 is stated.",
            "Environmental intervention is endorsed, but the placeholder E option descriptions do not justify choosing a guideline over binding text or vice versa.",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the quote-verified Campylobacter prior cache")
    parser.add_argument("--public_info_dir", default="public_info_Campylobacter")
    parser.add_argument("--output", default="public_priors_Campylobacter.json")
    parser.add_argument("--total_points", type=float, default=100.0)
    parser.add_argument("--acceptance_threshold", type=float, default=65.0)
    args = parser.parse_args()

    public_dir = Path(args.public_info_dir).resolve()
    catalogue_path = public_dir / "public_issue_catalogue.json"
    discussion_path = public_dir / "preliminary_discussion.json"
    general_path = public_dir / "General_Description_Campylobacter.txt"
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    discussion = json.loads(discussion_path.read_text(encoding="utf-8"))
    general_text = general_path.read_text(encoding="utf-8-sig")
    issues = catalogue["issues"]
    if catalogue.get("private_scores_included") is not False:
        raise ValueError("The public catalogue does not certify score exclusion")
    if set(CODING) != set(discussion["stakeholders"]):
        raise ValueError("Prior coding and discussion stakeholder sets differ")

    stakeholders = {}
    for name, code in CODING.items():
        messages = discussion["stakeholders"][name]
        discussion_text = "\n\n".join(row["message"] for row in messages)
        evidence_rows = [*code["relevant"], *code["issue_relations"], *code["option_relations"]]
        missing_quotes = [row["evidence_quote"] for row in evidence_rows if norm(row["evidence_quote"]) not in norm(discussion_text)]
        if missing_quotes:
            raise ValueError(f"Unverified evidence quote(s) for {name}: {missing_quotes}")
        stakeholders[name] = {
            "stakeholder": name,
            "issues": issues,
            "relevant_issues": [row["issue"] for row in code["relevant"]],
            "relevant_issue_evidence": code["relevant"],
            "issue_relations": code["issue_relations"],
            "option_relations": code["option_relations"],
            "uncertainties": code["uncertainties"],
            "validation_warnings": [],
            "source_file": f"{discussion_path.name}#{name}",
            "source_sha256": sha256_text(general_text + json.dumps(issues, sort_keys=True) + discussion_text),
            "source_message_count": len(messages),
            "source_conversation_indices": [row["conversation_index"] for row in messages],
            "total_points": args.total_points,
            "acceptance_threshold": args.acceptance_threshold,
        }

    payload = {
        "schema": "semantic_ordinal_prior_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "quote-verified conservative semantic coding cache",
        "elicitation_protocol_sha256": sha256_text(Path(__file__).with_name("preliminary_discussion_prior_protocol.md").read_text(encoding="utf-8")),
        "general_description_sha256": sha256_text(general_text),
        "issue_catalogue_sha256": sha256_text(json.dumps(issues, sort_keys=True)),
        "preliminary_discussion_sha256": discussion["source_sha256"],
        "preference_evidence_source": "target stakeholder messages from Preliminary_discussion.json/conversation",
        "general_description_role": "case context only",
        "issue_catalogue_role": "score-free semantic mapping only",
        "private_files_used": False,
        "private_scores_used": False,
        "preferred_deals_used": False,
        "numeric_llm_confidence_used": False,
        "evidence_quotes_textually_verified": True,
        "validation_mode": "conservative_semantic_coding_plus_exact_normalized_quote_verification",
        "prior_interpretation": "Maximum-entropy structural prior conditioned only on the listed qualitative public-discussion relations.",
        "stakeholders": stakeholders,
        "validation_warnings": [],
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output} for {len(stakeholders)} stakeholders")


if __name__ == "__main__":
    main()

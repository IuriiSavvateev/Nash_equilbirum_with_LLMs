"""Generate leakage-checked ordinal priors from the Campylobacter discussion.

The LLM receives only the general case description, a score-free issue catalogue,
and the target stakeholder's own public messages extracted from
``Preliminary_discussion.json/conversation``. It never receives private prompts, scores, or
preferred-deal files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


PRIVATE_MARKERS = (
    'your confidential information and preferences',
    'preferences by order of importance',
)
PRIVATE_SCORE_RE = re.compile(r'Issue\s+[A-Z]\s*\(max\s+score', re.IGNORECASE)


class IssueRelation(BaseModel):
    more_important: str = Field(description='Single-letter issue ID')
    less_important: str = Field(description='Single-letter issue ID')
    basis: Literal['explicit', 'semantic_mapping']
    evidence_quote: str
    explanation: str


class OptionRelation(BaseModel):
    preferred: str = Field(description='Option ID, such as A1')
    less_preferred: str = Field(description='Option ID from the same issue')
    basis: Literal['explicit', 'semantic_mapping']
    evidence_quote: str
    explanation: str


class RelevantIssue(BaseModel):
    issue: str = Field(description='Single-letter issue ID')
    basis: Literal['explicit', 'semantic_mapping']
    evidence_quote: str
    explanation: str


class PriorAssessment(BaseModel):
    stakeholder: str
    relevant_issues: list[RelevantIssue]
    issue_relations: list[IssueRelation]
    option_relations: list[OptionRelation]
    uncertainties: list[str]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def creates_cycle(edges: list[tuple[str, str]], new_edge: tuple[str, str]) -> bool:
    graph: dict[str, set[str]] = {}
    for high, low in [*edges, new_edge]:
        graph.setdefault(high, set()).add(low)
    stack, visited = [new_edge[1]], set()
    while stack:
        node = stack.pop()
        if node == new_edge[0]:
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, ()))
    return False


def reject_private_content(label: str, text: str) -> None:
    lower = text.casefold()
    if any(marker in lower for marker in PRIVATE_MARKERS) or PRIVATE_SCORE_RE.search(text):
        raise ValueError(f'Private prompt/score marker detected in public prior input: {label}')


def validate_assessment(
    raw: PriorAssessment,
    stakeholder: str,
    issues: dict,
    discussion_text: str,
) -> tuple[dict, list[str]]:
    all_options = {option for row in issues.values() for option in row['options']}
    warnings: list[str] = []

    relevant_rows = []
    seen_relevant = set()
    for item in raw.relevant_issues:
        issue = item.issue.upper()
        if issue not in issues or issue in seen_relevant:
            warnings.append(f'Dropped invalid/duplicate relevant issue {issue}')
            continue
        if not item.evidence_quote.strip() or not item.explanation.strip():
            warnings.append(f'Dropped relevant issue {issue}: missing evidence or explanation')
            continue
        seen_relevant.add(issue)
        relevant_rows.append({**item.model_dump(), 'issue': issue})

    issue_relations = []
    seen_issue = set()
    issue_edges: list[tuple[str, str]] = []
    for item in raw.issue_relations:
        high, low = item.more_important.upper(), item.less_important.upper()
        key = (high, low)
        if high not in issues or low not in issues or high == low or key in seen_issue:
            warnings.append(f'Dropped invalid/duplicate issue relation {high}>{low}')
            continue
        if not item.evidence_quote.strip() or not item.explanation.strip():
            warnings.append(f'Dropped issue relation {high}>{low}: missing evidence or explanation')
            continue
        if creates_cycle(issue_edges, key):
            warnings.append(f'Dropped cyclic issue relation {high}>{low}')
            continue
        seen_issue.add(key)
        issue_edges.append(key)
        issue_relations.append({**item.model_dump(), 'more_important': high, 'less_important': low})

    option_relations = []
    seen_option = set()
    option_edges: dict[str, list[tuple[str, str]]] = {}
    for item in raw.option_relations:
        high, low = item.preferred.upper(), item.less_preferred.upper()
        key = (high, low)
        if (
            high not in all_options
            or low not in all_options
            or high[0] != low[0]
            or high == low
            or key in seen_option
        ):
            warnings.append(f'Dropped invalid/duplicate option relation {high}>{low}')
            continue
        if not item.evidence_quote.strip() or not item.explanation.strip():
            warnings.append(f'Dropped option relation {high}>{low}: missing evidence or explanation')
            continue
        edges = option_edges.setdefault(high[0], [])
        if creates_cycle(edges, key):
            warnings.append(f'Dropped cyclic option relation {high}>{low}')
            continue
        seen_option.add(key)
        edges.append(key)
        option_relations.append({**item.model_dump(), 'preferred': high, 'less_preferred': low})

    return {
        'stakeholder': stakeholder,
        'issues': issues,
        'relevant_issues': sorted(seen_relevant),
        'relevant_issue_evidence': relevant_rows,
        'issue_relations': issue_relations,
        'option_relations': option_relations,
        'uncertainties': raw.uncertainties,
        'validation_warnings': warnings,
    }, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description='Create Campylobacter priors from preliminary public discussion')
    parser.add_argument('--public_info_dir', required=True)
    parser.add_argument('--output', default='public_priors_Campylobacter.json')
    parser.add_argument('--model', default='gpt-5.5')
    parser.add_argument(
        '--protocol_file',
        default=str(Path(__file__).with_name('preliminary_discussion_prior_protocol.md')),
    )
    parser.add_argument('--total_points', type=float, default=100.0)
    parser.add_argument('--acceptance_threshold', type=float, default=65.0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.total_points <= 0:
        raise ValueError('--total_points must be positive')
    if not 0 <= args.acceptance_threshold <= args.total_points:
        raise ValueError('--acceptance_threshold must be within [0, total_points]')

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f'Using existing prior file: {output}. Pass --force to regenerate it.')
        return

    public_dir = Path(args.public_info_dir)
    general_path = public_dir / 'General_Description_Campylobacter.txt'
    catalogue_path = public_dir / 'public_issue_catalogue.json'
    discussion_path = public_dir / 'preliminary_discussion.json'
    for path in (general_path, catalogue_path, discussion_path):
        if not path.exists():
            raise FileNotFoundError(f'Missing prepared public input: {path}')

    general_text = general_path.read_text(encoding='utf-8-sig')
    catalogue = json.loads(catalogue_path.read_text(encoding='utf-8'))
    discussion = json.loads(discussion_path.read_text(encoding='utf-8'))
    protocol_text = Path(args.protocol_file).read_text(encoding='utf-8')
    if catalogue.get('private_scores_included') is not False:
        raise ValueError('Score-free catalogue does not certify private-score exclusion')
    issues = catalogue.get('issues', {})
    if not issues:
        raise ValueError('Prepared public issue catalogue is empty')
    reject_private_content(general_path.name, general_text)
    # Scan only what is sent to the model. The catalogue's provenance metadata
    # (e.g. 'extraction_boundary') legitimately names the private-section marker
    # it stopped at, which previously caused a false-positive rejection.
    reject_private_content(f'{catalogue_path.name}:issues', json.dumps(issues, ensure_ascii=False))
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set. Load it before regenerating priors (see notebook Step 4).')

    client = OpenAI()
    stakeholders = {}
    all_warnings = []
    for stakeholder, messages in discussion.get('stakeholders', {}).items():
        discussion_text = '\n\n'.join(
            f'[Message {row["conversation_index"]}]\n{row["message"]}' for row in messages
        )
        reject_private_content(f'{discussion_path.name}:{stakeholder}', discussion_text)
        combined = (
            f'TARGET STAKEHOLDER: {stakeholder}\n\n'
            f'GENERAL CASE CONTEXT (context only; never use it as preference evidence):\n{general_text}\n\n'
            f'SCORE-FREE ISSUE/OPTION CATALOGUE (mapping reference only; never use it as preference evidence):\n'
            f'{json.dumps(issues, indent=2, ensure_ascii=False)}\n\n'
            f'TARGET STAKEHOLDER PRELIMINARY DISCUSSION (the only preference-evidence source):\n'
            f'{discussion_text}'
        )
        response = client.responses.parse(
            model=args.model,
            input=[
                {'role': 'system', 'content': protocol_text},
                {'role': 'user', 'content': combined},
            ],
            text_format=PriorAssessment,
        )
        if response.output_parsed is None:
            raise RuntimeError(f'No parsed prior assessment returned for {stakeholder}')
        validated, warnings = validate_assessment(
            response.output_parsed, stakeholder, issues, discussion_text
        )
        source_fingerprint = sha256_text(
            general_text + json.dumps(issues, sort_keys=True) + discussion_text
        )
        validated.update({
            'source_file': f'{discussion_path.name}#{stakeholder}',
            'source_sha256': source_fingerprint,
            'source_message_count': len(messages),
            'source_conversation_indices': [row['conversation_index'] for row in messages],
            'total_points': args.total_points,
            'acceptance_threshold': args.acceptance_threshold,
        })
        stakeholders[stakeholder] = validated
        all_warnings.extend(f'{stakeholder}: {warning}' for warning in warnings)

    payload = {
        'schema': 'semantic_ordinal_prior_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'elicitation_protocol_sha256': sha256_text(protocol_text),
        'general_description_sha256': sha256_text(general_text),
        'issue_catalogue_sha256': sha256_text(json.dumps(issues, sort_keys=True)),
        'preliminary_discussion_sha256': discussion.get('source_sha256'),
        'preference_evidence_source': 'target stakeholder messages from Preliminary_discussion.json/conversation',
        'general_description_role': 'case context only',
        'issue_catalogue_role': 'score-free semantic mapping only',
        'private_files_used': False,
        'private_scores_used': False,
        'preferred_deals_used': False,
        'numeric_llm_confidence_used': False,
        'evidence_quotes_textually_verified': False,
        'validation_mode': 'llm_semantic_assessment_plus_structural_id_duplicate_cycle_checks',
        'prior_interpretation': (
            'Maximum-entropy structural prior conditioned on validated qualitative '
            'relations extracted from preliminary public discussion.'
        ),
        'stakeholders': stakeholders,
        'validation_warnings': all_warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {output} for {len(stakeholders)} stakeholders')


if __name__ == '__main__':
    main()

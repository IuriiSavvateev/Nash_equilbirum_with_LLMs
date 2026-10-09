"""Extract score-free Campylobacter inputs from the supplied source files.

The issue and option formulations are read only from the public prefix of the
four final prompts.  Preliminary-discussion evidence is copied only from the
``conversation`` field of the supplied JSON.  The private prompt suffix is
neither parsed nor written to the output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PRIVATE_MARKER = "Your confidential information and preferences:"
ISSUE_RE = re.compile(
    r'^Issue\s+([A-Z]):\s+"([^"]+)"\s*\n(.*?)(?=^=+\s*$|\Z)',
    re.MULTILINE | re.DOTALL,
)
OPTION_RE = re.compile(
    r'^([A-Z]\d+)\.\s+"([^"]+)":\s*(.*?)(?=^[A-Z]\d+\.\s+"|\Z)',
    re.MULTILINE | re.DOTALL,
)
MODERATOR_NOTE_RE = re.compile(r'\s*\(Moderator note:.*?\)\s*', re.IGNORECASE | re.DOTALL)
LEADING_QUOTE_RE = re.compile(r'^\s*"[^"\n]*(?:\n[^"\n]*)*"\s*', re.DOTALL)


ROLE_BY_FILE = {
    "Consumers.txt": "Consumers",
    "Food_Industry.txt": "Food industry",
    "FSA.txt": "Food Safety Authorities",
    "NGOs.txt": "Environmental NGOs",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def public_prefix(path: Path) -> str:
    text = read_text(path)
    if PRIVATE_MARKER not in text:
        raise ValueError(f"Private-section marker not found in {path}")
    prefix, _private_suffix = text.split(PRIVATE_MARKER, 1)
    if re.search(r'Issue\s+[A-Z]\s*\(max\s+score', prefix, re.IGNORECASE):
        raise ValueError(f"Score marker found before the private boundary in {path}")
    return prefix.strip()


def parse_catalogue(prefix: str) -> dict:
    issues: dict[str, dict] = {}
    for issue_match in ISSUE_RE.finditer(prefix):
        issue, title, body = issue_match.groups()
        options = {}
        for option_match in OPTION_RE.finditer(body.strip()):
            option, option_title, description = option_match.groups()
            description = re.sub(r"\s+", " ", description).strip()
            options[option] = {"title": option_title.strip(), "description": description}
        if not options:
            raise ValueError(f"No options parsed for issue {issue}")
        issues[issue] = {"title": title.strip(), "options": options}
    if not issues:
        raise ValueError("No public issues parsed")
    return issues


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def extract_general_description(prefix: str) -> str:
    before_role = prefix.split('You represent the "', 1)[0].strip()
    if "Based on preliminary discussions" in before_role:
        before_role = before_role.split("Based on preliminary discussions", 1)[0].strip()
    return before_role


def clean_message(message: str) -> str:
    text = str(message).replace("\r\n", "\n").strip()
    # As in the OPM preparation: remove an opening quotation that merely repeats
    # a previous speaker and remove embedded moderator annotations.
    text = LEADING_QUOTE_RE.sub("", text, count=1).strip()
    text = MODERATOR_NOTE_RE.sub("\n", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-separated Campylobacter public inputs")
    parser.add_argument("--prompts_dir", required=True)
    parser.add_argument("--discussion_json", required=True)
    parser.add_argument("--output_dir", default="public_info_Campylobacter")
    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir).resolve()
    discussion_path = Path(args.discussion_json).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    catalogues = {}
    prefixes = {}
    prompt_hashes = {}
    for filename, role in ROLE_BY_FILE.items():
        path = prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        prefixes[role] = public_prefix(path)
        catalogues[role] = parse_catalogue(prefixes[role])
        prompt_hashes[filename] = sha256_bytes(path.read_bytes())

    first_role = next(iter(ROLE_BY_FILE.values()))
    reference = canonical_json(catalogues[first_role])
    mismatches = [role for role, value in catalogues.items() if canonical_json(value) != reference]
    if mismatches:
        raise ValueError(f"Public catalogues differ across prompts: {mismatches}")

    issues = catalogues[first_role]
    catalogue_payload = {
        "schema": "campylobacter_public_issue_catalogue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_prompt_files": list(ROLE_BY_FILE),
        "source_prompt_sha256": prompt_hashes,
        "cross_prompt_catalogue_match": True,
        "extraction_boundary": PRIVATE_MARKER,
        "private_prompt_suffix_parsed": False,
        "private_scores_included": False,
        "issues": issues,
    }
    catalogue_path = output / "public_issue_catalogue.json"
    catalogue_path.write_text(json.dumps(catalogue_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    general = extract_general_description(prefixes[first_role])
    general_path = output / "General_Description_Campylobacter.txt"
    general_path.write_text(general.rstrip() + "\n", encoding="utf-8")

    raw = json.loads(read_text(discussion_path))
    conversation = raw.get("conversation")
    if not isinstance(conversation, list):
        raise ValueError("The discussion JSON has no list-valued 'conversation' field")
    stakeholders = {role: [] for role in ROLE_BY_FILE.values()}
    excluded = []
    allowed_fields = ["speaker", "message", "id", "timestamp"]
    for index, row in enumerate(conversation):
        speaker = str(row.get("speaker", "")).strip()
        if speaker not in stakeholders:
            excluded.append({
                "conversation_index": index,
                "speaker": speaker,
                "reason": "not a negotiating stakeholder",
            })
            continue
        stakeholders[speaker].append({
            "conversation_index": index,
            "message_id": row.get("id"),
            "timestamp": row.get("timestamp"),
            "message": clean_message(row.get("message", "")),
        })
    if any(not rows for rows in stakeholders.values()):
        missing = [name for name, rows in stakeholders.items() if not rows]
        raise ValueError(f"No public conversation messages found for: {missing}")

    discussion_payload = {
        "schema": "campylobacter_preliminary_discussion_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": discussion_path.name,
        "source_sha256": sha256_bytes(discussion_path.read_bytes()),
        "source_json_field": "conversation",
        "allowed_fields_copied": allowed_fields,
        "ignored_top_level_fields": sorted(key for key in raw if key != "conversation"),
        "public_message_cleaning": [
            "remove a leading quoted previous-speaker paragraph",
            "remove embedded parenthetical Moderator note annotations",
        ],
        "excluded_non_stakeholder_items": excluded,
        "stakeholders": stakeholders,
    }
    prepared_discussion_path = output / "preliminary_discussion.json"
    prepared_discussion_path.write_text(
        json.dumps(discussion_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "schema": "campylobacter_public_input_extraction_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue_option_source": "public prefixes of all four supplied final prompts",
        "discussion_source": "only Preliminary_discussion.json/conversation",
        "private_scores_used": False,
        "catalogue_match_across_all_prompts": True,
        "outputs": {
            catalogue_path.name: sha256_bytes(catalogue_path.read_bytes()),
            general_path.name: sha256_bytes(general_path.read_bytes()),
            prepared_discussion_path.name: sha256_bytes(prepared_discussion_path.read_bytes()),
        },
    }
    (output / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(issues)} issues and {sum(map(len, stakeholders.values()))} stakeholder messages")
    print(f"Public inputs: {output}")


if __name__ == "__main__":
    main()

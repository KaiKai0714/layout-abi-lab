"""Prepare a local result bundle for a community pull request without uploading it."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .identity import identity_key, measurement_file_hashes
from .schema import MANIFEST_SCHEMA, load_json_object, sha256_file
from .validation import discover_result_bundles, validate_result

KNOWN_ENVIRONMENT_KEYS = {
    "schema",
    "schema_version",
    "created_utc",
    "layoutabi_version",
    "python",
    "platform",
    "machine",
    "container",
    "nvidia_smi",
    "torch",
}
KNOWN_EAGER_KEYS = {
    "schema",
    "schema_version",
    "graph_fingerprint",
    "device",
    "software",
    "measurement",
    "points",
}
KNOWN_COMPILE_KEYS = {"schema", "schema_version", "software", "points"}
KNOWN_MANIFEST_KEYS = {"schema", "schema_version", "files"}
KNOWN_KEYS = {
    "environment.json": KNOWN_ENVIRONMENT_KEYS,
    "eager_results.json": KNOWN_EAGER_KEYS,
    "compile_results.json": KNOWN_COMPILE_KEYS,
    "manifest.json": KNOWN_MANIFEST_KEYS,
}

BUNDLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PRIVATE_PATH_RE = re.compile(
    r"(?i)(?:[A-Za-z]:\\Users\\[^\\/]+|/home/[^/\\s\"']+|/Users/[^/\\s\"']+)"
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOSTNAME_KEY_RE = re.compile(r"(?i)^(host|hostname|computername|nodename)$")
USERNAME_KEY_RE = re.compile(r"(?i)^(user|username|login|account)$")


@dataclass
class PrivacyFinding:
    location: str
    detail: str


@dataclass
class SubmissionResult:
    source: Path
    destination: Path
    privacy_findings: list[PrivacyFinding] = field(default_factory=list)
    replicate_of: str | None = None
    exact_duplicate_of: str | None = None
    checksums_rewritten: bool = False


def rewrite_manifest_checksums(directory: Path) -> bool:
    """Recompute integrity hashes. Return True if the manifest file changed."""

    manifest_path = directory / "manifest.json"
    manifest = load_json_object(manifest_path)
    measured = [
        name
        for name in ("environment.json", "eager_results.json", "SUMMARY.md", "compile_results.json")
        if (directory / name).is_file()
    ]
    files = {name: sha256_file(directory / name) for name in measured}
    manifest["schema"] = MANIFEST_SCHEMA
    if "schema_version" not in manifest:
        manifest["schema_version"] = 1
    manifest["files"] = files
    encoded = json.dumps(manifest, indent=2) + "\n"
    previous = manifest_path.read_text(encoding="utf-8")
    if previous != encoded:
        manifest_path.write_text(encoded, encoding="utf-8")
        return True
    return False


def _walk_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def _scan_json_payload(name: str, payload: dict[str, Any]) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []
    known = KNOWN_KEYS.get(name)
    if known is not None:
        extra = sorted(str(key) for key in payload if key not in known)
        if extra:
            findings.append(
                PrivacyFinding(name, f"custom metadata keys: {', '.join(extra)}")
            )
    for key, value in payload.items():
        key_text = str(key)
        if HOSTNAME_KEY_RE.match(key_text):
            findings.append(PrivacyFinding(f"{name}.{key_text}", f"hostname field: {value!r}"))
        if USERNAME_KEY_RE.match(key_text):
            findings.append(PrivacyFinding(f"{name}.{key_text}", f"username field: {value!r}"))
    for text in _walk_strings(payload):
        for match in PRIVATE_PATH_RE.finditer(text):
            findings.append(PrivacyFinding(name, f"possible private path: {match.group(0)}"))
        for match in EMAIL_RE.finditer(text):
            findings.append(PrivacyFinding(name, f"possible email: {match.group(0)}"))
    return findings


def scan_privacy(directory: Path) -> list[PrivacyFinding]:
    """Report possible hostnames, usernames, private paths, and custom metadata."""

    findings: list[PrivacyFinding] = []
    for name in (
        "environment.json",
        "eager_results.json",
        "compile_results.json",
        "manifest.json",
        "SUMMARY.md",
    ):
        path = directory / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if name.endswith(".json"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                findings.append(PrivacyFinding(name, "could not parse JSON for privacy scan"))
                continue
            if isinstance(payload, dict):
                findings.extend(_scan_json_payload(name, payload))
        else:
            for match in PRIVATE_PATH_RE.finditer(text):
                findings.append(PrivacyFinding(name, f"possible private path: {match.group(0)}"))
            for match in EMAIL_RE.finditer(text):
                findings.append(PrivacyFinding(name, f"possible email: {match.group(0)}"))

    unique: list[PrivacyFinding] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (finding.location, finding.detail)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def _match_existing_bundle(
    source: Path, results_root: Path, destination: Path
) -> tuple[str | None, str | None]:
    environment = load_json_object(source / "environment.json")
    eager = load_json_object(source / "eager_results.json")
    manifest = load_json_object(source / "manifest.json")
    key = identity_key(environment, eager)
    hashes = measurement_file_hashes(manifest)
    replicate_of = None
    exact_of = None
    reference_roots = sorted(
        path for path in results_root.glob("reference_*") if path.is_dir()
    )
    published = discover_result_bundles(results_root / "community")
    for root in reference_roots:
        published.extend(discover_result_bundles(root))
    for bundle in published:
        if bundle.resolve() == destination.resolve() or bundle.resolve() == source.resolve():
            continue
        try:
            other_env = load_json_object(bundle / "environment.json")
            other_eager = load_json_object(bundle / "eager_results.json")
            other_manifest = load_json_object(bundle / "manifest.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if identity_key(other_env, other_eager) != key:
            continue
        relative = bundle.relative_to(results_root).as_posix()
        if replicate_of is None:
            replicate_of = relative
        if hashes and hashes == measurement_file_hashes(other_manifest):
            exact_of = relative
            break
    return replicate_of, exact_of


def prepare_submission(
    source: Path,
    name: str,
    *,
    results_root: Path,
    strict: bool = False,
    strict_privacy: bool = False,
) -> SubmissionResult:
    """Copy a validated local bundle into results/community/<name>/."""

    source = source.resolve()
    results_root = results_root.resolve()
    if not BUNDLE_NAME_RE.match(name):
        raise ValueError(
            "Bundle name must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    problems = validate_result(source, strict=strict)
    if problems:
        raise ValueError("Source bundle is invalid:\n" + "\n".join(f"- {item}" for item in problems))

    destination = results_root / "community" / name
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    checksums_rewritten = rewrite_manifest_checksums(destination)
    copied_problems = validate_result(destination, strict=strict)
    if copied_problems:
        shutil.rmtree(destination)
        raise ValueError(
            "Copied bundle failed validation:\n"
            + "\n".join(f"- {item}" for item in copied_problems)
        )

    findings = scan_privacy(destination)
    if strict_privacy and findings:
        shutil.rmtree(destination)
        details = "\n".join(f"- {item.location}: {item.detail}" for item in findings)
        raise ValueError("Privacy scan failed:\n" + details)

    replicate_of, exact_of = _match_existing_bundle(destination, results_root, destination)
    return SubmissionResult(
        source=source,
        destination=destination,
        privacy_findings=findings,
        replicate_of=replicate_of,
        exact_duplicate_of=exact_of,
        checksums_rewritten=checksums_rewritten,
    )


def format_submission_report(result: SubmissionResult) -> str:
    lines = [
        f"Prepared community bundle: {result.destination}",
        "This command does not upload anything.",
        f"Checksums rewritten: {'yes' if result.checksums_rewritten else 'no'}",
    ]
    if result.exact_duplicate_of:
        lines.append(
            f"Exact duplicate of {result.exact_duplicate_of}. "
            "Keep it only if you intentionally want a byte-identical replica."
        )
    elif result.replicate_of:
        lines.append(
            f"Replicate of {result.replicate_of} "
            "(same graph, device, software stack, and measurement protocol). "
            "It will be indexed as a replicate, not as a new device."
        )
    else:
        lines.append("No existing bundle with the same identity was found.")
    if result.privacy_findings:
        lines.append("Possible private metadata:")
        for finding in result.privacy_findings:
            lines.append(f"- {finding.location}: {finding.detail}")
        lines.append("Remove identifying fields if required by local policy, then re-run.")
    else:
        lines.append("Privacy scan: no hostname, username, private path, or extra metadata found.")
    lines.append("Next: git add the copied directory and open a pull request.")
    return "\n".join(lines)

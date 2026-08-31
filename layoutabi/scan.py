"""License, secret, and privacy scan of git-tracked files. No PyTorch import."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .submission import EMAIL_RE

REQUIRED_LICENSE_FILES = (
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "SECURITY.md",
    "CITATION.cff",
)

FORBIDDEN_TRACKED_NAMES = (
    "ROADMAP_TO_V1_PRIVATE_zh.md",
    ".env",
    "id_rsa",
    "id_ed25519",
)

ALLOWED_EMAILS = {"ethankai0714@gmail.com"}

# Require a plausible username. Do not reuse submission.PRIVATE_PATH_RE here:
# that pattern also matches its own regex source (`/home/[^...`).
SCAN_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?:[A-Za-z]:\\Users\\[A-Za-z0-9._-]+|/(?:home|Users)/[A-Za-z0-9._-]+)"
)

SKIP_PRIVATE_PATH_PREFIXES = ("tests/", "layoutabi/submission.py", "layoutabi/scan.py")

SECRET_PATTERNS = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "possible AWS access key"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"), "private key block"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "possible GitHub token"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "possible GitHub token"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "possible Slack token"),
)

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".whl",
    ".pyc",
    ".pyd",
    ".so",
    ".dll",
    ".exe",
}


@dataclass(frozen=True)
class ScanFinding:
    location: str
    detail: str


def _repo_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "layoutabi scan-release requires git metadata; run it from a clone"
        )
    names = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    return [root / name for name in names]


def _is_text(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\0" not in chunk


def _scan_text(relative: str, text: str) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for pattern, label in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(ScanFinding(relative, f"{label}: {match.group(0)[:24]}…"))
    if not relative.startswith(SKIP_PRIVATE_PATH_PREFIXES):
        for match in SCAN_PRIVATE_PATH_RE.finditer(text):
            findings.append(
                ScanFinding(relative, f"possible private path: {match.group(0)}")
            )
    for match in EMAIL_RE.finditer(text):
        email = match.group(0).lower()
        if email not in ALLOWED_EMAILS:
            findings.append(ScanFinding(relative, f"possible email: {match.group(0)}"))
    return findings


def scan_release(root: Path | None = None) -> list[ScanFinding]:
    """Scan git-tracked files for license gaps, secrets, and private metadata."""

    base = _repo_root(root)
    findings: list[ScanFinding] = []
    for name in REQUIRED_LICENSE_FILES:
        if not (base / name).is_file():
            findings.append(ScanFinding(name, "required license or security file is missing"))

    files = tracked_files(base)
    tracked_names = {path.relative_to(base).as_posix() for path in files}
    for forbidden in FORBIDDEN_TRACKED_NAMES:
        if forbidden in tracked_names or any(
            posix.endswith("/" + forbidden) for posix in tracked_names
        ):
            findings.append(ScanFinding(forbidden, "private or credential file must not be tracked"))

    for path in files:
        relative = path.relative_to(base).as_posix()
        if not path.is_file() or not _is_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(_scan_text(relative, text))

    unique: list[ScanFinding] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (finding.location, finding.detail)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def format_findings(findings: Iterable[ScanFinding]) -> str:
    lines = [f"{item.location}: {item.detail}" for item in findings]
    return "\n".join(lines)

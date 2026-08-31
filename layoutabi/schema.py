"""JSON Schema contracts, schema_version, and forward migration for result documents."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

ENVIRONMENT_SCHEMA = "layoutabi_environment_v1"
EAGER_SCHEMA = "layoutabi_eager_v1"
COMPILE_SCHEMA = "layoutabi_compile_v1"
MANIFEST_SCHEMA = "layoutabi_manifest_v1"
INDEX_SCHEMA = "layoutabi_results_index_v1"
AUDIT_SCHEMA = "layoutabi_compile_audit_v1"
DIAGNOSTICS_SCHEMA = "layoutabi_optimizer_diagnostics_v1"

CURRENT_VERSIONS = {
    ENVIRONMENT_SCHEMA: 1,
    EAGER_SCHEMA: 1,
    COMPILE_SCHEMA: 1,
    MANIFEST_SCHEMA: 1,
    INDEX_SCHEMA: 1,
    AUDIT_SCHEMA: 1,
    DIAGNOSTICS_SCHEMA: 1,
}

SCHEMA_FILES = {
    ENVIRONMENT_SCHEMA: "environment.schema.json",
    EAGER_SCHEMA: "eager.schema.json",
    COMPILE_SCHEMA: "compile.schema.json",
    MANIFEST_SCHEMA: "manifest.schema.json",
    INDEX_SCHEMA: "index.schema.json",
    AUDIT_SCHEMA: "audit.schema.json",
    DIAGNOSTICS_SCHEMA: "diagnostics.schema.json",
}

KIND_SCHEMAS = {
    "environment": ENVIRONMENT_SCHEMA,
    "eager": EAGER_SCHEMA,
    "compile": COMPILE_SCHEMA,
    "manifest": MANIFEST_SCHEMA,
    "index": INDEX_SCHEMA,
    "audit": AUDIT_SCHEMA,
    "diagnostics": DIAGNOSTICS_SCHEMA,
}

_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

# Future breaking changes register a migrator that produces the next version.
# The integer key is the version being migrated *to*.
MIGRATIONS: dict[str, dict[int, Callable[[dict[str, Any]], dict[str, Any]]]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def current_version(schema_name: str) -> int:
    try:
        return CURRENT_VERSIONS[schema_name]
    except KeyError as exc:
        raise ValueError(f"Unknown schema name: {schema_name}") from exc


@lru_cache(maxsize=None)
def load_json_schema(schema_name: str) -> dict[str, Any]:
    try:
        filename = SCHEMA_FILES[schema_name]
    except KeyError as exc:
        raise ValueError(f"Unknown schema name: {schema_name}") from exc
    path = _SCHEMA_DIR / filename
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"JSON Schema at {path} is not an object")
    return schema


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def validate_schema_instance(
    instance: Any, schema: dict[str, Any], pointer: str = "$"
) -> list[str]:
    """Validate ``instance`` against a restricted JSON Schema dialect."""

    problems: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_is_type(instance, item) for item in allowed):
            problems.append(f"{pointer}: expected type {expected_type}")
            return problems

    if "const" in schema and instance != schema["const"]:
        problems.append(f"{pointer}: expected const {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        problems.append(f"{pointer}: expected one of {schema['enum']!r}")
    numeric = isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if "minimum" in schema and numeric and instance < schema["minimum"]:
        problems.append(f"{pointer}: expected minimum {schema['minimum']}")
    if "minLength" in schema and isinstance(instance, str):
        if len(instance) < schema["minLength"]:
            problems.append(f"{pointer}: expected minLength {schema['minLength']}")
    if "pattern" in schema and isinstance(instance, str):
        if re.search(schema["pattern"], instance) is None:
            problems.append(f"{pointer}: expected pattern {schema['pattern']}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                problems.append(f"{pointer}: missing required property {key!r}")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            minimum = schema["minProperties"]
            problems.append(f"{pointer}: expected minProperties {minimum}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child = f"{pointer}.{key}"
            if key in properties:
                problems.extend(validate_schema_instance(value, properties[key], child))
            elif additional is False:
                problems.append(f"{pointer}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                problems.extend(validate_schema_instance(value, additional, child))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            problems.append(f"{pointer}: expected minItems {schema['minItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, value in enumerate(instance):
                problems.extend(
                    validate_schema_instance(value, items, f"{pointer}[{index}]")
                )

    if "anyOf" in schema:
        options = schema["anyOf"]
        if not any(not validate_schema_instance(instance, option, pointer) for option in options):
            problems.append(f"{pointer}: did not match anyOf")

    return problems


def migrate_document(payload: dict[str, Any], expected_schema: str) -> dict[str, Any]:
    """Return a deep-copied document migrated to the current schema version."""

    if expected_schema not in CURRENT_VERSIONS:
        raise ValueError(f"Unknown schema name: {expected_schema}")
    migrated = copy.deepcopy(payload)
    raw_version = migrated.get("schema_version", 1)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int) or raw_version < 1:
        raise ValueError(f"Invalid schema_version: {raw_version!r}")
    target = CURRENT_VERSIONS[expected_schema]
    if raw_version > target:
        raise ValueError(
            f"{expected_schema} schema_version {raw_version} is newer than supported {target}"
        )
    version = raw_version
    while version < target:
        version += 1
        migrator = MIGRATIONS.get(expected_schema, {}).get(version)
        if migrator is None:
            raise ValueError(f"No migrator for {expected_schema} -> {version}")
        migrated = migrator(migrated)
        migrated["schema_version"] = version
    migrated["schema_version"] = target
    return migrated


def normalize_document(
    payload: dict[str, Any], expected_schema: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Migrate and JSON-Schema-validate a document. Return (migrated, problems)."""

    name = payload.get("schema")
    if name != expected_schema:
        return None, [f"Unsupported schema {name!r}; expected {expected_schema}"]
    try:
        migrated = migrate_document(payload, expected_schema)
    except ValueError as exc:
        return None, [str(exc)]
    problems = validate_schema_instance(migrated, load_json_schema(expected_schema))
    return migrated, problems


def migrate_bundle_documents(directory: Path, *, write: bool = False) -> dict[str, dict[str, Any]]:
    """Forward-migrate bundle JSON documents in memory, optionally writing them back."""

    mapping = {
        "environment.json": ENVIRONMENT_SCHEMA,
        "eager_results.json": EAGER_SCHEMA,
        "manifest.json": MANIFEST_SCHEMA,
    }
    compile_path = directory / "compile_results.json"
    if compile_path.is_file():
        mapping["compile_results.json"] = COMPILE_SCHEMA

    migrated_files: dict[str, dict[str, Any]] = {}
    for name, schema_name in mapping.items():
        payload = load_json_object(directory / name)
        migrated, problems = normalize_document(payload, schema_name)
        if migrated is None or problems:
            detail = "; ".join(problems) if problems else "unknown schema problem"
            raise ValueError(f"{directory / name}: {detail}")
        migrated_files[name] = migrated

    if write:
        for name, payload in migrated_files.items():
            if name == "manifest.json":
                continue
            path = directory / name
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, indent=2) + "\n")
        manifest = migrated_files["manifest.json"]
        names = set(manifest.get("files", {}))
        names.update(name for name in migrated_files if name != "manifest.json")
        if (directory / "SUMMARY.md").is_file():
            names.add("SUMMARY.md")
        manifest["files"] = {
            name: sha256_file(directory / name)
            for name in sorted(names)
            if (directory / name).is_file()
        }
        manifest_path = directory / "manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
        migrated_files["manifest.json"] = load_json_object(manifest_path)
    return migrated_files

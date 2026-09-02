"""Deterministically aggregate validated reference and community result bundles."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .identity import (
    environment_summary,
    graph_fingerprint,
    identity_key,
    protocol_fingerprint,
)
from .schema import INDEX_SCHEMA, current_version, normalize_document, sha256_file
from .validation import discover_result_bundles, validate_result


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def discover_publishable_bundles(results_root: Path) -> list[Path]:
    """Return reference and accepted community bundles, excluding local measurements."""

    bundles: list[Path] = []
    for section in ("reference_l40s", "community"):
        root = results_root / section
        if root.is_dir():
            bundles.extend(discover_result_bundles(root))
    return sorted(set(bundles))


def _median(cell: Any) -> float | None:
    if not isinstance(cell, dict):
        return None
    value = cell.get("median_ms")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _ratio(cell: Any) -> float | None:
    if not isinstance(cell, dict):
        return None
    value = cell.get("direct_over_repair_kv")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _outcome(ratio: float | None) -> str:
    if ratio is None:
        return "unavailable"
    if ratio > 1.0:
        return "repair_win"
    if ratio < 1.0:
        return "direct_win"
    return "parity"


def _compiled_outcome(direct: dict[str, Any], repair: dict[str, Any], ratio: float | None) -> str:
    if not (direct.get("available") and repair.get("available")):
        return "unavailable"
    return _outcome(ratio)


def _alignment_tokens(names: Any) -> list[str]:
    if not isinstance(names, list):
        return []
    tokens: set[str] = set()
    for name in names:
        lowered = str(name).lower()
        tokens.update(match.group(0) for match in re.finditer(r"align\d+", lowered))
        if "ldg8" in lowered:
            tokens.add("ldg8")
    return sorted(tokens)


def _kernel_families(profiler: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not isinstance(profiler, dict):
        return result
    for policy, cell in profiler.items():
        names = cell.get("selected_cuda_names", []) if isinstance(cell, dict) else []
        families = _alignment_tokens(names)
        if families:
            result[str(policy)] = families
    return result


def _alignment_label(profiler: Any, policy: str) -> str:
    if not isinstance(profiler, dict):
        return "—"
    cell = profiler.get(policy)
    if not isinstance(cell, dict):
        return "—"
    names = cell.get("selected_cuda_names")
    tokens = _alignment_tokens(names)
    if tokens:
        return "+".join(tokens)
    if isinstance(names, list) and names:
        return "no-align-token"
    return "—"


def _n_mod_8(point: dict[str, Any]) -> int | None:
    raw = point.get("n_mod_8")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    consumer = point.get("consumer_n")
    if isinstance(consumer, int) and not isinstance(consumer, bool):
        return int(consumer) % 8
    return None


def _safety_action(dtype: Any, n_mod_8: int | None) -> str:
    if n_mod_8 is None:
        return "—"
    name = str(dtype or "").lower()
    if name not in {"fp16", "float16", "torch.float16"}:
        return "direct (non-fp16)"
    return "direct" if n_mod_8 == 0 else "repair_kv"


def _residue_tier(dtype: Any, consumer_n: Any) -> str:
    if not isinstance(consumer_n, int) or isinstance(consumer_n, bool):
        return "—"
    name = str(dtype or "").lower()
    if name not in {"fp16", "float16", "torch.float16"}:
        return "unclassified"
    if consumer_n % 8 == 0:
        return "fastest: align8/ldg8"
    if consumer_n % 2 == 0:
        return "intermediate: align2"
    return "slowest: align1"


def _oracle_action(outcome: str | None) -> str:
    if outcome == "repair_win":
        return "repair_kv"
    if outcome == "direct_win":
        return "direct"
    if outcome == "unavailable":
        return "unavailable"
    if outcome == "parity":
        return "parity"
    return "—"


def _compiled_by_resolution(compiled: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if compiled is None:
        return {}
    return {
        int(point["resolution"]): point
        for point in compiled.get("points", [])
        if isinstance(point, dict) and isinstance(point.get("resolution"), int)
    }


def _source_from_relative(relative: str) -> tuple[str, str]:
    section = relative.split("/", 1)[0]
    source = "reference" if section == "reference_l40s" else "community"
    return section, source


def _bundle_record(bundle: Path, results_root: Path) -> dict[str, Any]:
    environment = _load(bundle / "environment.json")
    eager = _load(bundle / "eager_results.json")
    compile_path = bundle / "compile_results.json"
    compiled = _load(compile_path) if compile_path.is_file() else None
    compiled_points = _compiled_by_resolution(compiled)
    rows = []
    kernel_families: dict[str, list[str]] = {}

    for point in eager.get("points", []):
        if not isinstance(point, dict) or not isinstance(point.get("resolution"), int):
            continue
        resolution = int(point["resolution"])
        aggregate = point.get("aggregate", {})
        chain = aggregate.get("chain", {}) if isinstance(aggregate, dict) else {}
        module = aggregate.get("module", {}) if isinstance(aggregate, dict) else {}
        compiled_point = compiled_points.get(resolution, {})
        policies = compiled_point.get("policies", {}) if isinstance(compiled_point, dict) else {}
        direct_compiled = policies.get("direct", {}) if isinstance(policies, dict) else {}
        repair_compiled = policies.get("repair_kv", {}) if isinstance(policies, dict) else {}
        if not isinstance(direct_compiled, dict):
            direct_compiled = {}
        if not isinstance(repair_compiled, dict):
            repair_compiled = {}
        compiled_ratio = _ratio(compiled_point)
        n_mod_8 = _n_mod_8(point)
        isolated_profiler = point.get("ktv_profiler")
        profiler = isolated_profiler if isinstance(isolated_profiler, dict) else point.get(
            "module_profiler"
        )
        profiler_source = "isolated_ktv" if isinstance(isolated_profiler, dict) else (
            "legacy_full_module" if isinstance(profiler, dict) else "unavailable"
        )
        rows.append(
            {
                "resolution": resolution,
                "consumer_n": point.get("consumer_n"),
                "n_mod_8": n_mod_8,
                "residue_tier_prior": _residue_tier(
                    point.get("dtype"), point.get("consumer_n")
                ),
                "safety_policy_action": _safety_action(point.get("dtype"), n_mod_8),
                "profiler_source": profiler_source,
                "direct_alignment_tokens": _alignment_label(profiler, "direct"),
                "repair_alignment_tokens": _alignment_label(profiler, "repair_kv"),
                "dtype": point.get("dtype"),
                "batch": point.get("batch"),
                "eager_chain": {
                    "direct_ms": _median(chain.get("direct")),
                    "repair_k_ms": _median(chain.get("repair_k")),
                    "repair_kv_ms": _median(chain.get("repair_kv")),
                    "ratio": _ratio(chain),
                    "outcome": _outcome(_ratio(chain)),
                    "repair_wins_all_seeds": chain.get("repair_kv_wins_all_seeds"),
                },
                "eager_module": {
                    "direct_ms": _median(module.get("direct")),
                    "repair_k_ms": _median(module.get("repair_k")),
                    "repair_kv_ms": _median(module.get("repair_kv")),
                    "ratio": _ratio(module),
                    "outcome": _outcome(_ratio(module)),
                    "repair_wins_all_seeds": module.get("repair_kv_wins_all_seeds"),
                },
                "compiled_module": {
                    "direct_available": bool(direct_compiled.get("available", False)),
                    "repair_available": bool(repair_compiled.get("available", False)),
                    "direct_ms": _median(direct_compiled.get("steady_state")),
                    "repair_kv_ms": _median(repair_compiled.get("steady_state")),
                    "ratio": compiled_ratio
                    if direct_compiled.get("available") and repair_compiled.get("available")
                    else None,
                    "outcome": _compiled_outcome(
                        direct_compiled, repair_compiled, compiled_ratio
                    ),
                    "correct": bool(
                        direct_compiled.get("correctness", {}).get("pass", False)
                        and repair_compiled.get("correctness", {}).get("pass", False)
                    ),
                },
            }
        )
        rows[-1]["eager_oracle"] = _oracle_action(rows[-1]["eager_module"]["outcome"])
        rows[-1]["compiled_oracle"] = _oracle_action(
            rows[-1]["compiled_module"]["outcome"]
        )
        for policy, families in _kernel_families(profiler).items():
            kernel_families[policy] = sorted(set(kernel_families.get(policy, [])).union(families))

    relative = bundle.relative_to(results_root).as_posix()
    category, source = _source_from_relative(relative)
    return {
        "id": relative,
        "category": category,
        "source": source,
        "role": source,
        "identity_key": identity_key(environment, eager),
        "replicate_of": None,
        "graph_fingerprint": graph_fingerprint(eager),
        "protocol_fingerprint": protocol_fingerprint(eager),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "environment": environment_summary(environment),
        "kernel_families": kernel_families,
        "rows": sorted(rows, key=lambda row: row["resolution"], reverse=True),
    }


def _assign_roles(records: list[dict[str, Any]]) -> None:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["identity_key"], []).append(index)
    for indices in groups.values():
        ranked = sorted(
            indices,
            key=lambda index: (
                0 if records[index]["source"] == "reference" else 1,
                records[index]["id"],
            ),
        )
        primary = records[ranked[0]]
        if primary["source"] == "reference":
            primary["role"] = "reference"
        else:
            primary["role"] = "community"
        primary["replicate_of"] = None
        for index in ranked[1:]:
            records[index]["role"] = "replicate"
            records[index]["replicate_of"] = primary["id"]


def _primary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["role"] != "replicate"]


def _stack_tuple(environment: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        environment.get("torch"),
        environment.get("cuda_build"),
        environment.get("cudnn"),
    )


def _filters(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for record in records for row in record["rows"]]
    stacks = sorted(
        {
            _stack_tuple(record["environment"])
            for record in records
        },
        key=lambda item: (str(item[0]), str(item[1]), str(item[2])),
    )
    return {
        "roles": sorted({record["role"] for record in records}),
        "devices": sorted({record["environment"]["device"] for record in records}),
        "dtypes": sorted(
            {str(row["dtype"]) for row in rows if row.get("dtype") is not None}
        ),
        "stacks": [
            {"torch": torch, "cuda_build": cuda_build, "cudnn": cudnn}
            for torch, cuda_build, cudnn in stacks
        ],
        "resolutions": sorted({int(row["resolution"]) for row in rows}),
        "outcomes": {
            "eager_module": sorted({row["eager_module"]["outcome"] for row in rows}),
            "compiled_module": sorted({row["compiled_module"]["outcome"] for row in rows}),
        },
    }


def build_index(results_root: Path) -> dict[str, Any]:
    """Validate and aggregate every publishable bundle below ``results_root``."""

    bundles = discover_publishable_bundles(results_root)
    if not bundles:
        raise ValueError(f"No publishable result bundles found below {results_root}")

    problems: list[str] = []
    for bundle in bundles:
        strict = "reference_l40s" in bundle.relative_to(results_root).parts
        for problem in validate_result(bundle, strict=strict):
            problems.append(f"{bundle}: {problem}")
    if problems:
        raise ValueError("Invalid result bundles:\n" + "\n".join(problems))

    records = [_bundle_record(bundle, results_root) for bundle in bundles]
    _assign_roles(records)
    primaries = _primary_records(records)
    rows = [row for record in records for row in record["rows"]]
    primary_rows = [row for record in primaries for row in record["rows"]]
    compiled_outcomes = [row["compiled_module"]["outcome"] for row in primary_rows]
    eager_outcomes = [row["eager_module"]["outcome"] for row in primary_rows]
    device_keys = {
        (record["environment"]["device"], record["environment"]["compute_capability"])
        for record in primaries
    }
    stack_keys = {_stack_tuple(record["environment"]) for record in primaries}
    index = {
        "schema": INDEX_SCHEMA,
        "schema_version": current_version(INDEX_SCHEMA),
        "summary": {
            "bundles": len(records),
            "reference_bundles": sum(record["role"] == "reference" for record in records),
            "community_bundles": sum(record["role"] == "community" for record in records),
            "replicate_bundles": sum(record["role"] == "replicate" for record in records),
            "devices": len(device_keys),
            "software_stacks": len(stack_keys),
            "measurement_rows": len(rows),
            "primary_measurement_rows": len(primary_rows),
            "eager_module_repair_wins": eager_outcomes.count("repair_win"),
            "eager_module_direct_wins": eager_outcomes.count("direct_win"),
            "eager_module_unavailable": eager_outcomes.count("unavailable"),
            "compiled_module_repair_wins": compiled_outcomes.count("repair_win"),
            "compiled_module_direct_wins": compiled_outcomes.count("direct_win"),
            "compiled_module_unavailable": compiled_outcomes.count("unavailable"),
        },
        "filters": _filters(records),
        "bundles": records,
    }
    _, schema_problems = normalize_document(index, INDEX_SCHEMA)
    if schema_problems:
        raise ValueError("Generated result index failed schema validation:\n" + "\n".join(schema_problems))
    return index


def _format_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}x"


def _format_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _measurement_row(
    record: dict[str, Any],
    row: dict[str, Any],
    link: str,
    extra: str = "",
) -> str:
    environment = record["environment"]
    label = record["id"].replace("reference_l40s/", "reference/")
    eager = row["eager_module"]
    compiled = row["compiled_module"]
    n_mod = row.get("n_mod_8")
    n_mod_text = "—" if n_mod is None else str(n_mod)
    return (
        f"| [{label}]({link}){extra} | {environment['device']} | "
        f"{environment['torch']} / {environment['cuda_build']} | "
        f"{row['resolution']} | {n_mod_text} | "
        f"{row.get('residue_tier_prior') or '—'} | "
        f"{row.get('profiler_source') or '—'} | "
        f"{row.get('direct_alignment_tokens') or '—'} | "
        f"{row.get('repair_alignment_tokens') or '—'} | "
        f"{row.get('safety_policy_action') or '—'} | "
        f"{row.get('eager_oracle') or '—'} | {_format_ratio(eager['ratio'])} | "
        f"{row.get('compiled_oracle') or '—'} | {_format_ratio(compiled['ratio'])} |"
    )


def _section_rows(
    records: list[dict[str, Any]],
    results_root: Path,
    output_path: Path,
    *,
    include_primary: bool = False,
) -> list[str]:
    header = (
        "| Bundle | Device | PyTorch / CUDA | Res | N%8 | FP16 residue tier prior | Profiler | "
        "Direct observed token(s) | Repair observed token(s) | Safety action | Eager oracle | "
        "Eager ratio | Compiled oracle | Compiled ratio |"
    )
    separator = "|---|---|---|---:|---:|---|---|---|---|---|---|---:|---|---:|"
    lines = [header, separator]
    for record in records:
        bundle_path = results_root / record["id"] / "SUMMARY.md"
        link = Path(os.path.relpath(bundle_path, output_path.parent)).as_posix()
        extra = ""
        if include_primary and record.get("replicate_of"):
            primary = record["replicate_of"].replace("reference_l40s/", "reference/")
            extra = f" (replicate of {primary})"
        for row in record["rows"]:
            lines.append(_measurement_row(record, row, link, extra))
    if len(lines) == 2:
        return ["None."]
    return lines


def render_markdown(index: dict[str, Any], results_root: Path, output_path: Path) -> str:
    """Render a stable human-readable index with links to every bundle summary."""

    summary = index["summary"]
    filters = index["filters"]
    stacks = ", ".join(
        f"{item['torch']} / {item['cuda_build']}" for item in filters["stacks"]
    ) or "—"
    lines = [
        "# Layout ABI result index",
        "",
        "This file is generated from checksum-validated reference and community bundles.",
        "The FP16 mechanism prior has three residue tiers: N divisible by 8 maps to",
        "align8/ldg8, even non-multiples of 8 to align2, and odd N to align1.",
        "Tokens are extracted from profiler names, not portable GEMM-family identifiers.",
        "The safety action remains binary: direct for N%8==0, otherwise repair.",
        "Oracle and ratio are whether materialization paid off at full-module scope;",
        "a ratio above 1 means repair-KV was faster. Replicates are not extra devices.",
        "",
        "## Coverage",
        "",
        "| Bundles | Reference | Community | Replicates | Devices | Software stacks | "
        "Primary rows | All rows |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['bundles']} | {summary['reference_bundles']} | "
        f"{summary['community_bundles']} | {summary['replicate_bundles']} | "
        f"{summary['devices']} | {summary['software_stacks']} | "
        f"{summary['primary_measurement_rows']} | {summary['measurement_rows']} |",
        "",
        "## Outcomes",
        "",
        "Outcome counts use primary bundles only.",
        "",
        "| Scope | Repair wins | Direct wins | Unavailable |",
        "|---|---:|---:|---:|",
        f"| Eager full module | {summary['eager_module_repair_wins']} | "
        f"{summary['eager_module_direct_wins']} | {summary['eager_module_unavailable']} |",
        f"| Compiled full module | {summary['compiled_module_repair_wins']} | "
        f"{summary['compiled_module_direct_wins']} | "
        f"{summary['compiled_module_unavailable']} |",
        "",
        "## Filters",
        "",
        "| Dimension | Values |",
        "|---|---|",
        f"| Role | {', '.join(filters['roles']) or '—'} |",
        f"| Device | {', '.join(filters['devices']) or '—'} |",
        f"| Dtype | {', '.join(filters['dtypes']) or '—'} |",
        f"| Stack | {stacks} |",
        f"| Resolution | {', '.join(str(item) for item in filters['resolutions']) or '—'} |",
        f"| Eager outcome | {', '.join(filters['outcomes']['eager_module']) or '—'} |",
        f"| Compiled outcome | {', '.join(filters['outcomes']['compiled_module']) or '—'} |",
        "",
        "## Reference measurements",
        "",
    ]
    reference = [record for record in index["bundles"] if record["role"] == "reference"]
    community = [record for record in index["bundles"] if record["role"] == "community"]
    replicates = [record for record in index["bundles"] if record["role"] == "replicate"]
    lines.extend(_section_rows(reference, results_root, output_path))
    lines.extend(["", "## Community measurements", ""])
    lines.extend(_section_rows(community, results_root, output_path))
    lines.extend(["", "## Replicate measurements", ""])
    lines.extend(_section_rows(replicates, results_root, output_path, include_primary=True))
    lines.extend(
        [
            "",
            "## Standalone mechanism audits",
            "",
            "These validator-backed artifacts are excluded from workload, device, and",
            "profitability-row counts because they are factorial mechanism controls:",
            "",
            "- [L40S compiled six-shape audit](results/reference_l40s/compile_audit/torch2.11_cuda12.8/SUMMARY.md)",
            "- [L40S 100-cell operand-pointer audit](results/reference_l40s/pointer_alignment/torch2.11_cuda12.8/SUMMARY.md)",
            "- [Orin 100-cell operand-pointer audit](results/community/orin_pointer_alignment/torch2.7_cuda12.8/SUMMARY.md)",
            "",
            "## Interpretation boundary",
            "",
            "The scientific object is producer layout → vendor GEMM family, not a single",
            "named operator. Public LinearAttention graphs are witnesses. The dedicated",
            "L40S sweep spans fastest/intermediate/slowest residue classes; older 128/256",
            "rows are retained as legacy profitability evidence, not isolated KTV proof.",
            "Positive and negative outcomes are both evidence. Replicates are not extra",
            "devices. Compiled-unavailable is not a direct/repair loss. Matching L40S",
            "and Orin pointer audits reproduce the bounded least-aligned-tier rule in",
            "all 200 controlled cells.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_index(
    *, results_root: Path, output_json: Path, output_markdown: Path, check: bool = False
) -> None:
    """Write the deterministic index or fail if committed outputs are stale."""

    index = build_index(results_root)
    json_text = json.dumps(index, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(index, results_root, output_markdown)
    outputs = ((output_json, json_text), (output_markdown, markdown_text))
    if check:
        stale = [path for path, text in outputs if _read_text(path) != text]
        if stale:
            names = ", ".join(str(path) for path in stale)
            raise ValueError(f"Generated result index is stale: {names}")
        return
    for path, text in outputs:
        _write_text(path, text)


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

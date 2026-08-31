"""CPU-only tests for JSON Schema validation and forward migration."""

from __future__ import annotations

import unittest

from layoutabi.schema import (
    DIAGNOSTICS_SCHEMA,
    ENVIRONMENT_SCHEMA,
    load_json_schema,
    migrate_document,
    normalize_document,
    validate_schema_instance,
)


class SchemaTest(unittest.TestCase):
    def test_type_const_and_required(self) -> None:
        schema = {
            "type": "object",
            "required": ["schema"],
            "additionalProperties": False,
            "properties": {"schema": {"const": "layoutabi_environment_v1"}},
        }
        self.assertEqual(
            validate_schema_instance({"schema": "layoutabi_environment_v1"}, schema),
            [],
        )
        problems = validate_schema_instance({"schema": "other", "extra": 1}, schema)
        self.assertTrue(any("const" in item for item in problems))
        self.assertTrue(any("unexpected property" in item for item in problems))

    def test_integer_does_not_accept_bool(self) -> None:
        schema = {"type": "integer"}
        self.assertEqual(validate_schema_instance(1, schema), [])
        self.assertTrue(validate_schema_instance(True, schema))

    def test_missing_schema_version_migrates_to_current(self) -> None:
        migrated = migrate_document({"schema": ENVIRONMENT_SCHEMA}, ENVIRONMENT_SCHEMA)
        self.assertEqual(migrated["schema_version"], 1)

    def test_newer_schema_version_is_not_silently_read(self) -> None:
        with self.assertRaisesRegex(ValueError, "newer than supported"):
            migrate_document(
                {"schema": ENVIRONMENT_SCHEMA, "schema_version": 99},
                ENVIRONMENT_SCHEMA,
            )

    def test_diagnostics_schema_rejects_wrong_name(self) -> None:
        schema = load_json_schema(DIAGNOSTICS_SCHEMA)
        self.assertEqual(schema["properties"]["schema"]["const"], DIAGNOSTICS_SCHEMA)
        problems = validate_schema_instance(
            {
                "schema": DIAGNOSTICS_SCHEMA,
                "schema_version": 1,
                "pattern_id": "linear_attention_ktv_v1",
                "decision": "noop",
                "reason": "unsupported",
            },
            schema,
        )
        self.assertEqual(problems, [])

    def test_wrong_schema_name_is_mismatch(self) -> None:
        migrated, problems = normalize_document(
            {"schema": "layoutabi_environment_v9"}, ENVIRONMENT_SCHEMA
        )
        self.assertIsNone(migrated)
        self.assertTrue(any("Unsupported schema" in item for item in problems))


if __name__ == "__main__":
    unittest.main()

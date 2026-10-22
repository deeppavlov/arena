import math

import pytest

from evaluator.response_schema import validate_response_content, validate_response_schema


def score_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string", "minLength": 1, "pattern": r"\S"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize("score", [1, 3, 5])
def test_accepts_valid_score_response(score: int) -> None:
    validator = validate_response_schema(score_schema())
    validate_response_content(f'{{"score":{score},"reason":" Верно. "}}', validator)


@pytest.mark.parametrize(
    "content",
    [
        '{"score":true,"reason":"valid"}',
        '{"score":"1","reason":"valid"}',
        '{"score":null,"reason":"valid"}',
        '{"score":0,"reason":"valid"}',
        '{"score":6,"reason":"valid"}',
        '{"score":1,"reason":""}',
        '{"score":1,"reason":" \\t\\n"}',
        '{"score":1,"reason":"\\u2003"}',
        '{"score":1}',
        '{"score":1,"reason":"valid","extra":true}',
        '{"score":NaN,"reason":"valid"}',
        '{"score":Infinity,"reason":"valid"}',
        '{"score":-Infinity,"reason":"valid"}',
        '{"score":1e400,"reason":"valid"}',
        '{"score":1,"score":5,"reason":"valid"}',
        '{"score":1,"reason":"a","reason":"b"}',
        '```json\n{"score":1,"reason":"valid"}\n```',
        '{"score":1,"reason":"valid"} trailing',
        '[{"score":1,"reason":"valid"}]',
        '',
    ],
)
def test_rejects_invalid_score_response(content: str) -> None:
    with pytest.raises(ValueError):
        validate_response_content(content, validate_response_schema(score_schema()))


@pytest.mark.parametrize(
    "schema",
    [
        None,
        True,
        [],
        {"type": "unknown"},
        {"type": "string", "minLength": -1},
        {"type": "object", "required": ["x", "x"]},
        {"type": "object", "required": "score"},
        {"type": "object", "additionalProperties": "false"},
        {"type": "object", "properties": []},
        {"type": "integer", "minimum": True},
        {"type": "object", "properties": {"x": "invalid"}},
        {"$ref": "https://example.invalid/schema"},
        {"$ref": "file:///etc/passwd"},
        {"$ref": "#"},
        {"$defs": {"x": {"type": "string"}}},
        {"type": "string", "pattern": "(a+)+$"},
        {"type": "string", "format": "uri"},
        {"type": "object", "properties": {"x": {"$ref": "https://example.invalid/"}}},
        {"type": "number", "minimum": math.nan},
        {"type": "number", "maximum": math.inf},
        {"description": "a" * 32769},
        {"enum": list(range(3000))},
        {"enum": (1, 2)},
        {1: "string"},
    ],
)
def test_rejects_invalid_or_unsafe_schema(schema: object) -> None:
    with pytest.raises(ValueError):
        validate_response_schema(schema)


def test_rejects_deep_and_recursive_schema() -> None:
    schema: dict[str, object] = {"type": "string"}
    for _ in range(20):
        schema = {"type": "array", "items": schema}
    with pytest.raises(ValueError):
        validate_response_schema(schema)
    schema["items"] = schema
    with pytest.raises(ValueError):
        validate_response_schema(schema)


def test_response_schema_snapshot_is_not_mutated_by_caller() -> None:
    schema = score_schema()
    validator = validate_response_schema(schema)
    schema["required"] = []
    with pytest.raises(ValueError):
        validate_response_content('{}', validator)


def test_supports_bounded_arrays_objects_and_enumerations() -> None:
    validator = validate_response_schema(
        {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"enum": ["yes", "no"]},
                    "value": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "label": {"type": "string", "maxLength": 3},
                    "fixed": {"const": None},
                },
                "additionalProperties": False,
            },
        }
    )
    validate_response_content('[{"kind":"yes","value":0.5,"label":"abc","fixed":null}]', validator)
    for content in ('[]', '[{"kind":"maybe"}]', '[{"value":true}]', '[{"value":0}]'):
        with pytest.raises(ValueError):
            validate_response_content(content, validator)


def test_rejects_nonfinite_json_even_without_numeric_constraint() -> None:
    validator = validate_response_schema({"type": "object"})
    with pytest.raises(ValueError):
        validate_response_content('{"extra":1e400}', validator)


def test_property_names_can_match_forbidden_schema_keywords() -> None:
    validator = validate_response_schema(
        {"type": "object", "properties": {"$ref": {"type": "string"}}}
    )
    validate_response_content('{"$ref":"https://example.invalid"}', validator)


def test_schema_depth_and_node_limits() -> None:
    schema: dict[str, object] = {"type": "string"}
    for _ in range(15):
        schema = {"items": schema}
    validate_response_schema(schema)
    with pytest.raises(ValueError):
        validate_response_schema({"items": schema})
    validate_response_schema({"enum": list(range(2046))})
    with pytest.raises(ValueError):
        validate_response_schema({"enum": list(range(2047))})


def test_response_nesting_limit() -> None:
    validator = validate_response_schema({"type": "array"})
    validate_response_content('[' * 64 + '0' + ']' * 64, validator)
    with pytest.raises(ValueError):
        validate_response_content('[' * 65 + '0' + ']' * 65, validator)


def test_rejects_extremely_nested_response_as_value_error() -> None:
    validator = validate_response_schema({"type": "array"})
    with pytest.raises(ValueError):
        validate_response_content('[' * 2000 + '0' + ']' * 2000, validator)


@pytest.mark.parametrize("number", ["1", "1.0", "1e0", "3.0"])
def test_integer_schema_uses_standard_json_schema_semantics(number: str) -> None:
    validator = validate_response_schema({"type": "integer"})
    validate_response_content(number, validator)


@pytest.mark.parametrize("number", ["1.0", "1e0", "true", "1.5"])
def test_strict_integer_policy_is_explicit(number: str) -> None:
    validator = validate_response_schema({"type": "integer"}, strict_integers=True)
    validate_response_content("1", validator)
    with pytest.raises(ValueError):
        validate_response_content(number, validator)

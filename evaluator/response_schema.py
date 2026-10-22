import json
import math
from typing import Any

from jsonschema import Draft202012Validator, SchemaError, ValidationError, validators
from jsonschema.protocols import Validator

MAX_SCHEMA_BYTES = 32768
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_NODES = 2048
MAX_RESPONSE_DEPTH = 64
SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "enum",
    "const",
    "title",
    "description",
    "pattern",
}
STRICT_VALIDATOR = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda checker, instance: type(instance) is int
    ),
)


def _check_schema_bounds(schema: object) -> None:
    pending = [(schema, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if depth > MAX_SCHEMA_DEPTH or nodes > MAX_SCHEMA_NODES:
            raise ValueError("Response schema exceeds structural limits")
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("Response schema keys must be strings")
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
        elif value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError("Response schema must contain finite JSON values")


def _check_schema_keywords(schema: object) -> None:
    if not isinstance(schema, dict) or schema.keys() - SCHEMA_KEYWORDS:
        raise ValueError("Unsupported response schema keyword or schema type")
    if "pattern" in schema and schema["pattern"] != r"\S":
        raise ValueError("Only the nonblank string pattern is supported")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Response schema properties must be an object")
    for child in properties.values():
        _check_schema_keywords(child)
    if "items" in schema:
        _check_schema_keywords(schema["items"])
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        _check_schema_keywords(additional)


def validate_response_schema(schema: object, *, strict_integers: bool = False) -> Validator:
    _check_schema_bounds(schema)
    try:
        serialized = json.dumps(schema, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        raise ValueError("Response schema must contain finite JSON values") from None
    if len(serialized) > MAX_SCHEMA_BYTES:
        raise ValueError("Response schema exceeds size limit")
    snapshot = json.loads(serialized)
    _check_schema_keywords(snapshot)
    try:
        Draft202012Validator.check_schema(snapshot)
    except SchemaError:
        raise ValueError("Invalid response schema") from None
    validator = STRICT_VALIDATOR if strict_integers else Draft202012Validator
    return validator(snapshot)


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Response JSON numbers must be finite")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError("Response JSON numbers must be finite")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Response JSON contains duplicate keys")
        result[key] = value
    return result


def _check_response_depth(value: object) -> None:
    pending = [(value, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > MAX_RESPONSE_DEPTH:
            raise ValueError("Response JSON exceeds nesting limit")
        if isinstance(value, dict):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)


def validate_response_content(content: str, validator: Validator) -> None:
    try:
        value = json.loads(
            content,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
        _check_response_depth(value)
        validator.validate(value)
    except (ValueError, RecursionError, ValidationError):
        raise ValueError("Model response does not match response schema") from None

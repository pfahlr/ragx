"""STDIO validation filter for MCP JSON-RPC envelopes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from jsonschema import ValidationError

from apps.mcp_server.service.errors import CanonicalError
from apps.mcp_server.validation import SchemaRegistry

__all__ = ["ValidationFilter"]


class ValidationFilter:
    """Validate JSON-RPC envelopes on ingress and egress."""

    def __init__(
        self,
        *,
        schema_registry: SchemaRegistry,
        mode: str = "shadow",
        logger: logging.Logger | None = None,
    ) -> None:
        self._schema_registry = schema_registry
        self._mode = mode
        self._logger = logger or logging.getLogger(__name__)

    def check_request(self, payload: Mapping[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        """Validate an incoming JSON-RPC request envelope."""

        try:
            self._schema_registry.load_envelope().validate(dict(payload))
        except ValidationError as exc:
            self._log_failure("ingress", payload, exc)
            if self._mode.lower() == "enforce":
                error = CanonicalError.to_jsonrpc_error("INVALID_INPUT")
                error["message"] = "Request failed envelope validation"
                data_field = error.get("data")
                if isinstance(data_field, dict):
                    payload_details: dict[str, Any] = dict(data_field)
                else:
                    payload_details = {}
                payload_details["details"] = {
                    "schemaPath": [str(part) for part in exc.schema_path],
                    "instancePath": [str(part) for part in exc.absolute_path],
                    "message": exc.message,
                }
                error["data"] = payload_details
                return False, error
        return True, None

    def check_response(self, payload: Mapping[str, Any]) -> None:
        """Validate an outgoing envelope and log failures."""

        try:
            self._schema_registry.load_envelope().validate(dict(payload))
        except ValidationError as exc:
            self._log_failure("egress", payload, exc)

    def _log_failure(
        self,
        stage: str,
        payload: Mapping[str, Any],
        error: ValidationError,
    ) -> None:
        self._logger.warning(
            "STDIO envelope validation failure",  # noqa: EM102
            extra={
                "stage": stage,
                "mode": self._mode,
                "schema_path": list(error.schema_path),
                "instance_path": list(error.absolute_path),
                "message": error.message,
                "method": payload.get("method"),
            },
        )

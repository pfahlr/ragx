"""FastAPI middleware enforcing MCP envelope validation on HTTP responses."""

from __future__ import annotations

import json
import logging
from typing import Any

from jsonschema import ValidationError
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from apps.mcp_server.service.errors import CanonicalError
from apps.mcp_server.validation import SchemaRegistry

__all__ = ["EnvelopeValidationMiddleware"]


class EnvelopeValidationMiddleware(BaseHTTPMiddleware):
    """Validate MCP envelopes on HTTP egress and optionally enforce failures."""

    def __init__(
        self,
        app,
        *,
        schema_registry: SchemaRegistry,
        mode: str = "shadow",
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(app)
        self._schema_registry = schema_registry
        self._mode = mode
        self._logger = logger or logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        cloned, body = await self._clone_response(response)
        if not body or "application/json" not in (cloned.media_type or ""):
            return cloned
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return cloned
        try:
            self._schema_registry.load_envelope().validate(payload)
        except ValidationError as exc:
            self._handle_failure(request, payload, exc)
            if self._mode.lower() == "enforce":
                canonical_code = "INVALID_OUTPUT"
                error = CanonicalError.to_jsonrpc_error(canonical_code)
                error["message"] = "Envelope validation failed"
                data_field = error.get("data")
                if isinstance(data_field, dict):
                    enriched: dict[str, Any] = dict(data_field)
                else:
                    enriched = {}
                enriched["details"] = {
                    "schemaPath": [str(part) for part in exc.schema_path],
                    "instancePath": [str(part) for part in exc.absolute_path],
                    "message": exc.message,
                }
                error["data"] = enriched
                payload = {
                    "ok": False,
                    "error": {
                        "code": canonical_code,
                        "message": error["message"],
                        "details": enriched,
                    },
                    "meta": payload.get("meta", {}),
                }
                status_code = CanonicalError.to_http_status(canonical_code)
                return JSONResponse(payload, status_code=status_code)
        return cloned

    async def _clone_response(self, response: Response) -> tuple[Response, bytes]:
        body = b""
        if response.body_iterator is None:
            body = getattr(response, "body", b"") or b""
        else:
            async for chunk in response.body_iterator:
                body += chunk
            response.body_iterator = iterate_in_threadpool(iter([body]))
        headers = dict(response.headers)
        headers["content-length"] = str(len(body))
        cloned = Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )
        return cloned, body

    def _handle_failure(
        self,
        request: Request,
        payload: dict[str, Any],
        error: ValidationError,
    ) -> None:
        self._logger.warning(
            "Envelope validation failure",  # noqa: EM102 - structured kwargs
            extra={
                "path": request.url.path,
                "mode": self._mode,
                "schema_path": list(error.schema_path),
                "instance_path": list(error.absolute_path),
                "message": error.message,
                "status": payload.get("status"),
            },
        )

"""Best-effort detection of a model's context-window size.

Providers expose this information inconsistently (verified against their
current API documentation):

* Anthropic  ``GET /v1/models/{model}`` returns ``max_input_tokens``.
* Ollama     ``POST /api/show`` returns ``model_info.<arch>.context_length``.
* Kilo gateway, OpenRouter, LM Studio, vLLM and similar OpenAI-compatible
  servers return ``context_length`` (or ``max_context_length`` /
  ``max_model_len`` / ``context_window``) from their model list endpoint.
* OpenAI's own ``GET /v1/models`` returns only ``id``/``created``/``object``/
  ``owned_by``; there is no context field, so a small local catalog is used
  for well-known OpenAI model families.

Detection is optional and strictly best-effort: every network path is wrapped
so a failure returns ``None`` and the caller keeps the user's configured
value. Results are cached in memory per ``(provider, base_url, model)`` so a
session detects each model at most once.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
import ssl
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000
_DETECT_TIMEOUT = 10.0

#: Minimal model catalog for the official OpenAI API, which does not report a
#: context window. Values are best-effort and can always be overridden in the
#: settings dialog; prefixes are matched longest-first against the model name.
_OPENAI_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("gpt-6", 1_050_000),
    ("gpt-5.6", 1_050_000),
    ("gpt-5.4", 400_000),
    ("gpt-5.2", 400_000),
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_047_576),
    ("chatgpt-4o", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5", 16_385),
    ("o4", 200_000),
    ("o3", 200_000),
    ("o1", 200_000),
)

#: In-memory cache keyed by ``(provider, base_url, model)``.
_CACHE: dict[tuple[str, str, str], int] = {}

_FetchJson = Callable[..., Any]


def clear_cache() -> None:
    """Forget all cached detections (used by tests and after settings changes)."""
    _CACHE.clear()


def _coerce_int(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` if it is not usable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(float(value.strip()))
        except (TypeError, ValueError):
            return None
    else:
        return None
    return number if number > 0 else None


def _normalize(model: str) -> str:
    return (model or "").strip().lower()


def _model_matches(candidate: str, model: str) -> bool:
    """Match a catalog id against a requested model name.

    Handles exact ids, provider-prefixed ids (``openai/gpt-5.4``), and variant
    suffixes (``gpt-5.4:free``) in either direction.
    """
    cand = _normalize(candidate)
    want = _normalize(model)
    if not cand or not want:
        return False
    if cand == want:
        return True
    cand_base = cand.split(":", 1)[0]
    want_base = want.split(":", 1)[0]
    if cand_base == want_base:
        return True
    if cand_base.endswith("/" + want_base) or want_base.endswith("/" + cand_base):
        return True
    return False


def _openai_item_context(item: dict[str, Any]) -> int | None:
    """Read the first usable context field from one model-list entry."""
    for key in ("context_length", "max_context_length", "max_model_len", "context_window"):
        tokens = _coerce_int(item.get(key))
        if tokens:
            return tokens
    top_provider = item.get("top_provider")
    if isinstance(top_provider, dict):
        return _coerce_int(top_provider.get("context_length"))
    return None


def parse_openai_models(payload: Any, model: str) -> int | None:
    """Extract the context window for ``model`` from a ``/models`` response."""
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and _model_matches(item.get("id", ""), model):
            tokens = _openai_item_context(item)
            if tokens:
                return tokens
    return None


def parse_anthropic_model(payload: Any) -> int | None:
    """Extract ``max_input_tokens`` from an Anthropic model response."""
    if isinstance(payload, dict):
        return _coerce_int(payload.get("max_input_tokens"))
    return None


def parse_ollama_show(payload: Any) -> int | None:
    """Extract the model's maximum context length from ``/api/show``.

    ``model_info`` reports the architecture's maximum; the ``parameters``
    block reports the configured ``num_ctx`` and is only used as a fallback.
    """
    if not isinstance(payload, dict):
        return None
    model_info = payload.get("model_info")
    if isinstance(model_info, dict):
        for key in ("context_length", "max_position_embeddings", "n_ctx"):
            tokens = _coerce_int(model_info.get(key))
            if tokens:
                return tokens
        for key, value in model_info.items():
            if key.endswith((".context_length", ".max_position_embeddings")):
                tokens = _coerce_int(value)
                if tokens:
                    return tokens
    params = payload.get("parameters")
    if isinstance(params, str):
        for line in params.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "num_ctx":
                return _coerce_int(parts[1])
    return None


def openai_catalog_context(model: str) -> int | None:
    """Look up a well-known OpenAI model in the local best-effort catalog."""
    tail = _normalize(model).rsplit("/", 1)[-1].split(":", 1)[0]
    for prefix, tokens in sorted(_OPENAI_CONTEXT_WINDOWS, key=lambda p: -len(p[0])):
        if tail.startswith(prefix):
            return tokens
    return None


def _api_root(base_url: str, default: str) -> str:
    """Strip a chat/completion path suffix to get the provider's API root."""
    base = (base_url or "").strip().rstrip("/") or default
    for suffix in ("/chat/completions", "/messages", "/api/chat", "/api/generate"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/")


def _openai_models_urls(base_url: str) -> list[str]:
    """Candidate ``/models`` URLs, trying both with and without ``/v1``."""
    root = _api_root(base_url, "https://api.openai.com/v1")
    urls = [f"{root}/models"]
    if not root.endswith("/v1") and "/v1/" not in root:
        urls.append(f"{root}/v1/models")
    return urls


def _anthropic_model_url(base_url: str, model: str) -> str:
    root = _api_root(base_url, "https://api.anthropic.com")
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    return f"{root}/models/{urllib.parse.quote(model, safe='')}"


def _ollama_show_url(base_url: str) -> str:
    return f"{_api_root(base_url, 'http://localhost:11434')}/api/show"


def _default_fetch_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout: float = _DETECT_TIMEOUT,
) -> Any:
    """Perform one JSON HTTP request (kept small and dependency-free)."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    if url.lower().startswith("https"):
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001 -- fall back to the platform trust store
            context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def _openai_headers(settings: Any) -> dict[str, str]:
    key = getattr(settings, "llm_api_key", "") or ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _anthropic_headers(settings: Any) -> dict[str, str]:
    key = getattr(settings, "llm_api_key", "") or ""
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key:
        headers["x-api-key"] = key
    return headers


def detect_context_tokens(
    settings: Any,
    *,
    fetch_json: _FetchJson | None = None,
    timeout: float = _DETECT_TIMEOUT,
) -> int | None:
    """Detect the configured model's context window, or ``None``.

    Honours ``settings.llm_context_auto``; returns ``None`` when auto-detection
    is disabled or the provider does not expose the value. Never raises.
    """
    if not getattr(settings, "llm_context_auto", True):
        return None

    provider = (getattr(settings, "llm_provider", "") or "openai").strip().lower()
    model = (getattr(settings, "llm_model", "") or "").strip()
    if not model:
        return None
    base_url = (getattr(settings, "llm_base_url", "") or "").strip().rstrip("/")

    cache_key = (provider, base_url, model)
    cached = _CACHE.get(cache_key)
    if cached:
        return cached

    fetch = fetch_json or _default_fetch_json
    try:
        tokens = _detect_by_provider(provider, base_url, model, settings, fetch, timeout)
    except Exception as e:  # noqa: BLE001 -- detection must never break a chat
        log.debug("Context window detection failed for %s/%s: %s", provider, model, e)
        tokens = None

    if tokens and tokens > 0:
        _CACHE[cache_key] = tokens
        log.info("Detected context window for %s/%s: %s tokens", provider, model, f"{tokens:,}")
        return tokens
    log.debug("Could not detect context window for %s/%s; using configured value", provider, model)
    return None


def _detect_by_provider(
    provider: str,
    base_url: str,
    model: str,
    settings: Any,
    fetch: _FetchJson,
    timeout: float,
) -> int | None:
    if provider == "anthropic":
        payload = fetch(
            _anthropic_model_url(base_url, model),
            headers=_anthropic_headers(settings),
            timeout=timeout,
        )
        return parse_anthropic_model(payload)

    if provider == "ollama":
        payload = fetch(
            _ollama_show_url(base_url),
            method="POST",
            headers={"Content-Type": "application/json"},
            body={"model": model},
            timeout=timeout,
        )
        return parse_ollama_show(payload)

    # OpenAI and any OpenAI-compatible server (Kilo gateway, OpenRouter,
    # vLLM, LM Studio, ...): try each candidate model-list URL in turn.
    for url in _openai_models_urls(base_url):
        try:
            payload = fetch(url, headers=_openai_headers(settings), timeout=timeout)
        except Exception as e:  # noqa: BLE001 -- try the next candidate URL
            log.debug("Model list request failed (%s): %s", url, e)
            continue
        tokens = parse_openai_models(payload, model)
        if tokens:
            return tokens

    # The official OpenAI API does not report a context window; consult the
    # local catalog so common models still get an accurate value.
    return openai_catalog_context(model)

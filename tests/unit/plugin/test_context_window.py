"""Tests for best-effort context-window detection (no network)."""

from __future__ import annotations

import types

import pytest

from kicad_plugin import context_window as cw


@pytest.fixture(autouse=True)
def _clear_cache():
    cw.clear_cache()
    yield
    cw.clear_cache()


def _settings(**overrides):
    base = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "llm_base_url": "",
        "llm_api_key": "test-key",
        "llm_context_auto": True,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _FakeFetch:
    """Records calls and returns canned payloads keyed by URL (or a callable)."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, *, method="GET", headers=None, body=None, timeout=0):
        self.calls.append({"url": url, "method": method, "headers": headers, "body": body})
        if callable(self.responses):
            return self.responses(url, method)
        return self.responses


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_openai_models_exact_match():
    payload = {
        "data": [
            {"id": "gpt-5.4", "context_length": 400_000},
            {"id": "other", "context_length": 8_000},
        ]
    }
    assert cw.parse_openai_models(payload, "gpt-5.4") == 400_000


def test_parse_openai_models_provider_prefix_and_suffix():
    payload = {"data": [{"id": "openai/gpt-5.4", "context_length": 400_000}]}
    assert cw.parse_openai_models(payload, "gpt-5.4") == 400_000
    assert cw.parse_openai_models(payload, "openai/gpt-5.4:free") == 400_000


def test_parse_openai_models_top_provider_fallback():
    payload = {
        "data": [{"id": "some/model", "top_provider": {"context_length": 1_000_000}}],
    }
    assert cw.parse_openai_models(payload, "some/model") == 1_000_000


def test_parse_openai_models_no_match_or_field():
    assert cw.parse_openai_models({"data": [{"id": "a"}]}, "b") is None
    assert cw.parse_openai_models({"data": [{"id": "b"}]}, "b") is None
    assert cw.parse_openai_models(None, "b") is None


def test_parse_anthropic_model():
    assert cw.parse_anthropic_model({"max_input_tokens": 1_000_000}) == 1_000_000
    assert cw.parse_anthropic_model({"max_input_tokens": None}) is None
    assert cw.parse_anthropic_model([]) is None


def test_parse_ollama_show_arch_prefixed():
    payload = {"model_info": {"llama.context_length": 131_072}}
    assert cw.parse_ollama_show(payload) == 131_072


def test_parse_ollama_show_parameters_fallback():
    payload = {"parameters": "temperature 0.7\nnum_ctx 8192\n"}
    assert cw.parse_ollama_show(payload) == 8192


def test_parse_ollama_show_prefers_model_info_over_parameters():
    payload = {
        "model_info": {"qwen2.context_length": 32_768},
        "parameters": "num_ctx 4096",
    }
    assert cw.parse_ollama_show(payload) == 32_768


def test_openai_catalog_longest_prefix_and_unknown():
    assert cw.openai_catalog_context("gpt-4o") == 128_000
    assert cw.openai_catalog_context("openai/gpt-4o-mini") == 128_000
    assert cw.openai_catalog_context("gpt-6-astra") == 1_050_000
    assert cw.openai_catalog_context("llama3.2") is None


# ---------------------------------------------------------------------------
# URL derivation
# ---------------------------------------------------------------------------


def test_openai_models_urls_kilo_gateway():
    urls = cw._openai_models_urls("https://api.kilo.ai/api/gateway/chat/completions")
    assert urls[0] == "https://api.kilo.ai/api/gateway/models"


def test_openai_models_urls_openrouter_and_default():
    assert cw._openai_models_urls("https://openrouter.ai/api/v1/chat/completions") == [
        "https://openrouter.ai/api/v1/models"
    ]
    assert cw._openai_models_urls("") == ["https://api.openai.com/v1/models"]


def test_anthropic_and_ollama_urls():
    assert (
        cw._anthropic_model_url("https://api.anthropic.com/v1/messages", "claude-opus-5")
        == "https://api.anthropic.com/v1/models/claude-opus-5"
    )
    assert cw._ollama_show_url("") == "http://localhost:11434/api/show"
    assert (
        cw._ollama_show_url("http://localhost:11434/api/chat") == "http://localhost:11434/api/show"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_detect_disabled_returns_none_without_calling_provider():
    def _boom(*args, **kwargs):
        raise AssertionError("fetch must not run when auto-detect is off")

    assert cw.detect_context_tokens(_settings(llm_context_auto=False), fetch_json=_boom) is None


def test_detect_openai_compatible_gateway():
    settings = _settings(
        llm_model="kilo-auto/efficient",
        llm_base_url="https://api.kilo.ai/api/gateway/chat/completions",
    )
    fetch = _FakeFetch({"data": [{"id": "kilo-auto/efficient", "context_length": 1_000_000}]})

    tokens = cw.detect_context_tokens(settings, fetch_json=fetch)

    assert tokens == 1_000_000
    assert fetch.calls[0]["url"] == "https://api.kilo.ai/api/gateway/models"


def test_detect_anthropic_uses_model_endpoint():
    settings = _settings(
        llm_provider="anthropic",
        llm_model="claude-opus-5",
        llm_base_url="https://api.anthropic.com/v1/messages",
    )
    fetch = _FakeFetch({"max_input_tokens": 1_000_000})

    tokens = cw.detect_context_tokens(settings, fetch_json=fetch)

    assert tokens == 1_000_000
    assert fetch.calls[0]["url"] == "https://api.anthropic.com/v1/models/claude-opus-5"
    assert fetch.calls[0]["headers"]["anthropic-version"] == "2023-06-01"


def test_detect_ollama_posts_show_request():
    settings = _settings(
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        llm_base_url="http://localhost:11434",
    )
    fetch = _FakeFetch({"model_info": {"qwen2.context_length": 32_768}})

    tokens = cw.detect_context_tokens(settings, fetch_json=fetch)

    assert tokens == 32_768
    assert fetch.calls[0]["method"] == "POST"
    assert fetch.calls[0]["body"] == {"model": "qwen2.5:7b"}


def test_detect_openai_official_falls_back_to_catalog():
    settings = _settings(llm_model="gpt-4o", llm_base_url="")
    fetch = _FakeFetch({"data": [{"id": "gpt-4o", "object": "model"}]})

    tokens = cw.detect_context_tokens(settings, fetch_json=fetch)

    assert tokens == 128_000
    assert fetch.calls[0]["url"] == "https://api.openai.com/v1/models"


def test_detect_result_is_cached_per_model():
    settings = _settings(
        llm_model="kilo-auto/efficient",
        llm_base_url="https://api.kilo.ai/api/gateway/chat/completions",
    )
    fetch = _FakeFetch({"data": [{"id": "kilo-auto/efficient", "context_length": 1_000_000}]})

    assert cw.detect_context_tokens(settings, fetch_json=fetch) == 1_000_000
    assert cw.detect_context_tokens(settings, fetch_json=fetch) == 1_000_000
    assert len(fetch.calls) == 1


def test_detect_swallows_provider_errors():
    settings = _settings()

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    # No context field and a matching catalog entry still resolves; an unknown
    # model with a failing fetch degrades to None.
    assert cw.detect_context_tokens(settings, fetch_json=_boom) == 128_000
    assert cw.detect_context_tokens(_settings(llm_model="unknown-model"), fetch_json=_boom) is None

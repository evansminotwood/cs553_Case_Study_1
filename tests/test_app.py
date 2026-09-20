import sys
import types
from unittest.mock import MagicMock

import pytest

FAKE_LOCAL_TEXT = "Always split aces and eights, kid."
FAKE_REMOTE_TEXT = "Hit on 16, stand on 17."


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeInferenceClient:
    """Stands in for huggingface_hub.InferenceClient in tests."""

    def __init__(self, *args, **kwargs):
        pass

    def chat_completion(self, *args, **kwargs):
        for word in FAKE_REMOTE_TEXT.split(" "):
            yield _FakeChunk(word + " ")


class _FailingInferenceClient:
    """Simulates a real remote failure (e.g. rate limit, timeout, HTTP error)."""

    def __init__(self, *args, **kwargs):
        pass

    def chat_completion(self, *args, **kwargs):
        raise RuntimeError("429 Too Many Requests")
        yield  # pragma: no cover - makes this a generator


def _install_stub_modules():
    """Stub out the GPU-only deps so app.py imports fast on any runner."""
    fake_pipe = MagicMock(
        return_value=[
            {"generated_text": [{"role": "assistant", "content": FAKE_LOCAL_TEXT}]}
        ]
    )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = MagicMock(return_value=fake_pipe)
    sys.modules["transformers"] = fake_transformers

    fake_spaces = types.ModuleType("spaces")
    fake_spaces.GPU = lambda fn: fn
    sys.modules["spaces"] = fake_spaces

    return fake_pipe


@pytest.fixture
def app():
    sys.modules.pop("app", None)
    fake_pipe = _install_stub_modules()
    import app as app_module

    yield app_module, fake_pipe

    sys.modules.pop("app", None)
    sys.modules.pop("transformers", None)
    sys.modules.pop("spaces", None)


class _FakeToken:
    token = "fake-hf-token"


def test_local_generate_returns_pipeline_text(app):
    app_module, fake_pipe = app

    result = app_module.local_generate(
        [{"role": "user", "content": "16 vs dealer 10, what do I do?"}],
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
    )

    assert result == FAKE_LOCAL_TEXT
    fake_pipe.assert_called_once()


def test_respond_local_mode_yields_model_output(app):
    app_module, _ = app

    outputs = list(
        app_module.respond(
            "16 vs dealer 10, what do I do?",
            [],
            "You are a professional gambler.",
            64,
            0.7,
            0.9,
            True,  # use_local_model
            False,  # simulate_remote_outage
            None,  # hf_token not needed locally
        )
    )

    assert FAKE_LOCAL_TEXT in outputs[-1]
    assert "Local Model" in outputs[-1]


def test_respond_api_mode_without_login_prompts_user(app):
    app_module, _ = app

    outputs = list(
        app_module.respond(
            "16 vs dealer 10, what do I do?",
            [],
            "You are a professional gambler.",
            64,
            0.7,
            0.9,
            False,  # use_local_model
            False,  # simulate_remote_outage
            None,  # no hf_token -> should ask user to log in
        )
    )

    assert outputs == ["⚠️ Please log in with your Hugging Face account first."]


def test_respond_api_mode_success_labels_remote(app, monkeypatch):
    app_module, _ = app
    monkeypatch.setattr(app_module, "InferenceClient", _FakeInferenceClient)

    outputs = list(
        app_module.respond(
            "16 vs dealer 10, what do I do?",
            [],
            "You are a professional gambler.",
            64,
            0.7,
            0.9,
            False,  # use_local_model
            False,  # simulate_remote_outage
            _FakeToken(),
        )
    )

    assert "Remote API" in outputs[-1]
    assert FAKE_REMOTE_TEXT.strip() in outputs[-1]


def test_respond_simulated_outage_falls_back_to_local(app, monkeypatch):
    app_module, fake_pipe = app
    monkeypatch.setattr(app_module, "InferenceClient", _FakeInferenceClient)

    outputs = list(
        app_module.respond(
            "16 vs dealer 10, what do I do?",
            [],
            "You are a professional gambler.",
            64,
            0.7,
            0.9,
            False,  # use_local_model
            True,  # simulate_remote_outage
            _FakeToken(),
        )
    )

    assert "unavailable" in outputs[-1].lower()
    assert "Local Model" in outputs[-1]
    assert "fallback" in outputs[-1].lower()
    assert FAKE_LOCAL_TEXT in outputs[-1]
    fake_pipe.assert_called_once()


def test_respond_real_remote_failure_falls_back_to_local(app, monkeypatch):
    app_module, fake_pipe = app
    monkeypatch.setattr(app_module, "InferenceClient", _FailingInferenceClient)

    outputs = list(
        app_module.respond(
            "16 vs dealer 10, what do I do?",
            [],
            "You are a professional gambler.",
            64,
            0.7,
            0.9,
            False,  # use_local_model
            False,  # simulate_remote_outage (failure comes from the client itself)
            _FakeToken(),
        )
    )

    assert "unavailable" in outputs[-1].lower()
    assert "Local Model" in outputs[-1]
    assert FAKE_LOCAL_TEXT in outputs[-1]
    fake_pipe.assert_called_once()

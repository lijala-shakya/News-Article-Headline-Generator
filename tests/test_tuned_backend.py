from types import SimpleNamespace

from apps import groq_client, local_model_client


def test_should_use_local_model_when_enabled(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    monkeypatch.setenv("USE_TUNED_MODEL", "true")
    monkeypatch.setenv("LOCAL_TUNED_ADAPTER_DIR", str(adapter_dir))

    assert groq_client.should_use_local_model() is True


def test_generate_raw_headlines_falls_back_to_groq_when_local_model_unavailable(monkeypatch):
    monkeypatch.setattr(groq_client, "should_use_local_model", lambda: False)

    class DummyClient:
        class _Completions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"headlines": ["A"]}'))]
                )

        completions = _Completions()

    monkeypatch.setattr(groq_client, "get_client", lambda: DummyClient())

    result = groq_client.generate_raw_headlines(
        article_text="Some article text",
        style="neutral",
        num_candidates=1,
        max_chars=50,
    )

    assert result == ["A"]


def test_generate_local_headlines_uses_standard_beam_search(monkeypatch):
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, output_ids, skip_special_tokens=True):
            return "headline"

    class DummyModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [[4, 5, 6]]

    tokenizer = DummyTokenizer()
    model = DummyModel()

    monkeypatch.setattr(local_model_client, "_load_local_model", lambda: (tokenizer, model))

    result = local_model_client.generate_local_headlines("Some article", num_candidates=3)

    assert result == ["headline"]
    assert model.calls[0]["num_return_sequences"] == 3
    assert model.calls[0]["num_beam_groups"] == 1
    assert model.calls[0]["diversity_penalty"] == 0.0
    assert model.calls[0]["trust_remote_code"] is True

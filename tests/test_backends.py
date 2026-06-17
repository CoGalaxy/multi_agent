from verifiable_multi_agent.backends import DeepSeekBackend, OllamaBackend
import httpx


def test_deepseek_backend_uses_role_specific_judge_model(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    backend = DeepSeekBackend(
        api_key="sk-test",
        model="deepseek-v4-flash",
        judge_model="deepseek-v4-pro",
    )

    assert backend.complete("system", "user", role="verifier") == "ok"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["json"]["model"] == "deepseek-v4-pro"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_backend_retries_transient_connect_error(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ok after retry"}}]}

    def fake_post(url, json, headers, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("temporary tls failure")
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    backend = DeepSeekBackend(api_key="sk-test", max_retries=1)
    backend.retry_sleep = 0

    assert backend.complete("system", "user", role="executor") == "ok after retry"
    assert calls["count"] == 2


def test_ollama_backend_requests_json_mode(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": '{"claim":"ok"}'}}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    backend = OllamaBackend(model="qwen3.5:4b")

    assert backend.complete("system", "user", role="synthesizer") == '{"claim":"ok"}'
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["format"] == "json"
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_predict"] == 4096

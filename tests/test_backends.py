from verifiable_multi_agent.backends import DeepSeekBackend


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

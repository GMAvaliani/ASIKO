"""12. Работа без сети и API-ключа; адаптер внешней модели и строгая проверка его ответов."""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from asiko.model.project import Project
from asiko.nlp.llm import LlmAdapter, LlmConfig, LlmError, extract_json
from asiko.nlp.secrets import SecretStore


class MockServer:
    """Имитация Anthropic Messages API и OpenAI-совместимого API."""

    def __init__(self, reply):
        self.reply = reply   # функция (path, body) -> (status, json-строка текста ответа модели)
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, dict(self.headers), body))
                status, text = outer.reply(self.path, body)
                if self.path.endswith("/v1/messages"):
                    resp = {"id": "msg_x", "type": "message", "role": "assistant", "model": body["model"],
                            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
                else:
                    resp = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
                data = json.dumps(resp if status == 200 else {"error": {"message": text}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


GOOD = {"schema": "asiko.plan/1", "base_revision": 0, "kind": "edit",
        "operations": [{"op": "track.add", "params": {"name": "Новая"}}]}


@pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
def test_valid_plan_accepted_and_origin_forced(provider):
    srv = MockServer(lambda path, body: (200, json.dumps(dict(GOOD, origin="ui"))))
    sec = SecretStore()
    cfg = LlmConfig(provider=provider, endpoint=srv.url, model="claude-opus-5-5" if provider == "anthropic" else "m")
    sec.set(sec.account(provider, srv.url), "sk-test-secret-123")
    plan = LlmAdapter(cfg, sec).plan("добавь дорожку", Project())
    assert plan["origin"] == "llm"           # модель не может выдать себя за интерфейс
    assert plan["operations"][0]["op"] == "track.add"
    path, headers, body = srv.requests[0]
    sent = json.dumps(body, ensure_ascii=False)
    assert "sk-test-secret-123" not in sent      # ключ только в заголовке
    assert "добавь дорожку" in sent
    assert ".wav" not in sent and "sha256" not in sent   # пути и контрольные суммы не отправляются
    srv.close()


def test_invalid_answer_repaired_once_then_rejected():
    answers = iter([json.dumps({"schema": "asiko.plan/1", "base_revision": 0, "kind": "edit",
                                "operations": [{"op": "system.shell", "params": {"cmd": "rm -rf ~"}}]}),
                    json.dumps({"schema": "asiko.plan/1", "base_revision": 0, "kind": "edit",
                                "operations": [{"op": "track.set", "target": {"track_id": "x"}, "params": {"gain_db": 500}}]})])
    srv = MockServer(lambda p, b: (200, next(answers)))
    sec = SecretStore()
    cfg = LlmConfig(provider="openai_compatible", endpoint=srv.url, model="m")
    with pytest.raises(LlmError) as e:
        LlmAdapter(cfg, sec).plan("сделай что-нибудь", Project())
    assert "не прошёл проверку" in str(e.value)
    assert len(srv.requests) == 2   # одна попытка исправления
    srv.close()


def test_structured_output_fallback():
    def reply(path, body):
        if body.get("response_format", {}).get("type") in ("json_schema", "json_object"):
            return 400, "unsupported"
        return 200, "Вот план:\n```json\n" + json.dumps(GOOD) + "\n```"

    srv = MockServer(reply)
    ad = LlmAdapter(LlmConfig(provider="openai_compatible", endpoint=srv.url, model="m"), SecretStore())
    plan = ad.plan("x", Project())
    assert ad.mode == "text" and plan["kind"] == "edit"
    srv.close()


def test_no_key_and_no_network(monkeypatch):
    sec = SecretStore()
    ad = LlmAdapter(LlmConfig(provider="anthropic", endpoint="https://api.anthropic.com", model="claude-opus-5-5"), sec)
    ok, why = ad.available()
    assert not ok and "ключ" in why.lower()
    with pytest.raises(LlmError):
        ad.plan("x", Project())

    def no_net(*a, **k):
        raise OSError("сеть отключена")

    monkeypatch.setattr(socket, "create_connection", no_net)
    ad2 = LlmAdapter(LlmConfig(provider="openai_compatible", endpoint="http://198.51.100.1:9/v1", model="m"), sec)
    with pytest.raises(LlmError) as e:
        ad2.plan("x", Project())
    assert "Нет соединения" in str(e.value) and "продолжают работать" in str(e.value)


def test_bad_key_characters_reported():
    sec = SecretStore()
    cfg = LlmConfig(provider="openai_compatible", endpoint="http://127.0.0.1:9/v1", model="m")
    sec.set(sec.account(cfg.provider, cfg.endpoint), "ключ с пробелом")
    ad = LlmAdapter(cfg, sec)
    ok, why = ad.available()
    assert not ok and "недопустимые символы" in why
    with pytest.raises(LlmError):
        ad.plan("x", Project())


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("текст\n```json\n{\"a\": 2}\n```") == {"a": 2}
    with pytest.raises(LlmError):
        extract_json("нет json")


def test_key_not_in_project_or_settings(tmp_path, work):
    from asiko.storage.project_io import save_project

    sec = SecretStore()
    sec.set(sec.account("anthropic", "https://api.anthropic.com"), "sk-very-secret")
    p = work / "проект.asiko"
    save_project(Project(), p)
    assert "sk-very-secret" not in p.read_text("utf-8")


def test_keyring_integration_with_fake_backend(monkeypatch):
    """Путь через системное хранилище секретов (подменённый backend keyring)."""
    import keyring
    from keyring.backend import KeyringBackend

    class FakeKeyring(KeyringBackend):
        priority = 10
        name = "Тестовое хранилище"

        def __init__(self):
            super().__init__()
            self.data = {}

        def get_password(self, service, username):
            return self.data.get((service, username))

        def set_password(self, service, username, password):
            self.data[(service, username)] = password

        def delete_password(self, service, username):
            self.data.pop((service, username), None)

    fake = FakeKeyring()
    old = keyring.get_keyring()
    keyring.set_keyring(fake)
    monkeypatch.delenv("ASIKO_NO_KEYRING", raising=False)
    try:
        sec = SecretStore()
        assert sec.secure and sec.backend_name == "Тестовое хранилище"
        acc = sec.account("anthropic", "https://api.anthropic.com/")
        assert sec.set(acc, "sk-abc") is True
        assert fake.data[("ASIKO", "anthropic|https://api.anthropic.com")] == "sk-abc"
        assert SecretStore().get(acc) == "sk-abc"
        sec.delete(acc)
        assert sec.get(acc) is None
    finally:
        keyring.set_keyring(old)

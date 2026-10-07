"""Адаптер внешней языковой модели.

Модель только предлагает JSON-план (asiko.plan/1). План всегда проходит ту же строгую
проверку схемы и проекта, что и локальные команды, независимо от того, поддерживает ли
провайдер structured output. Модель не выполняет команды ОС, не генерирует код
обработки и не имеет доступа к файловой системе.

Провайдеры:
* anthropic — Anthropic Messages API через официальный SDK `anthropic`.
* openai_compatible — любой сервер с OpenAI-совместимым /chat/completions
  (другие провайдеры, локальные модели); запросы через стандартную библиотеку.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..commands.schema import plan_json_schema, validate_plan_schema
from .local_parser import Selection
from .prompt import project_summary, system_prompt, user_message

PROVIDERS = {
    "none": "Не использовать (только локальные шаблоны)",
    "anthropic": "Anthropic (Claude)",
    "openai_compatible": "OpenAI-совместимый сервер",
}
DEFAULTS = {
    "anthropic": {"endpoint": "https://api.anthropic.com", "model": "claude-opus-5-5"},
    "openai_compatible": {"endpoint": "http://localhost:8000/v1", "model": ""},
}
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
MAX_TOKENS = 16000


class LlmError(Exception):
    """Ошибка внешней модели с сообщением для пользователя."""


@dataclass
class LlmConfig:
    provider: str = "none"
    endpoint: str = ""
    model: str = ""
    timeout: float = 120.0
    include_names: bool = False
    effort: str = "medium"


@dataclass
class CheckReport:
    ok: bool
    messages: list[str] = field(default_factory=list)
    structured: str = "unknown"
    seconds: float = 0.0


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    return ctx


def extract_json(text: str) -> dict:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise LlmError("Ответ модели не содержит JSON-плана.")
        t = t[i:j + 1]
    try:
        d = json.loads(t)
    except json.JSONDecodeError as e:
        raise LlmError(f"Ответ модели не является корректным JSON ({e.msg}).")
    if not isinstance(d, dict):
        raise LlmError("Ответ модели должен быть JSON-объектом.")
    return d


def _output_schema() -> dict:
    s = plan_json_schema(llm_only=True)
    s.pop("$schema", None)
    s.pop("title", None)
    return s


class LlmAdapter:
    def __init__(self, config: LlmConfig, secrets):
        self.config = config
        self.secrets = secrets
        self.mode: str | None = None   # обнаруженный режим: schema | json | text
        self.fallbacks_ok: bool | None = None

    # ---------------------------------------------------------------- availability
    def key(self) -> str | None:
        if self.config.provider == "none":
            return None
        return self.secrets.get(self.secrets.account(self.config.provider, self.config.endpoint))

    def available(self) -> tuple[bool, str]:
        c = self.config
        if c.provider == "none":
            return False, "Внешняя модель не настроена — работает локальный обработчик шаблонов."
        if not c.endpoint or not c.model:
            return False, "Не заданы адрес или модель внешнего сервиса."
        key = self.key()
        if c.provider == "anthropic" and not key:
            return False, "Не задан API-ключ — работает локальный обработчик шаблонов."
        if key and (not key.isascii() or any(ch.isspace() or ord(ch) < 32 for ch in key)):
            return False, ("API-ключ содержит недопустимые символы (пробелы или не латиница) — "
                           "проверьте, что ключ скопирован полностью и без лишних символов.")
        return True, ""

    # ---------------------------------------------------------------- public
    def plan(self, text: str, project, selection: Selection | None = None, cancel=None) -> dict:
        ok, why = self.available()
        if not ok:
            raise LlmError(why)
        summary = project_summary(project, selection, include_names=self.config.include_names)
        sys_p = system_prompt()
        user = user_message(text, summary)
        raw = self._complete(sys_p, [{"role": "user", "content": user}], cancel)
        plan = self._postprocess(raw, text)
        errs = validate_plan_schema(plan)
        if errs:
            if cancel is not None and cancel():
                raise LlmError("Отменено.")
            # одна попытка исправления: возвращаем модели список ошибок проверки
            repair = [{"role": "user", "content": user},
                      {"role": "assistant", "content": raw},
                      {"role": "user", "content": "План не прошёл проверку схемы:\n- " + "\n- ".join(errs[:20])
                       + "\nВерни исправленный JSON-план целиком."}]
            raw = self._complete(sys_p, repair, cancel)
            plan = self._postprocess(raw, text)
            errs = validate_plan_schema(plan)
            if errs:
                raise LlmError("Ответ модели не прошёл проверку и не будет применён:\n- " + "\n- ".join(errs[:10]))
        return plan

    def check(self) -> CheckReport:
        """Проверка подключения и возможностей провайдера (structured output)."""
        t0 = time.perf_counter()
        ok, why = self.available()
        if not ok:
            return CheckReport(False, [why])
        rep = CheckReport(True)
        try:
            if self.config.provider == "anthropic":
                rep.messages += self._anthropic_model_info()
            raw = self._complete("Ответь JSON-объектом плана без операций: kind=\"info\", message=\"ok\".",
                                 [{"role": "user", "content": "Проверка связи. base_revision = 0."}], None)
            plan = self._postprocess(raw, "проверка")
            errs = validate_plan_schema(plan)
            rep.structured = self.mode or "unknown"
            rep.messages.append({"schema": "Structured output по JSON-схеме поддерживается.",
                                 "json": "Поддерживается только JSON-режим без схемы — ответы проверяются приложением.",
                                 "text": "Structured output не поддерживается — JSON извлекается из текста и проверяется приложением."
                                 }.get(self.mode or "", "Режим ответа не определён."))
            if errs:
                rep.messages.append("Пробный ответ не прошёл проверку схемы: " + "; ".join(errs[:3]))
        except LlmError as e:
            rep.ok = False
            rep.messages.append(str(e))
        rep.seconds = time.perf_counter() - t0
        return rep

    # ---------------------------------------------------------------- internals
    def _postprocess(self, raw: str, text: str) -> dict:
        plan = extract_json(raw)
        plan["origin"] = "llm"          # происхождение задаёт приложение, а не модель
        plan.setdefault("schema", "asiko.plan/1")
        plan.setdefault("kind", "edit")
        plan.setdefault("operations", [])
        plan["text"] = text
        return plan

    def _complete(self, system: str, messages: list[dict], cancel) -> str:
        if cancel is not None and cancel():
            raise LlmError("Отменено.")
        if self.config.provider == "anthropic":
            return self._anthropic(system, messages)
        if self.config.provider == "openai_compatible":
            return self._openai(system, messages)
        raise LlmError("Неизвестный провайдер.")

    # ---- Anthropic (официальный SDK)
    def _anthropic_client(self):
        try:
            import anthropic
        except ImportError:
            raise LlmError("Библиотека anthropic не установлена.")
        base = self.config.endpoint.strip().rstrip("/") or None
        return anthropic, anthropic.Anthropic(api_key=self.key(), base_url=base, timeout=self.config.timeout,
                                              max_retries=1)

    def _anthropic_model_info(self) -> list[str]:
        anthropic, client = self._anthropic_client()
        try:
            info = client.models.retrieve(self.config.model)
            out = [f"Модель найдена: {getattr(info, 'display_name', self.config.model)}"]
            mi = getattr(info, "max_input_tokens", None)
            if mi:
                out.append(f"Контекст: {mi} токенов")
            return out
        except anthropic.NotFoundError:
            raise LlmError(f"Модель «{self.config.model}» не найдена у провайдера.")
        except anthropic.AuthenticationError:
            raise LlmError("Ключ API не принят провайдером.")
        except anthropic.APIConnectionError:
            raise LlmError("Нет соединения с сервисом модели (сеть недоступна?). Редактирование проекта продолжает работать.")
        except anthropic.APIStatusError as e:
            return [f"Сведения о модели недоступны (HTTP {e.status_code})."]

    def _anthropic(self, system: str, messages: list[dict]) -> str:
        anthropic, client = self._anthropic_client()
        default_endpoint = (self.config.endpoint.rstrip("/") in ("", "https://api.anthropic.com"))
        modes = [self.mode] if self.mode else ["schema", "text"]
        use_fb = (self.fallbacks_ok is not False and default_endpoint and self.config.model in FALLBACK_MODELS)
        last_err = None
        for mode in modes:
            for fb in ([True, False] if use_fb else [False]):
                oc: dict = {}
                if self.config.effort:
                    oc["effort"] = self.config.effort
                if mode == "schema":
                    oc["format"] = {"type": "json_schema", "schema": _output_schema()}
                kwargs = dict(model=self.config.model, max_tokens=MAX_TOKENS, system=system, messages=messages)
                if oc:
                    kwargs["output_config"] = oc
                try:
                    if fb:
                        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                           fallbacks="default", **kwargs)
                    else:
                        resp = client.messages.create(**kwargs)
                except anthropic.BadRequestError as e:
                    msg = str(getattr(e, "message", e))
                    last_err = msg
                    if fb:
                        if "fallback" in msg.lower():
                            self.fallbacks_ok = False
                        continue
                    if "effort" in msg and self.config.effort:
                        self.config.effort = ""
                        return self._anthropic(system, messages)
                    break  # пробуем следующий режим (например, без схемы)
                except anthropic.AuthenticationError:
                    raise LlmError("Ключ API не принят провайдером.")
                except anthropic.PermissionDeniedError:
                    raise LlmError("Нет доступа к модели с этим ключом.")
                except anthropic.NotFoundError:
                    raise LlmError(f"Модель «{self.config.model}» или адрес сервиса не найдены.")
                except anthropic.RateLimitError:
                    raise LlmError("Превышен лимит запросов к модели. Повторите позже.")
                except anthropic.APITimeoutError:
                    raise LlmError("Сервис модели не ответил вовремя.")
                except anthropic.APIConnectionError:
                    raise LlmError("Нет соединения с сервисом модели (сеть недоступна?). "
                                   "Редактирование проекта и локальные команды продолжают работать.")
                except anthropic.APIStatusError as e:
                    raise LlmError(f"Ошибка сервиса модели (HTTP {e.status_code}).")
                if fb:
                    self.fallbacks_ok = True
                self.mode = mode
                if resp.stop_reason == "refusal":
                    det = getattr(resp, "stop_details", None)
                    cat = getattr(det, "category", None) if det else None
                    raise LlmError("Модель отклонила запрос" + (f" (категория: {cat})." if cat else "."))
                if resp.stop_reason == "max_tokens":
                    raise LlmError("Ответ модели оборван (лимит длины) и не будет применён.")
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                if not text.strip():
                    raise LlmError("Модель вернула пустой ответ.")
                return text
        raise LlmError("Провайдер отклонил запрос: " + (last_err or "неизвестная ошибка"))

    # ---- OpenAI-совместимый HTTP
    def _openai(self, system: str, messages: list[dict]) -> str:
        url = self.config.endpoint.strip().rstrip("/") + "/chat/completions"
        modes = [self.mode] if self.mode else ["schema", "json", "text"]
        last = None
        for mode in modes:
            body: dict = {"model": self.config.model, "temperature": 0,
                          "messages": [{"role": "system", "content": system}] + messages}
            if mode == "schema":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "asiko_plan", "schema": _output_schema()}}
            elif mode == "json":
                body["response_format"] = {"type": "json_object"}
            headers = {"Content-Type": "application/json"}
            key = self.key()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout, context=_ssl_context()) as r:
                    data = json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                code = e.code
                if code in (401, 403):
                    raise LlmError("Ключ API не принят сервером.")
                if code == 404:
                    raise LlmError("Адрес сервиса или модель не найдены (HTTP 404).")
                if code == 429:
                    raise LlmError("Превышен лимит запросов. Повторите позже.")
                if code in (400, 422):
                    last = f"HTTP {code}"
                    continue  # режим не поддерживается — пробуем следующий
                raise LlmError(f"Ошибка сервера модели (HTTP {code}).")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise LlmError(f"Нет соединения с сервисом модели ({getattr(e, 'reason', e)}). "
                               f"Редактирование проекта и локальные команды продолжают работать.")
            except json.JSONDecodeError:
                raise LlmError("Сервер вернул не-JSON ответ.")
            try:
                choice = data["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise LlmError("Неожиданный формат ответа сервера.")
            if choice.get("finish_reason") == "length":
                raise LlmError("Ответ модели оборван (лимит длины) и не будет применён.")
            self.mode = mode
            if not content:
                raise LlmError("Модель вернула пустой ответ.")
            return content
        raise LlmError(f"Сервер отклонил все режимы запроса ({last}).")

"""Хранение API-ключа в системном хранилище секретов (keyring).

Windows — Credential Locker, macOS — Keychain, Linux — Secret Service.
Если безопасного хранилища нет, ключ держится только в памяти до закрытия программы.
Ключ никогда не записывается в проект, настройки или журналы.
"""
from __future__ import annotations

SERVICE = "ASIKO"


class SecretStore:
    def __init__(self):
        import os

        self._memory: dict[str, str] = {}
        self.backend_name = "память (до закрытия программы)"
        self.secure = False
        self._kr = None
        if os.environ.get("ASIKO_NO_KEYRING"):   # тесты/CI: не трогать системное хранилище
            return
        try:
            import keyring
            from keyring.backends import fail

            kr = keyring.get_keyring()
            bad = isinstance(kr, fail.Keyring) or "fail" in type(kr).__module__ or "null" in type(kr).__module__.lower()
            if not bad:
                try:
                    from keyring.backends import chainer

                    if isinstance(kr, chainer.ChainerBackend) and not kr.backends:
                        bad = True
                except ImportError:
                    pass
            if not bad:
                self.secure = True
                self.backend_name = getattr(kr, "name", type(kr).__name__)
            self._kr = keyring if self.secure else None
        except Exception:
            self._kr = None

    @staticmethod
    def account(provider: str, endpoint: str) -> str:
        return f"{provider}|{endpoint.strip().rstrip('/')}"

    def get(self, account: str) -> str | None:
        if self._kr is not None:
            try:
                v = self._kr.get_password(SERVICE, account)
                if v:
                    return v
            except Exception:
                pass
        return self._memory.get(account)

    def set(self, account: str, secret: str) -> bool:
        """True — сохранено в системном хранилище, False — только в памяти."""
        if self._kr is not None:
            try:
                self._kr.set_password(SERVICE, account, secret)
                self._memory.pop(account, None)
                return True
            except Exception:
                pass
        self._memory[account] = secret
        return False

    def delete(self, account: str) -> None:
        self._memory.pop(account, None)
        if self._kr is not None:
            try:
                self._kr.delete_password(SERVICE, account)
            except Exception:
                pass

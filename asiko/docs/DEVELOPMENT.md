# Разработка, сборка и распространение ASIKO

## Окружение

* Python 3.13 (поддерживается 3.12–3.13), зависимости зафиксированы:
  `requirements.lock` (выполнение) и `requirements-dev.lock` (тесты и сборка),
  сгенерированы `uv pip compile pyproject.toml --universal` — с маркерами для Windows/macOS/Linux.
* Linux: системные пакеты `libportaudio2` (звук), `libegl1 libgl1 libxkbcommon0 libfontconfig1
  libdbus-1-3 libxcb-cursor0` (Qt). Тестовый ролик для проверки видео уже лежит в
  `tests/fixtures/` (ffmpeg нужен, только чтобы пересоздать его).

```bash
cd asiko
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.lock && pip install -e . --no-deps
python -m asiko                       # приложение
python -m pytest                      # тесты (offscreen Qt, без сети, без ключа)
python -m asiko --self-test [папка]   # самопроверка на реальных файлах (JSON-отчёт)
```

Переменные: `ASIKO_HOME` — корень данных/кэша/настроек (изоляция тестов, переносная
установка); `ASIKO_NO_KEYRING=1` — не обращаться к системному хранилищу секретов (тесты/CI);
`QT_QPA_PLATFORM=offscreen` — без дисплея.

## Структура

```
src/asiko/
  model/       проект (dataclass), пресеты, автоматизация по сэмплам, миграции формата
  commands/    схема asiko.plan/1, обработчики операций, транзакции и история, защита, варианты A→B
  audio/       импорт и ресемплинг, анализ, DSP, рендер, экспорт, фиксация PCM, предпрослушивание, воспроизведение
  nlp/         локальный обработчик шаблонов, описание проекта для модели, адаптер LLM, хранилище ключей
  storage/     сохранение/открытие, автосохранение, поиск исходников, переносимый проект
  ui/          окно, таймлайн, панели, инспектор, диалоги, видео, фоновые задачи
  demo.py      демонстрационные проекты; selftest.py — самопроверка; testsignals.py — синтез сигналов
tests/         120 автотестов, включая сквозные сценарии A/B/C
packaging/     asiko.spec (PyInstaller), build.py, значки
```

## Сборка дистрибутива

```bash
python packaging/build.py
```

Шаги: PyInstaller (`packaging/asiko.spec`, оконное приложение, onedir) → лицензии сторонних
компонентов в `licenses/` → демонстрационные проекты в `Демо-проекты/` → **самопроверка
собранного исполняемого файла** (`--self-test`, код возврата и `dist/selftest-report.json`) →
архив: `.zip` (Windows), `.dmg` (macOS, через `hdiutil`), `.tar.gz` (Linux).

Особенности:

* numba (через librosa) кэширует JIT-код в `NUMBA_CACHE_DIR` пользователя — каталог
  приложения только для чтения. Первый анализ долей после установки дольше (~10 с).
* На Linux в сборку вкладывается системная `libportaudio.so.2`; Windows/macOS-колёса
  sounddevice содержат PortAudio сами.
* Неиспользуемые модули Qt (WebEngine, 3D, Charts…) и scikit-learn исключены.

## CI

`.github/workflows/asiko.yml` — матрица Windows x86_64, macOS arm64 (Apple Silicon), Linux x86_64:
установка зафиксированных зависимостей → тесты → сборка → самопроверка собранного приложения
→ артефакты `ASIKO-<ос>` (архив + отчёт самопроверки) на 30 дней. Запускается при изменениях
в `asiko/` или вручную (workflow_dispatch).

## Подпись и notarization (не выполнены — нет сертификатов)

Текущие сборки **не подписаны**. Приложение не требует отключать защиту ОС: на Windows
SmartScreen предлагает «Выполнить в любом случае», на macOS — «Всё равно открыть» в
«Конфиденциальность и безопасность». Чтобы распространять без предупреждений:

**Windows (Authenticode).** Нужен сертификат подписи кода (OV/EV). После сборки:

```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f cert.pfx /p $env:PFX_PASSWORD `
  dist\ASIKO\ASIKO.exe
```

Затем упаковать в zip (или собрать установщик Inno Setup/MSIX и подписать его).

**macOS (Developer ID + notarization).** Нужны Apple Developer ID Application и учётные данные
App Store Connect:

```bash
codesign --deep --force --options runtime --timestamp \
  --entitlements packaging/macos-entitlements.plist \
  --sign "Developer ID Application: <Имя> (<TEAMID>)" dist/ASIKO-*/ASIKO.app
hdiutil create -volname ASIKO -srcfolder dist/ASIKO-<версия>-macos-arm64 -format UDZO ASIKO.dmg
codesign --sign "Developer ID Application: <Имя> (<TEAMID>)" ASIKO.dmg
xcrun notarytool submit ASIKO.dmg --apple-id "$APPLE_ID" --team-id "$TEAM_ID" --password "$APP_PASSWORD" --wait
xcrun stapler staple ASIKO.dmg
```

numba/llvmlite генерируют машинный код во время работы, поэтому для hardened runtime в
`packaging/macos-entitlements.plist` включены `com.apple.security.cs.allow-jit` и
`com.apple.security.cs.allow-unsigned-executable-memory`. В CI секреты подписи добавляются
как GitHub Secrets; шаги подписи стоит выполнять только при их наличии.

# Исполнение извлечённого S47

Это точный пакет кандидата `sc71-s47-s31-effort` (предложение 50), а не самостоятельный сервер ответов. Все семь файлов побайтно сверены с ZIP-снимком кандидата в каноническом `source/search.sqlite3`: совпали `prepare.py`, `retrieve.py`, `verify_quote.py`, `check_answer.py`, `prompt.md`, `tools.json`, `solution.json`. Исходники не исправлялись и не адаптировались при извлечении.

## Что нужно для запуска

Четыре Python-скрипта используют только стандартную библиотеку Python (`json`, `re`, `sqlite3`, `sys`, `pathlib`). Для индекса необходим SQLite с FTS5 и функцией BM25. Модель им не нужна. Синтетическая проверка этих скриптов прошла на Python 3.11.5 и SQLite 3.45.1; это проверка инструментов, а не полного харнеса.

Для получения окончательных ответов нужен адаптер Arena `cases/sc71/harness/hermes/`, Hermes Agent версии `86fa9e6ca1de7ee71132d2be0103f3e37d1fa872`, его зависимости из `uv.lock` и Python 3.14. Зафиксированная модель — `deepseek-ai/DeepSeek-V4-Flash-0731`, `reasoning_effort=max`, API Chat Completions через внутренний gateway. В кандидате выбран режим `retrieval`, инструменты Hermes `terminal` и `file`, предел `max_iterations=10000`; отдельно регистрируются `arena_search`, `verify_quote`, `check_answer`. Модель, KB, библиотеки Hermes и инфраструктурный gateway в семь файлов не входят.

## Каталоги и параметры

В штатном контейнере пакет смонтирован в `/candidate`, исходные материалы — в `/assets`, временный рабочий каталог — `/tmp/arena-workspace`. Нормативные документы имеют пути `/assets/kb/norm/<64hex>.md`; `/assets/kb/index.tsv` содержит строки `origin_id<TAB>размер<TAB>путь исходника`. Образ также содержит ссылку `/kb -> /assets/kb`, поэтому цитата `/kb/norm/<64hex>.md` разрешается в тот же файл. Для воспроизведения полного результата нужны именно исходные материалы кампании, включая `index.tsv`: этот индекс влияет на определение типа копии. Без него скрипт технически работает, но сведения о происхождении копий будут неполными.

Кандидат не читает переменные `ARENA_*` или `KBROOT`. `prepare.py` получает `assets_dir` и `workspace_dir` из JSON stdin; `retrieve.py` — `workspace_dir`. Оба проверяющих скрипта жёстко читают `/assets/kb/norm`: передача другого `assets_dir` не перенастраивает их. Для их обычного CLI нужен соответствующий mount; переменная окружения не заменит его. `ARENA_MODEL_ACCESS_FILE` относится к исследовательскому launcher, а не к этим скриптам или штатному stdin решателя.

## Контракты скриптов

Каждый вызов читает один JSON-объект из stdin и пишет один JSON-объект в stdout.

| Скрипт | Минимальный stdin | Результат |
| --- | --- | --- |
| `prepare.py` | `{"assets_dir":"/assets","workspace_dir":"/tmp/arena-workspace"}` | `{"ok":true,"context":"…","segments":123,"docheads":45,"documents":45}`; создаёт `idx.sqlite3` в workspace |
| `retrieve.py` | `{"workspace_dir":"/tmp/arena-workspace","arguments":{"query":"срок подачи заявления"}}` | `{"results":[{"text":"…","source":"/kb/norm/<64hex>.md","locator":"…","kind":"…"}]}` |
| `verify_quote.py` | `{"arguments":{"origin_id":"<64hex>","quote":"Текст цитаты"}}` | `{"found":true,"origin_id":"…","normalized_chars":12}` либо `found:false`, иногда с `error` |
| `check_answer.py` | `{"arguments":{"answer":"Полный черновик ответа"}}` | `{"ok":true,"total_quotes":1,"checked_origins":["…"],"issues":[]}`; для отказа дополнительно `refusal:true` |

Для каждого запуска подготовки нужен свежий workspace: существующий `idx.sqlite3` повторно не перестраивается, поскольку таблицы создаются без `IF NOT EXISTS`. Поиск возвращает пустой `results`, если индекс отсутствует; поэтому сначала нужно проверить успешный результат подготовки.

Нормализация проверки цитаты схлопывает пробелы и применяет `casefold`. `check_answer` проверяет существование распознанных ссылок и цитаты длиной от 20 нормализованных символов среди процитированных файлов; он не доказывает полноту, актуальность и смысловую корректность ответа. Значения `found:false` и `ok:false` являются результатами инструментов: требование исправления реализовано промптом, а не автоматическим запретом харнеса на выдачу ответа.

## Синтетическая проверка CLI без модели

Ниже команда для терминала внутри изолированного контейнера с Python и SQLite FTS5, где извлечённые семь файлов смонтированы read-only в `/candidate`, а `/assets` и `/tmp` пусты и доступны на запись. Сеть контейнеру не нужна. Команда создаёт один вымышленный документ; запускать её на реальном `/assets` кампании не требуется.

```bash
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

origin = "a" * 64
quote = "Заявление подается не позднее пяти рабочих дней."
norm = Path("/assets/kb/norm")
norm.mkdir(parents=True, exist_ok=True)
(norm / f"{origin}.md").write_text("ПОЛОЖЕНИЕ О ЗАЯВЛЕНИЯХ\n\n1. " + quote, encoding="utf-8")
Path("/assets/kb/index.tsv").write_text(f"{origin}\t200\tsynthetic.docx\n", encoding="utf-8")
base = {"assets_dir": "/assets", "workspace_dir": "/tmp/s47-synthetic"}

def run(name, arguments=None):
    payload = base | ({"arguments": arguments} if arguments is not None else {})
    result = subprocess.run([sys.executable, f"/candidate/{name}"], input=json.dumps(payload), capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

assert run("prepare.py")["documents"] == 1
assert run("retrieve.py", {"query": "заявление рабочих дней"})["results"]
assert run("verify_quote.py", {"origin_id": origin, "quote": quote})["found"]
assert not run("verify_quote.py", {"origin_id": origin, "quote": "Выдуманное утверждение без опоры на документ."})["found"]
answer = f"«{quote}» [/kb/norm/{origin}.md]"
assert run("check_answer.py", {"answer": answer})["ok"]
assert not run("check_answer.py", {"answer": answer.replace(origin, "b" * 64)})["ok"]
print("prepare, retrieve, verify_quote, check_answer: OK")
PY
```

Эквивалентная локальная проверка уже выполнена на временной синтетической KB: подготовка и поиск успешны, настоящая цитата и ответ приняты, выдуманная цитата и ссылка на отсутствующий документ отклонены. При этой локальной проверке только переменная `KB` проверяющих модулей переназначалась в памяти на временный каталог; извлечённые файлы оставались неизменными. Контейнерный запуск из примера отдельно не выполнялся.

## Полный решатель

Адаптер получает stdin с `input`, параметрами из `solution.json`, фиксированной `model`, `model_base_url`, `model_request_timeout` и `prepare_timeout`. Для профиля финального запуска ожидание gateway — 131460 секунд, подготовка — 172800 секунд. Временный gateway URL передаёт доверенная инфраструктура; его не следует сохранять в пакете кандидата. После подготовки Hermes получает исходный `prompt.md` и контекст индекса, вызывает инструменты и возвращает один JSON `{"answer":"…"}`. Для режима `retrieval` адаптер требует хотя бы один успешный вызов `arena_search`. Полное воспроизведение требует установленного харнеса, исходной KB и доступной закреплённой модели; синтетическая проверка инструментов этих зависимостей не заменяет.

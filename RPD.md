# RPD — карта проекта Arena

Читай перед любой задачей. Карта помогает выбрать исходники и правила для работы; подробные контракты принадлежат документам по ссылкам. При изменении структуры или точек входа обновляй карту по фактическим файлам.

## Первое чтение

1. [README.md](README.md) — назначение арены, жизненный цикл кампании и границы данных.
2. [AGENTS.md](AGENTS.md) и [common/IMMUNE.md](common/IMMUNE.md) — правила работы и согласованность изменений.
3. Выбери маршрут ниже; для задачи конкретного кейса прочитай его README, для кампании — явно переданные условия и состояние.

Arena — универсальный репозиторий для поиска и сравнения решений. Оркестрация задаётся промптами и протоколом; отдельного агентного движка и планировщика здесь нет. Решение и метрика могут обходиться без LLM. Условия кейса и кампании определяют, что разрешено менять, включая харнес и его параметры.

## Маршруты по задачам

| Задача | Что открыть |
| --- | --- |
| Создать кейс | [cases/template/README.md](cases/template/README.md), его [метрику](cases/template/metric/README.md), [описание данных](cases/template/dataset/README.md) и [уточнения промптов](cases/template/prompts/README.md) |
| Подготовить кампанию | [prompts/prepare.md](prompts/prepare.md), [common/preparation.md](common/preparation.md), [common/public-evaluation.md](common/public-evaluation.md), [common/runner-contract.md](common/runner-contract.md) |
| Запустить или продолжить поиск | [prompts/start.md](prompts/start.md), [prompts/orchestrator.md](prompts/orchestrator.md), [common/research-protocol.md](common/research-protocol.md), [common/orchestration.md](common/orchestration.md), [common/search-history.md](common/search-history.md), [common/long-runs.md](common/long-runs.md) |
| Делегировать исследование или проверить кандидата | [prompts/researcher.md](prompts/researcher.md) или [prompts/reviewer.md](prompts/reviewer.md), [common/orchestration.md](common/orchestration.md); независимые направления — в отдельных worktree |
| Отобрать и отправить финалистов | [prompts/finalists.md](prompts/finalists.md), [common/public-evaluation.md](common/public-evaluation.md), [common/isolation.md](common/isolation.md) |
| Изменить оценку или ранжирование | [common/public-evaluation.md](common/public-evaluation.md), [common/preparation.md](common/preparation.md), пакет метрики кейса и группа оценки в карте кода ниже |
| Подключить харнес, подготовку или материалы | [common/runner-contract.md](common/runner-contract.md), [runner/README.md](runner/README.md), [deploy/README.md](deploy/README.md) |
| Изменить сервисы или развёртывание | [deploy/README.md](deploy/README.md), [common/isolation.md](common/isolation.md), затем инструкции `deploy/` конкретного кейса |
| Изменить историю, ресурсы или трекинг | [common/search-history.md](common/search-history.md), [common/mlflow.md](common/mlflow.md), [common/isolation.md](common/isolation.md) |

Подготовка и поиск — отдельные запуски по команде пользователя. Поиск проверяет готовность; при пробелах сообщает о них и завершает работу, не запускает подготовку автоматически. [prompts/prepare-reference.md](prompts/prepare-reference.md) нужен только для отдельной подготовки синтетического эталона подходящего LLM-профиля.

## Карта кода

| Область | Файлы и ответственность |
| --- | --- |
| Форматы и валидация | [evaluator/contracts.py](evaluator/contracts.py) — метрика, пример, кандидат, отправка; [response_schema.py](evaluator/response_schema.py) — структурированный ответ; [numeric.py](evaluator/numeric.py) — допустимые числовые оценки |
| Пакет метрики | [metric_package.py](evaluator/metric_package.py) — состав, исполняемость файлов, переносимость и загрузка; [prepare.py](evaluator/prepare.py) — CLI `freeze`/`verify` |
| Вычисление оценки | [metrics.py](evaluator/metrics.py) — профили `command`/`llm`, оценка ответа и агрегация; [judge.py](evaluator/judge.py) — модельный клиент, голоса встроенного судьи и запись его попыток для train-трейсов; [ranking.py](evaluator/ranking.py) — выбор победителей с учётом направления и ресурсов |
| Закрытое оценивание | [service.py](evaluator/service.py) — public/private API, допуск, неизменяемые условия и состояние серии; [__main__.py](evaluator/__main__.py) — запуск через `python -m evaluator` |
| Открытые замеры и отправка | [train.py](evaluator/train.py) — train API, поднаборы, ответы, трейсы и связь с журналом; [client.py](evaluator/client.py) — упаковка кандидатов, `search-readiness`, `submit`, `status` и ожидание public/private через `wait` |
| История поиска | [search.py](evaluator/search.py) — SQLite-журнал, лимит предложений, родословная, снимки, проверки, остановка, кривая и AUC; процессы не запускает |
| Исполнение кандидата | [execution.py](evaluator/execution.py) — сборка runner из операторских настроек и метаданные выбранного решателя; [evaluator/runner.py](evaluator/runner.py) — реестр харнесов, контейнер, mounts и измерение; [runner/main.py](runner/main.py) — комплектный адаптер внутри контейнера |
| Модель и надёжность | [gateway.py](evaluator/gateway.py) — фиксированная модель, scoped-доступ, tools, схемы, повторы и учёт; [reliability.py](evaluator/reliability.py) — общие правила повторов и таймаутов; [capacity.py](evaluator/capacity.py) — слоты исполнения, отдельные от лимита предложенных решений |
| Материалы и подготовка | [assets.py](evaluator/assets.py) — неизменяемые публикации; [build.py](evaluator/build.py) — измеряемая подготовка и зависимости; [build_receipts.py](evaluator/build_receipts.py) — доверенные квитанции и их проверка |
| Снимки и файловые границы | [candidates.py](evaluator/candidates.py), [filesystem.py](evaluator/filesystem.py) — безопасные пути, digest, executable-права, восстановление и обход без симлинков |
| Ресурсы и наблюдаемость | [usage.py](evaluator/usage.py) — доверенные счётчики; [trace_store.py](evaluator/trace_store.py) — модельные вызовы; [tracking.py](evaluator/tracking.py) — очистка и необязательный экспорт в MLflow |

Типичный путь решения: `train.py` или `service.py` → `execution.py` → `evaluator/runner.py` → адаптер харнеса → ответ → `metrics.py`. Модельный решатель обращается через gateway; оценщик имеет отдельный учёт расходов. Комплектный адаптер поддерживает `solve.py` либо `prompt.md` с необязательным `prepare.py`; сторонние харнесы соблюдают общий протокол.

Build и выбор опубликованных материалов — отдельные возможности, явно включаемые оператором. `assets.py` фиксирует содержимое; измерение создания материалов и квитанции принадлежат build. Обычная задача не требует этих возможностей, базы знаний или MLflow.

## Кейсы, кампании и инфраструктура

| Путь | Что здесь искать |
| --- | --- |
| [cases/template/](cases/template/README.md) | Основа нового кейса: задача, данные, метрика, примеры решений и модельных метрик |
| [cases/sc71/](cases/sc71/README.md) | Пакет SC-71; его доменные правила и настройки не переносятся в общее ядро |
| [cases/sc71/campaigns/sc71-second-20261024/](cases/sc71/campaigns/sc71-second-20261024/README.md) | Подготовленный второй запуск SC-71, отдельное развёртывание и свидетельства готовности; выбор кампании задаёт пользователь |
| [cases/sc71/harness/hermes/](cases/sc71/harness/hermes/README.md) | Адаптер Hermes, runtime, профиль и образ харнеса |
| [cases/sc71/metric/](cases/sc71/metric/README.md), [tools/assets.md](cases/sc71/tools/assets.md) | Метрика SC-71 и инструменты подготовки материалов |
| [cases/sc71/acceptance/](cases/sc71/acceptance/README.md), [deploy/gpu6/](cases/sc71/deploy/gpu6/README.md) | Приёмка кейса и операторская инфраструктура целевого узла |
| [cases/sc71/proposals/](cases/sc71/proposals/README.md) | Исходные идеи; `original/` — архив, не журнал зарегистрированных решений |
| `<campaign_dir>/` | Явно выбранная рабочая память кампании; расположение каталога само по себе не задаёт условия и не открывает новую private-серию |
| `cases/<case>/ways/`, `cases/<case>/results/` | Решения и результаты проверок; состав конкретного пакета уточняет README кейса |
| [archive/](archive/README.md) | Исторические отчёты разработки, свидетельства кейсов и локальные материалы; не входят в стартовый контекст |
| [Dockerfile](Dockerfile), [compose.yaml](compose.yaml), [deploy/README.md](deploy/README.md) | Общие образы и доверенный запуск; [runner/Dockerfile](runner/Dockerfile) — отдельный образ комплектного runner |
| [integrations/docker/](integrations/docker/README.md) | Необязательный монитор исследовательского контейнера; поиск не запускает и не возобновляет |

В `<campaign_dir>`: `campaign.md` задаёт условия, `preparation.md` хранит свидетельства готовности, `board/claims/` и `board/findings/` — гипотезы и результаты, `board/reviews/` — допуск точных снимков и материалов, журнал — все версии и измерения, `selection.md` — основания отбора. Все worktree используют один абсолютный путь к состоянию и журналу. Не выбирай кампанию автоматически по имени или дате каталога.

## Границы, важные до изменений

- Зафиксированные модель, метрика, эталоны, split и условия кампании неизменяемы. `freeze`/`verify` проверяют формат, состав и исполняемость файлов пакета, но не смысл метрики и не готовность сервера.
- Любой раздел может быть пустым; его оценивание пропускается. Train открыт; исследователь получает от public только общий score, от private — только статус. Закрытые данные и результаты читает оператор внутри закрытого контура; документационный обзор не даёт допуска к ним. Секреты и закрытые данные не попадают в Git, образы и общий MLflow.
- Все модельные обращения с данными идут к внутренним моделям. Если решатель и судья используют LLM, модель одна. На пример сохраняется один корректный ответ; технический повтор не позволяет выбирать лучший ответ.
- Лимит предложенных решений, число финалистов K и число итоговых решений N — разные настройки. Неудачная версия расходует предложение. Агент не открывает новый private-допуск переименованием. Пользователь может назначить отдельную серию на тех же данных с сохранением прежней истории по [правилам изоляции](common/isolation.md).
- При непустом private поисковой кампании нужен его сервис в режиме `campaign` с каноническим журналом и лимитом предложенных решений; перед стартом проверяют живой `search-readiness`. При пустом private сервис и отправка не нужны, итоговый выбор сохраняется по условиям кампании и доступным свидетельствам. `finalize` закрывает поиск и сохраняет предварительный пакет, `--replace` исправляет выбор из зарегистрированных снимков. Необратимо принимает пакет private-сервис после проверок допуска. Режим `standalone` предназначен для отдельного оценщика вне поиска.
- При достижении лимита предложенных решений новые предложения запрещены, плановые проверки зарегистрированных кандидатов продолжаются. По команде человека остановить поиск запрещены и новые проверки: дожидаются только выполняющихся, отбор использует накопленные свидетельства.
- Ресурсы решателя включают подготовку и повторы; ресурсы оценщика хранятся отдельно. Неизвестный расход не равен нулю. Ранжирование и AUC меняются только вместе с их контрактами и проверками.
- Без задания на эксперимент и зафиксированного числа решений реальные модели не запускаются. Документационная или кодовая задача сама по себе не запускает кампанию.

## Окружение и проверки

Python ≥ 3.12; зависимости и настройки инструментов — [pyproject.toml](pyproject.toml), версии — [uv.lock](uv.lock). Установка: `uv sync --frozen`. MLflow подключается отдельно через extra `tracking`.

Перед передачей изменений:

```bash
uv run ruff check .
uv run pyright
uv run pytest
uv run python -m evaluator.smoke
```

Основные модульные проверки лежат в [tests/](tests/); обычно `evaluator/<module>.py` соответствует `tests/test_<module>.py`. Контейнер и адаптер дополнительно покрыты `test_runner_main.py`, `test_runner_assets.py`, `test_runner_builds.py`, `test_runner_build_entrypoint.py`; совместная работа оценки — `test_integration.py`, командные модельные метрики — `test_metric_model.py`.

Корневой pytest собирает только `tests/`, pyright проверяет `evaluator`, `runner`, `integrations`, `tests`. Для кейса запускай его проверки отдельно по README. Для SC-71: `uv run pytest cases/sc71/tests` и `uv run pyright -p cases/sc71/pyrightconfig.json`.

Общие тесты и smoke используют синтетические входы и подставные ответы без реальных моделей и кампаний. [tests/docker_smoke.py](tests/docker_smoke.py) — отдельная контейнерная проверка с подставным upstream; она не заменяет приёмку нового образа на целевом узле.

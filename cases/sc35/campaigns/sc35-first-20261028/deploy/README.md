# Развёртывание SC-35 на GPU6

Установка `sc35-first-20261028` использует отдельные контейнеры, сети, тома и состояние. Корень — `/cephfs/home/malyshev_sa/arena-sc35-first-20261028`, Compose project — `arena-sc35-first`. Условия находятся в [campaign.md](../campaign.md), фактическая готовность и версии образов — в [preparation.md](../preparation.md). Подготовка не запускает исследователя; после неё нужна отдельная команда пользователя.

| Ресурс | Имя или адрес |
| --- | --- |
| Research network | `sc35-first-researchers`, `172.31.91.0/24`, internal |
| Worker network | `sc35-first-workers`, `172.31.92.0/24`, internal |
| Control network | `sc35-first-control`, `172.31.93.0/24` |
| Internet egress | `sc35-first-egress`, `172.31.94.0/24` |
| Operator tracking network | `sc35-first-operator-tracking`, `172.31.95.0/24`, internal |
| Журнал кампании | Volume `sc35-first-campaign-runtime`, локальный диск GPU6 |
| Train / public / private API | Loopback GPU6 `18930` / `18931` / `18932` |
| Общий MLflow | Loopback GPU6 `5054`, отдельные volumes `sc35-first-mlflow-shared-backend` и `sc35-first-mlflow-shared-artifacts` |
| Операторский MLflow | Существующий сервис на `gpu6:5050` |

Перед созданием сети и привязкой порта оператор проверяет отсутствие конфликта. Сети и volumes объявлены external: Compose их не создаёт. IPv6 отключён, `net.bridge.bridge-nf-call-iptables=1`. Firewall изменяет только `ARENA_SC35_1_FORWARD` и `ARENA_SC35_1_INPUT`. Worker может обращаться только к своему gateway `172.31.92.10:8081`. Исследователь получает свои train/public/private API, общий MLflow, scoped gateway и proxy для общедоступных методических материалов. `network_check.py` проверяет разрешённые маршруты и запрет доступа к хосту, закрытому MLflow, обеим установкам SC71 и прямому интернету.

## Образы и зависимости

Сначала соберите текущее ядро Arena с `ARENA_TRACKING=true`, затем [evaluator.Dockerfile](evaluator.Dockerfile) поверх этого образа с `--build-arg ARENA_BASE_IMAGE=arena-sc35-core:20261028`. Он добавляет закреплённые `lxml==6.1.3` и `jsonschema==4.26.0`, необходимые командной метрике. Hermes и baseline собираются по [инструкции харнеса](../../../harness/README.md). Инфраструктурный образ собирается из [infra.Dockerfile](infra.Dockerfile), который устанавливает firewall именно SC35 и использует общий Squid-конфиг SC71. Установки SC71 и их образы не перезаписываются.

На GPU6 legacy Docker builder не использует соседний `Dockerfile.dockerignore`. Создавайте отдельный чистый build context и кладите соответствующий ignore-файл в его корневой `.dockerignore`; не передавайте Docker каталоги данных, секретов и состояния. Замороженный пакет метрики монтируется при запуске, а реальные примеры не входят в образы.

Для независимого обновления экспортёра соберите образ монитора из корневого `Dockerfile` с `--target evaluator --build-arg ARENA_TRACKING=true` и закрепите его тег в `ARENA_MONITOR_IMAGE` файла `operator/images.env`. Образ должен включать согласованные `evaluator/tracking.py`, `evaluator/tracking_artifacts.py` и `integrations/docker/research_monitor.py`. При отсутствии `ARENA_MONITOR_IMAGE` Compose использует `ARENA_EVALUATOR_IMAGE`; отдельный тег позволяет заменить монитор без пересоздания оценщиков и изменения их текущих заданий.

Реестр в Compose содержит `hermes` и `sc35-content` с отдельными образами. Hermes поддерживает измеряемый build и выбор опубликованных материалов. Baseline использует собственные `/candidate/assets` и не поддерживает эти операции. Везде фиксируются `deepseek-ai/DeepSeek-V4-Flash-0731`, `reasoning_effort=max` и профиль Hermes 30/30/3. Таймаут upstream-попытки — 21600 секунд; текущий `evaluator.reliability.request_timeout` вычисляет 131460 секунд для gateway с пятью повторами и задержками 30/90/240/540/900. Подготовка допускает 172800 секунд, пример — 604800 секунд. Технические таймауты не задают бюджет поиска.

## Операторская конфигурация

`release` содержит проверенный код. `workspace` — отдельная очищенная копия для исследователя без исходного `archive/local`, генераторов закрытого набора, operator-data, ключей и чужих кампаний. В `operator` находятся закрытые данные, состояние и измерения, в `secrets` — ключ модели и три независимых submit-токена. `research-access` содержит только копии train/public/private submit-токенов; scoped `model-access.json` создаётся launcher перед отдельным поиском. До этого `operator/launch/active.json` отсутствует.

Скопируйте [images.env.example](images.env.example) в `operator/images.env`, закрепите фактически собранные теги и добавьте `ARENA_DOCKER_GID` из группы Docker socket и `SC35_INPUT_ASSETS_DIGEST` из публикации разрешённых исходных материалов. Общие evaluator-настройки заданы непосредственно в Compose; дополнительный `operator/evaluator.env` не нужен. `operator/gateway.env` содержит только операторский `MODEL_BASE_URL` внутреннего upstream. Секрет ключа читается из `secrets/model-api-key`. Compose фиксирует модель, параметры, каталоги учёта, отсутствие модельного судьи и стандартную обработку JSON Schema; настройки судьи SC71 не переносятся.

Транспортный предел клиента исследователя и серверного запроса совпадает: 256 MiB на пакет. Он не ограничивает число файлов или предложений и не заменяет лимит решений кампании.

Подготовьте каталоги `operator/state/{train,public,private,capacity,research-monitor}`, `operator/{builds,usage,traces,outer-usage,outer-traces,launch,search}`, `staging`, `derived`, `workspace`, `research-access` с необходимым владельцем UID 10001. Закрытые каталоги имеют права `0700`. State, usage, assets, derived и builds используются по одинаковым абсолютным путям в evaluator и на Docker-хосте. `operator/state/capacity` должен быть доступен на запись всем оценщикам; общий пул ограничен четырьмя одновременно исполняемыми заданиями. Исследователь не получает Docker socket или операторские mounts.

До запуска монитора создайте `${ARENA_ROOT}/operator/state/research-monitor/tmp` с владельцем `10001:10001`, правами `0700` и доступом на запись из его контейнера. Compose задаёт этот путь как `TMPDIR` внутри уже подключённого state mount. Каталог должен находиться на диске, с местом для полного сериализованного JSON и временных частей: экспортёр сначала пишет JSON во временный файл. Контейнерный `/tmp` — tmpfs размером 2 GiB, поэтому для такого экспорта он не подходит. Явно заданный недоступный `TMPDIR` считается ошибкой; экспортёр не переключается молча на `/tmp`.

В `assets/sc35` размещаются только разрешённые кодовые утилиты, профиль, таксономия и схема ответа. Создайте пустые `assets/derived` и `assets/prepared` для вложенных mounts. Исходные документы и эталоны находятся отдельно. Один и тот же замороженный пакет устанавливается в `operator/metric`, JSONL частей — в `operator/dataset/{train,public,private}/dataset.jsonl`; их состав сохраняется 113/128/640. Train открыт через API и разрешённые отдельные файлы, public/private не попадают в workspace.

В volume `sc35-first-campaign-runtime` разместите файлы этой кампании и пустой канонический журнал: `python -m evaluator.search --journal /campaign/search.sqlite3 init --max-solutions 100`. Исследователь монтирует его по `/workspace/cases/sc35/campaigns/sc35-first-20261028`, private — readonly по `/search`. Журнал не хранится на CephFS. Private использует отдельный `operator/search` для операторского свидетельства остановки; копирование состояния другой кампании не допускается.

Скопируйте [research-monitor.json](research-monitor.json) в `operator/research-monitor.json`. Очереди трекинга явно заданы в Compose и совпадают с монитором. У оценщиков задано `ARENA_TRACKING_DELIVERY=queued`: `export_configured` сохраняет результат и снимки вызовов в устойчивую очередь без запросов к MLflow. Единственный сетевой публикатор этих очередей — `research-monitor`; его `flush` доставляет их независимо от режима производителя. Общий MLflow принимает train и разрешённый public-score; закрытые payload направляются только оператору. `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` сохраняется. Существующий операторский MLflow подключается к `sc35-first-operator-tracking` с alias `operator-source`, IP `172.31.95.2`; новый proxy использует `172.31.95.3`. Подключение не меняет backend/artifact volumes существующего сервиса и не требует его перезапуска.

Общий `shared-tracking` запускается с `--workers 1` и лимитом памяти 4 GiB. Это дополняет ограничение размера запросов новым экспортёром: большие JSON передаются частями, а большие вызовы представлены в span ссылкой на полный артефакт. Состав и чтение таких артефактов описаны в [общем контракте MLflow](../../../../../common/mlflow.md). При изменении числа workers пересоздайте только `shared-tracking` с сохранением существующих backend/artifact volumes.

Операторский proxy подключает [общий конфиг MLflow](../../../../../deploy/mlflow/README.md) с потоковой передачей запросов и ответов. В release обязательно сохраняйте файл `deploy/mlflow/operator-nginx.conf` относительно корня исходников. Контейнерную регрессию крупных артефактов выполняйте по общей инструкции; при изменении proxy применяйте Compose только к `operator-tracking` с `--no-deps`, без перезапуска остальных сервисов кампании и backend MLflow.

## Обновление доставки во время поиска

Замена файлов и Compose environment не меняет уже импортированный код работающих оценщиков. Для действующей кампании сохраните их процессы и задания, очереди, журнал и контейнер исследователя; обновите только образ `research-monitor` и необходимые сервисы трекинга. Перед публикацией новым монитором примените актуальный [firewall.sh](firewall.sh): запреты оценщик → MLflow стоят перед разрешением `ESTABLISHED,RELATED`, поэтому закрывают и прежние соединения старого экспортёра. Старый оценщик сохраняет недоставленный результат в очередь, а новый монитор доставляет его без повторного решения или оценивания. После штатного будущего запуска обновлённых оценщиков их режим `queued` исключает такие сетевые попытки.

Правила ограничены адресами SC35: источники train `172.31.91.10`, `172.31.93.5`, `172.31.95.5`, public `172.31.91.11`, `172.31.93.7`, `172.31.95.6` и private `172.31.91.12`, `172.31.93.6`, `172.31.95.7`; назначения — `172.31.91.20`, `172.31.93.20`, `172.31.95.2`, `172.31.95.3`, `172.28.0.2`, TCP-порт `5000`. Источники исследователя и монитора в этот запрет не входят. После применения проверьте их доступ к разрешённым MLflow, доступ оценщиков к gateway и API, а также доставку существующих очередей. Подтверждайте завершение по чтению payload и `arena_export_complete=true`, сохраняя прежние ID экспорта и run; сетевой сбой не создаёт новое решение.

## Приёмка

Запустите `mlflow-compose.yaml`, затем `compose.yaml` без профиля `search`. Примените firewall нового инфраструктурного образа, проверьте сети через `network_check.py`, права фактических mounts и `evaluator.prepare verify` внутри evaluator. Синтетическая приёмка использует отдельные данные и состояние, подставной upstream и штатный путь submit → result/status → учёт/журнал/трекинг. Она не регистрирует поисковые решения, не расходует реальный private и не вызывает настоящую модель.

Для изменений экспортёра выполните реальную контейнерную проверку `uv run python tests/mlflow_export_smoke.py` по [общей инструкции MLflow](../../../../../deploy/mlflow/README.md): отдельные синтетические случаи с суммарной записью больше 256 MiB и одиночным вызовом больше 256 MiB, восстановлением полного содержимого и повторной доставкой. Проверка nginx `tests/mlflow_proxy_smoke.py` остаётся отдельной проверкой транспорта и не заменяет этот сценарий. Зафиксируйте фактические результаты до применения новых образов.

`mlflow_probe.py export --directory <operator-probe-dir>` публикует синтетическую train-запись. Затем `mlflow_probe.py read --receipt <receipt.json>` из исследовательской сети читает полный вход и выход span и артефакта вызова. Сохраните experiment/run/trace ID и `payload_verified=true` в отчёте подготовки, а разрешённую квитанцию — в `research-access/mlflow-probe-receipt.json`. Успешный экспорт метрик без чтения payload не подтверждает готовность MLflow.

Проверьте авторизованный `evaluator.client search-readiness` с `--require-campaign --max-solutions 100 --finalists 10 --output-count 3 --release-id sc35-first-20261028`. Должны быть `mode=campaign`, `gate_enabled=true`, `journal_ready=true`, `evaluation_enabled=true`; ответ без токена сохраняется в `preparation.md`. После всей приёмки создайте `operator/READY` и `operator/layout-version` со значением `sc35-first-20261028-layout-v1`. Маркеры подтверждают только описанную приёмку этой установки.

## Отдельный старт

После новой команды пользователя:

```bash
ssh gpu6 'bash /cephfs/home/malyshev_sa/arena-sc35-first-20261028/release/cases/sc35/campaigns/sc35-first-20261028/deploy/launch.sh'
```

Launcher проверяет собственные маркеры, состояние сервисов, доступ к слотам исполнения и gateway, живой private readiness и замороженную метрику. Перед созданием scoped-доступа и `active.json` он синхронно повторяет чтение полного MLflow payload в одноразовом Compose-контейнере исследователя: `run --rm --no-deps` с техническим entrypoint `/opt/arena/.venv/bin/python`, скриптом `mlflow_probe.py` и квитанцией `/run/arena/mlflow-probe-receipt.json`. Эта проверка не запускает `researcher.py` или модель; ошибка прекращает launcher до создания внешнего модельного run. После успешной проверки создаются доступ и постоянный контейнер исследователя. Повторный запуск при существующем `operator/launch/active.json` прекращается до Docker-команд. Сам исследователь повторяет проверку допуска перед первым модельным ходом. Для возобновления требуется разобрать сохранённое состояние и снова прочитать этот MLflow payload через исследовательский доступ; запуск этой команды не входит в подготовку.

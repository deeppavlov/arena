# Развёртывание второй кампании SC-71

Это отдельная установка для `sc71-second-20261024`, разрешённая пользователем 24 октября 2026 года, с прежними данными и новой серией private. Старые исходники, контейнеры, журналы, результаты и private-состояние первой кампании сохраняются. Подготовка не запускает исследователя; для поиска нужна отдельная команда пользователя. Фактическая готовность и версии установленных образов находятся в [preparation.md](../preparation.md).

Операторский корень — `/cephfs/home/malyshev_sa/arena-sc71-second-20261024`, Compose project — `arena-sc71-second`, release ID — `sc71-second-20261024`. `release` содержит проверенные исходники, `workspace` — очищенную копию для исследователя, `operator` — конфигурацию, закрытые данные и измерения, `secrets` — ключи, `research-access` — только исследовательские submit-доступы. Содержимое закрытых каталогов не попадает в Git, образы или рабочую копию исследователя. Разрешённая исходная KB остаётся неизменной; производные поисковые представления при подготовке не строятся.

| Ресурс | Имя или адрес |
| --- | --- |
| Research network | `sc71-second-researchers`, `172.31.81.0/24`, internal |
| Worker network | `sc71-second-workers`, `172.31.82.0/24`, internal |
| Control network | `sc71-second-control`, `172.31.83.0/24` |
| Internet egress | `sc71-second-egress`, `172.31.84.0/24` |
| Operator tracking network | `sc71-second-operator-tracking`, `172.31.85.0/24`, internal |
| Журнал кампании | Volume `sc71-second-campaign-runtime`, локальный диск GPU6 |
| Train / public / private API | Loopback GPU6 `18830` / `18831` / `18832` |
| Исследовательский MLflow | Loopback GPU6 `5053`, отдельные backend/artifact volumes |
| Операторский MLflow | Существующий `gpu6:5050`, без перезапуска |

Эти подсети не пересекаются с сетями первой кампании. Firewall использует только новые цепочки `ARENA_SC71_2_FORWARD` и `ARENA_SC71_2_INPUT`; прежние цепочки не очищаются. IPv6 отключён, `net.bridge.bridge-nf-call-iptables=1`. Worker достигает только своего gateway `172.31.82.10:8081`. Исследователь достигает API и MLflow своей кампании, собственного gateway и proxy; прямой доступ к хосту, предыдущей кампании, закрытому MLflow и интернету блокируется.

## Сборка

Из корня проверенного `release` соберите три образа. На GPU6 используется legacy Docker builder без buildx: он не применяет соседний `Dockerfile.dockerignore`. Поэтому Hermes и инфраструктура собираются из отдельных операторских копий `release`, в которых нужный ignore-файл установлен как корневой `.dockerignore`. Исходный `release` остаётся неизменным. Он содержит только подготовленные исходники, без датасетов, ключей и runtime-состояния. Hermes использует закреплённый upstream `86fa9e6ca1de7ee71132d2be0103f3e37d1fa872`, Python 3.14 и свой `uv.lock`; остальные Python-зависимости закреплены `uv.lock` арены. Версия модели — `deepseek-ai/DeepSeek-V4-Flash-0731`, `reasoning_effort=max`, профиль делегирования 30/30/3 для исследователя и решателей.

```bash
docker build --target evaluator --build-arg ARENA_TRACKING=true -t arena-sc71-second-evaluator:20261024 .
mkdir -p "$ARENA_ROOT/build"
ARENA_HERMES_CONTEXT=$(mktemp -d "$ARENA_ROOT/build/hermes-context.XXXXXX")
cp -a . "$ARENA_HERMES_CONTEXT"
cp cases/sc71/harness/hermes/Dockerfile.dockerignore "$ARENA_HERMES_CONTEXT/.dockerignore"
docker build -f "$ARENA_HERMES_CONTEXT/cases/sc71/harness/hermes/Dockerfile" -t arena-sc71-second-hermes:20261024 "$ARENA_HERMES_CONTEXT"
ARENA_INFRA_CONTEXT=$(mktemp -d "$ARENA_ROOT/build/infra-context.XXXXXX")
cp -a . "$ARENA_INFRA_CONTEXT"
cp cases/sc71/campaigns/sc71-second-20261024/deploy/infra.Dockerfile.dockerignore "$ARENA_INFRA_CONTEXT/.dockerignore"
docker build -f "$ARENA_INFRA_CONTEXT/cases/sc71/campaigns/sc71-second-20261024/deploy/infra.Dockerfile" -t arena-sc71-second-infra:20261024 "$ARENA_INFRA_CONTEXT"
```

Последний Dockerfile получает только новый firewall и общий Squid-конфиг. Не собирайте этот образ через старый Dockerfile инфраструктуры: он содержит firewall первой кампании. Каталог Hermes `source-cache` уже присутствует в чистом checkout с README и ignore-файлом; сам `source.tar` необязателен, при его отсутствии Dockerfile загружает закреплённый upstream. Ошибка `COPY source-cache: file not found` при наличии каталога означает неверный build context, а не необходимость создавать или менять кеш. Зафиксируйте идентификаторы готовых образов в `operator/images.env` как `ARENA_EVALUATOR_IMAGE`, `ARENA_HERMES_IMAGE`, `ARENA_INFRA_IMAGE`; туда же внесите `ARENA_ROOT`, фактический `ARENA_DOCKER_GID`, `SC71_INPUT_ASSETS_DIGEST`, `SC71_FINALISTS=10`, `SC71_OUTPUT_COUNT=3`, `SC71_PRIVATE_RELEASE_ID=sc71-second-20261024` и образ MLflow. Секреты в этот файл не включаются.

## Подготовка среды

Создайте перечисленные сети и отдельный volume журнала, а также volumes из [mlflow-compose.yaml](mlflow-compose.yaml). Внешние сети и volumes не создаются основным Compose автоматически. Внутри journal volume разместите файлы новой кампании и инициализируйте `search.sqlite3` с лимитом 100, нулём предложений и без переноса записей первой кампании. Исследователь получает этот volume по `/workspace/cases/sc71/campaigns/sc71-second-20261024`; private читает его readonly по `/search`.

Подготовьте операторские `evaluator.env`, `gateway.env`, `research-monitor.json`, каталог `operator/search` и новые каталоги состояния/учёта. По [общему контракту](../../../../../deploy/README.md) задайте фиксированные модель и параметры, новые абсолютные пути state/usage/assets/builds, сеть `sc71-second-workers`, `ARENA_RUNNER_MODEL_BASE_URL=http://model-gateway:8081/v1`, новый образ Hermes и реестр харнесов с разрешёнными build/asset-selection. Имя `model-gateway` разрешается в доступный адрес control-сети для оценщика и worker-сети для решателя. Технические таймауты сохраняются: одна upstream-попытка 21600 секунд, ожидание gateway 130410 секунд, подготовка 172800 секунд, пример 604800 секунд; первая попытка и максимум пять технических повторов. Gateway явно получает `MODEL_STRICT_SCHEMA_INTEGERS=true` через Compose.

До запуска создайте `$ARENA_ROOT/operator/state/capacity` для `ARENA_SLOT_DIR` с владельцем UID 10001 и правами `0700`; путь должен совпадать с фактической конфигурацией и быть доступен на запись через mount `operator/state` всем оценщикам и build-сервису.

Одинаковый проверенный замороженный пакет метрики устанавливается в `operator/metric` для train/public/private. Существующие неизменённые split устанавливает оператор, не раскрывая агенту закрытые строки. Новая серия использует отдельные state и release ID по явной команде пользователя; прежняя серия не стирается. Train/build, public и private получают отдельные submit-токены. Исследовательский контейнер получает только `research-access`, исходную KB, свои staging/derived и workspace. До запуска создайте пустые каталоги `assets/derived` и `assets/prepared` как точки вложенных монтирований внутри readonly `/assets`; они не содержат готовых представлений KB. Без этих каталогов Docker не сможет создать вложенные mounts при read-only root filesystem.

MLflow запускается отдельным Compose-файлом. Новый proxy `operator-tracking` находится по `172.31.85.3`; существующий операторский MLflow подключается к новой закрытой сети с alias `operator-source` и адресом `172.31.85.2`. Подключение не меняет его backend/artifact volumes и не требует перезапуска. Общий новый сервер получает адреса `172.31.81.20` и `172.31.83.20`. Исследователю доступен только новый общий сервер.

## Приёмка без поиска

Сначала запустите отдельные сервисы MLflow и основные сервисы без профиля `search`. Примените новый firewall, проверьте права mounts, замороженную метрику внутри evaluator и настоящую сетевую изоляцию через `network_check.py`. Скрипт проверяет также запрет доступа к API и MLflow первой кампании. Выполните `python -m cases.sc71.harness.hermes.mock_check` в новом Hermes-образе без сети; это проверка с подставным локальным upstream, без настоящих моделей.

```bash
docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc71/campaigns/sc71-second-20261024/deploy/mlflow-compose.yaml up -d
docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc71/campaigns/sc71-second-20261024/deploy/compose.yaml up -d
python cases/sc71/campaigns/sc71-second-20261024/deploy/network_check.py
```

Для разрешённой пользователем модельной технической приёмки используйте одну конфигурацию и три существующих синтетических сценария из [acceptance](../../../acceptance/README.md), с новым campaign ID, отдельным state, новым образом и путями новой установки. Технические проверки не занимают места в поисковом журнале. Синтетический train-трейс должен читаться целиком из исследовательской сети; одних метрик MLflow недостаточно. Private проверяется только авторизованным `search-readiness`: `mode=campaign`, `gate_enabled=true`, `journal_ready=true`, `evaluation_enabled=true`, лимит 100, K=10, N=3 и release ID новой кампании. Реальный private не отправляется при подготовке.

Только после всех проверок оператор создаёт собственные `operator/READY` и `operator/layout-version` со значением `sc71-second-20261024-layout-v2`. Маркеры первой кампании не подходят. Свидетельства и оставшиеся ограничения сохраняются в `preparation.md`; маркер не заменяет эти проверки.

## Отдельный запуск поиска

После новой команды пользователя оператор вызывает:

```bash
ssh gpu6 'bash /cephfs/home/malyshev_sa/arena-sc71-second-20261024/release/cases/sc71/campaigns/sc71-second-20261024/deploy/launch.sh'
```

Launcher проверяет собственные маркеры, состояние сервисов и метрику, создаёт scoped-доступ исследователю и запускает только новый контейнер. Повторный запуск при существующем `operator/launch/active.json` отклоняется до Docker-команд; возобновление требует разбора сохранённого состояния. Сам исследователь до первого модельного хода повторно проверяет живой private readiness, канонический журнал и условия. Поиск не начинается в ходе подготовки.

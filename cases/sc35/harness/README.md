# SC-35: Hermes и исходный содержательный метод

Исследователь и агентский решатель используют закреплённый Hermes `86fa9e6ca1de7ee71132d2be0103f3e37d1fa872`, модель `deepseek-ai/DeepSeek-V4-Flash-0731` и `reasoning_effort=max`. Реализация [runtime](../../sc71/harness/hermes/runtime.py), профиль и [launcher исследователя](../../sc71/harness/hermes/researcher.py) переиспользуются из SC-71. Сборка SC-35 копирует текущие версии адаптера, evaluator и runner поверх закреплённого базового образа. Условия и фактические версии установки принадлежат выбранной кампании.

## Образы и зависимости

[Dockerfile](Dockerfile) собирается из корня репозитория с обязательным `--build-arg HERMES_IMAGE=<закреплённый образ Hermes>`. Он сохраняет Python 3.14 и upstream Hermes базового образа, обновляет код арены и устанавливает [requirements.txt](requirements.txt) в оба окружения Python. `lxml==6.1.3` и `jsonschema==4.26.0` нужны обработке DOCX и метрике. `content.Dockerfile` получает уже собранный SC-35 Hermes image в том же аргументе и меняет только entrypoint. BuildKit использует отдельные allowlist-файлы `.dockerignore`; при legacy builder оператор создаёт отдельный build context с соответствующим ignore-файлом в корне.

```bash
docker build -f cases/sc35/harness/Dockerfile --build-arg HERMES_IMAGE=arena-sc71-second-hermes:20261024 -t arena-sc35-hermes:20261028 .
docker build -f cases/sc35/harness/content.Dockerfile --build-arg HERMES_IMAGE=arena-sc35-hermes:20261028 -t arena-sc35-content:20261028 .
```

Оператор закрепляет версии итоговых образов до кампании. Наличие образа не доказывает готовность сервисов: синтетическая приёмка выполняется на целевом GPU6. Исследователь запускает `/opt/arena/cases/sc71/harness/hermes/researcher.py` с явными `--prompt-file`, `--journal`, `--private-url`, `--private-token-file`, `--private-finalists`, `--private-output-count` и `--private-release-id` новой кампании. Значение SC-71 по умолчанию для prompt-file в SC-35 не используется.

## Харнес `hermes`

Адаптер [hermes.py](hermes.py) соблюдает [runner contract](../../../common/runner-contract.md). Кандидат содержит `prompt.md` и `solution.json` с `{"harness":"hermes","parameters":{}}`. Он может добавлять `prepare.py`, `tools.json` и скрипты инструментов; контракт этих файлов совпадает с [описанием Hermes](../../sc71/harness/hermes/README.md). `parameters.mode` разрешён только `files`; методы поиска нормативных требований подключаются собственным инструментом в `tools.json`. `toolsets` допускает `terminal`, `file`, `delegation`, `todo`; `max_iterations` — от 1 до 10000. Модель, endpoint, image, mounts и профиль делегирования кандидатом не меняются.

Исходный input содержит `case_id`, `docx_base64` и предусмотренные форматом идентификаторы документа и канонического текста. Адаптер проверяет envelope, сохраняет текущий документ в `/tmp/arena-workspace/input.docx`, исходный envelope — в `input.json`. Hermes получает компактное сообщение с идентичностью и путями, без base64 документа. `prepare.py` и инструменты получают полный исходный input через stdin вместе с временным scoped gateway; реквизиты gateway на диск не записываются. Разрешённые материалы находятся в `/assets`, код кандидата — в `/candidate`. Для точных координат используется канонический `Projection` SC-35 с единицами Unicode code point, включая завершающий перевод строки каждого абзаца.

Финальный ответ — единственный JSON `sc35-span-review-v1` без Markdown-ограждений. Адаптер возвращает его строкой в `{"answer":"..."}`; схему, категории и интервалы проверяет замороженная метрика. Неверный ответ не исправляется повторной модельной генерацией. Ошибки runtime/gateway завершают запуск; технические повторы выполняет только gateway по общему профилю. Токены учитывает gateway, полное время — внешний runner.

Оператор разрешает `supports_build=true` и `supports_asset_selection=true`. Операция build выполняет только `candidate/build.py` через существующий контракт, не получает вход примера и не запускает Hermes. Подготовка метода учитывается доверенным build-сервисом и входит в ресурсы решения; подготовка арены её не выполняет.

## Харнес `sc35-content`

[content.py](content.py) запускает `/candidate/solve.py`; исходный метод находится в [ways/baseline](../ways/baseline). Это программный граф, формальные проверки и типографика с одним содержательным модельным этапом. Он вызывает ту же закреплённую модель через scoped gateway без самостоятельных повторов. Внешний исследователь остаётся Hermes. Регистрация использует `uses_model=true`, `supports_build=false`, `supports_asset_selection=false`. Этот адаптер сохраняет переданный в DEH метод и не регистрирует его как предложенное решение до отдельного старта поиска.

## Синтетические проверки

```bash
uv run pytest cases/sc35/tests/test_harness.py
docker run --rm --entrypoint python arena-sc35-hermes:20261028 -m cases.sc35.acceptance.hermes_check
docker run --rm --entrypoint python arena-sc35-hermes:20261028 -m cases.sc71.harness.hermes.mock_check
```

Проверки SC-35 используют вымышленный DOCX и подставной HTTP model endpoint. Проверяются чтение DOCX инструментом настоящего Hermes, компактный модельный контекст, фиксированный профиль, отсутствие секретов в файлах и отсутствие повторного запроса после ошибки gateway. Общая проверка Hermes дополнительно проверяет делегирование и вложенное делегирование. HTTP-путь оценщиков проверяется отдельно через `acceptance/http_check.py`; реальные модели, датасеты и private-допуск для приёмки не используются.

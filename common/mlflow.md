# MLflow

Каждый законченный публичный эксперимент, включая неудачный, должен иметь run и трейсы в `http://gpu6:5050`. Пишет используемый агент или харнес через обычный MLflow SDK. Отдельный адаптер для Codex, Hermes или чата не нужен. Сначала сохраняй результат и трейсы локально; при недоступном сервере сохрани очередь экспорта и отметь эксперимент как `tracking_pending`. Сбой трекинга не повод повторять решение примеров. Эксперимент считается оформленным после получения run ID.

## Что сохранять

Один run соответствует одному кандидату на конкретном публичном наборе или подмножестве. Параллельные исследователи используют отдельные процессы и run ID.

| Поле | Содержание |
| --- | --- |
| `campaign_id`, `case_id`, `candidate_id` | Кампания, универсальный кейс, кандидат |
| `candidate_digest` | SHA-256 неизменяемого пакета кандидата |
| `model`, `harness`, `harness_version` | Зафиксированный решатель и его конфигурация; судья использует ту же модель |
| `metric_version`, `judge_prompt_digest` | Версия критериальной метрики и промпта судьи |
| `dataset_id`, `dataset_version`, `subset_id` | Набор, версия и идентификатор реально выбранного подмножества |
| `example_ids`, `gold_version` | Список публичных примеров и версия эталона; `null`, если эталон не применяется |
| `metrics` | Оценка кандидата, оценки критериев, число примеров, вызовы, токены, длительности |
| `metrics.solver_tokens`, `metrics.solver_seconds`, `metrics.solver_unreported_attempts` | Обязательные сумма известных токенов решателя, время решения с подготовкой и повторами, число попыток без корректного usage |
| `metrics.judge_tokens`, `metrics.judge_seconds`, `metrics.judge_unreported_attempts` | Такие же обязательные замеры судьи отдельно от решателя |
| `cost` | Известная стоимость и валюта; `null`, пока нет доверенных ставок input/output |
| `status`, `error` | Результат эксперимента и причина сбоя; инфраструктурный сбой не становится нулевой оценкой модели |
| `calls` | Вызовы решателя, судьи и инструментов: вход, выход, роль, ID примера, критерий, номер оценки, время, usage и ошибка |

Сохраняй гипотезу, выбранные параметры, ответы, оценки, решение «оставить/откатить» и все доступные публичные трейсы как артефакты того же run. Один корректно полученный ответ решателя на пример, три независимые корректные оценки каждого критерия; каждый критерий — отдельный запрос. Технические повторы сохраняются отдельными попытками с прежними входом и настройками. Подмножество разрешено, но результаты разных подмножеств не выдаются за один сопоставимый замер. `null` сохраняется в JSON и не заменяется числовым нулём. Перед экспортом исключи ключи API, заголовки авторизации и содержимое секретов.

Ресурсы каждого примера храни отдельно для `solver` и `judge`. Каждая запись содержит `wall_seconds` и вложенный `usage` с `prompt_tokens`, `completion_tokens`, `total_tokens`, `attempts`, `unreported_attempts`. Суммы токенов берутся из usage модельного сервера для всех вызовов, включая подготовительные модельные вызовы харнеса и повторы. При `unreported_attempts > 0` известная сумма является неполной; пометка сохраняется и в итоговых метриках. Время решателя измеряется снаружи харнеса и включает подготовку, инструменты, модель и ожидания повторов. Эти замеры обязательны для [единого правила выбора](public-evaluation.md), а стоимость в валюте может оставаться неизвестной.

## Пример экспорта публичного результата

Если харнес уже пишет в MLflow, используй его трекинг. Иначе сохрани код ниже в `log_public.py` в рабочем каталоге исследователя. Зависимость нужна только его окружению: `uv run --with 'mlflow>=3.6,<4' python log_public.py public_record.json`. Ничего устанавливать в приватный оценщик не требуется. Версию клиента при подготовке запуска согласуй с установленным сервером; сервер сейчас не проверяется.

В `public_record.json` нужны `split: "public"`, `params` с полями таблицы, `metrics` с числами либо `null`, `status: "FINISHED"` или `"FAILED"`, `cost`, `calls` и `start_time_ns`/`end_time_ns` в Unix nanoseconds. Каждый элемент `calls` содержит `name`, `inputs`, `outputs`, `attributes`, `status: "OK"` или `"ERROR"` и те же временные поля. Поля `example_ids`, гипотезу, ошибки и дополнительные результаты храни в этом же JSON. `attributes` вызова связывает его с примером, ролью, критерием и номером оценки. Скрипт экспортирует уже сделанные вызовы, не обращается к моделям и не запускает эксперимент повторно.

```python
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record["split"] != "public":
    raise ValueError("Only public records may be exported")
params = record["params"]
status = record["status"]
if status not in {"FINISHED", "FAILED"}:
    raise ValueError("A finished experiment is required")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://gpu6:5050"))
mlflow.set_experiment(f"arena-public/{params['case_id']}")
with mlflow.start_run(run_name=params["candidate_id"]) as run:
    mlflow.set_tags({"split": "public", "campaign_id": params["campaign_id"]})
    mlflow.log_params({key: value for key, value in params.items() if value is not None})
    mlflow.log_dict(record, "public_record.json")
    mlflow.log_metrics({key: value for key, value in record["metrics"].items() if value is not None})
    root = mlflow.start_span_no_context(
        "public_experiment",
        inputs=params,
        attributes={"run_id": run.info.run_id},
        start_time_ns=record["start_time_ns"],
    )
    for call in record["calls"]:
        span = mlflow.start_span_no_context(
            call["name"],
            parent_span=root,
            inputs=call["inputs"],
            attributes=call["attributes"],
            start_time_ns=call["start_time_ns"],
        )
        span.end(outputs=call["outputs"], status=call["status"], end_time_ns=call["end_time_ns"])
    root.end(
        outputs=record["metrics"],
        status="OK" if status == "FINISHED" else "ERROR",
        end_time_ns=record["end_time_ns"],
    )
MlflowClient().set_terminated(run.info.run_id, status=status)
print(run.info.run_id)
```

Времена трейсов отражают исходный эксперимент; время самого run в этом примере — время экспорта. Реальную длительность также сохраняй в `metrics.wall_seconds`. При повторном экспорте запиши новый run ID в локальный журнал и пометь прежний run как заменённый; не считай дубликат ещё одним экспериментом.

API проверены по официальной документации: [tracking](https://mlflow.org/docs/latest/tracking/tracking-api/), [управление span](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_span_no_context), [завершение span](https://mlflow.org/docs/latest/api_reference/_modules/mlflow/entities/span.html), [MlflowClient](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html). Этот пример не был запущен против `gpu6:5050`.

## Приватная часть

Приватные задания, эталоны, ответы, критерийные оценки по примерам, ошибки с содержимым данных, промпты и трейсы не отправляются на общий MLflow. Они остаются в закрытом хранилище оценщика с отдельными правами доступа. Имена экспериментов, профили, теги и скрытый URL не являются разграничением доступа.

После завершения всего зафиксированного пакета финалистов оператор может опубликовать на общем сервере только разрешённый итог по каждому кандидату: идентификатор и digest кандидата, итоговую оценку, агрегированные `solver_tokens`, `solver_seconds`, `solver_unreported_attempts`, `judge_tokens`, `judge_seconds`, `judge_unreported_attempts` и публичный идентификатор кампании/метрики. До завершения пакета промежуточные оценки и расходы не публикуются. Отдельные приватные примеры, их идентификаторы и детали не экспортируются даже после окончания кампании. Публичный скрипт выше для приватного экспорта не применяется.

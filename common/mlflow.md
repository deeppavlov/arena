# MLflow

Законченные train-эксперименты, включая неудачные, сохраняются локально и в MLflow. Public даёт только итоговый score кандидата: его можно сохранить с известными исследователю ID и конфигурацией. Закрытые данные, ответы и трейсы public/private не экспортируются в общее исследовательское хранилище. Адрес трекинга задаётся `MLFLOW_TRACKING_URI`, для существующего окружения по умолчанию — `http://gpu6:5050`.

Пишет используемый агент или харнес через обычный SDK. При недоступном сервере сохрани очередь экспорта и отметку `tracking_pending`; сбой трекинга не повод повторять решение примеров. Журнал кампании остаётся источником порядка кандидатов и их родословной, MLflow связывается с ним через ID. Повтор экспорта не создаёт новое решение и не расходует лимит кандидатов.

## Что сохранять

Один run соответствует проверке кандидата на конкретном наборе. Параллельные исследователи используют отдельные процессы и run ID.

| Поле | Содержание |
| --- | --- |
| `campaign_id`, `case_id`, `candidate_id` | Кампания, кейс, кандидат |
| `candidate_digest`, `revision`, `parent_ids` | Неизменяемый пакет, ревизия кода и все родители, включая неудачные решения |
| `proposal_index`, `hypothesis`, `change` | Номер предложенного решения, гипотеза и смысл изменения |
| `model`, `harness`, `harness_version`, `parameters` | Реальная конфигурация кандидата; модель отсутствует у немодельного решателя |
| `metric_version`, `metric_package_digest`, `direction` | Версия и digest замороженного пакета метрики, направление `maximize` или `minimize` |
| `split`, `dataset_id`, `dataset_version`, `subset_id`, `conditions_id` | Части данных и условия сопоставимости; train и public не смешиваются |
| `proposed_at`, `started_at`, `completed_at` | Моменты предложения, начала и завершения для восстановления параллельного поиска |
| `status`, `error` | Результат проверки или явный сбой; ошибка не превращается в выдуманную оценку |

Для train дополнительно сохраняются состав подвыборки, версия эталона, ответы, оценки критериев, вызовы, инструменты, трейсы и решение «оставить/отклонить». Ресурсы каждого примера хранятся отдельно для `solver` и `judge`: `wall_seconds` и `usage` с `prompt_tokens`, `completion_tokens`, `total_tokens`, `attempts`, `unreported_attempts`. Итоговые `solver_tokens`, `solver_seconds`, `solver_unreported_attempts` и аналогичные `judge_*` обязательны. Расходы агрегации хранятся отдельно и включаются в итоговые `judge_*`. Время включает подготовку и повторы. При неизвестном usage сумма является нижней границей; нулевые токены немодельного решения означают полный учёт, а не отсутствие измерения. Стоимость в валюте может оставаться `null`.

Public-run содержит только доступные исследователю метаданные кандидата и итоговый score. Не добавляй к нему закрытые ID примеров, расходы выполнения, оценки критериев, ответы, ошибки с содержимым данных или трейсы. Условия сопоставимости серии задаются кампанией. Отсутствующие измерения не заменяются нулями.

На пример сохраняется один корректный ответ решателя. Порядок оценки и число голосов определяет пакет метрики; у встроенного профиля `llm` это поле `votes`. Записывай фактические обращения и расходы собственной модельной метрики, включая агрегатор. Технические повторы записываются отдельными попытками с прежними входом и настройками. Для вычисляемой метрики записываются её результат, версия и время, без искусственных судейских вызовов. Перед экспортом исключи секреты и заголовки авторизации. Полный `model_base_url` запуска тоже секретен: путь `/runs/<run_id>/<capability>/v1` содержит допуск к учёту. Удаляй `capability` из URL, входов инструментов, трасс HTTP, ошибок и остальных полей до их записи или экспорта. Пример ниже не выполняет очистку автоматически и принимает уже очищенный train-record.

Порядок кандидатов, родство, остановка поиска и выгрузка кривой/AUC описаны в [журнале поиска](search-history.md). MLflow не заменяет этот журнал. Разные подвыборки или условия не объединяются в одну сопоставимую кривую.

## Пример экспорта train-результата

Если харнес уже пишет в MLflow, используй его трекинг. Иначе сохрани пример как `log_train.py` в рабочем каталоге: `uv run --with 'mlflow>=3.6,<4' python log_train.py train_record.json`. Зависимость нужна только окружению исследователя; оценщик её не требует. Версию клиента согласуй с установленным сервером.

В `train_record.json` нужны `split: "train"`, `params` с параметрами run, числовые либо `null` значения `metrics`, `status: "FINISHED"` или `"FAILED"`, `calls` и `start_time_ns`/`end_time_ns` в Unix nanoseconds. Родителей, гипотезу, ошибки и дополнительные результаты сохраняй в этом JSON. Каждый вызов содержит `name`, `inputs`, `outputs`, `attributes`, `status: "OK"` или `"ERROR"` и временные поля. `attributes` связывает вызов с примером, ролью, критерием и номером оценки. Скрипт экспортирует уже выполненную проверку и не обращается к моделям.

```python
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record["split"] != "train":
    raise ValueError("Only train records may be exported")
params = record["params"]
status = record["status"]
if status not in {"FINISHED", "FAILED"}:
    raise ValueError("A finished experiment is required")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://gpu6:5050"))
mlflow.set_experiment(f"arena-train/{params['case_id']}")
with mlflow.start_run(run_name=params["candidate_id"]) as run:
    mlflow.set_tags({"split": "train", "campaign_id": params["campaign_id"]})
    mlflow.log_params({key: value for key, value in params.items() if value is not None})
    mlflow.log_dict(record, "train_record.json")
    mlflow.log_metrics({key: value for key, value in record["metrics"].items() if value is not None})
    root = mlflow.start_span_no_context(
        "train_experiment",
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

Времена трейсов отражают исходный эксперимент, время самого run — экспорт. Длительность также сохраняй в `metrics.wall_seconds`. При повторном экспорте запиши новый run ID в локальный журнал, пометь прежний run как заменённый и не считай дубликат новым экспериментом. Пример не проверялся против сервера трекинга.

## Закрытые результаты

Public/private задания, эталоны, ответы, отдельные оценки, промпты и трейсы остаются в закрытом хранилище оценщика. Названия экспериментов, теги и скрытые URL не разграничивают доступ.

Private API возвращает исследователю только статус, включая завершённый финал. Оценки, ресурсы и выбранные лучшие решения смотрит оператор в закрытом результате. Private автоматически не попадает в поиск, общий MLflow или график эволюции; завершение кампании само по себе не разрешает экспорт. Описанный скрипт принимает только train.

# Тот же судья на public

Решатель запускается существующим харнесом с закреплёнными настройками. Сохрани один окончательный ответ на выбранный пример и его трейс. Для оценки используй ту же функцию, которую вызывает приватный сервис:

```python
import json
import os
from pathlib import Path

from evaluator.contracts import Example, Metric
from evaluator.judge import ModelClient, judge_answer

metric = Metric.model_validate_json(Path("metric.json").read_text())
example = Example.model_validate_json(Path("example.json").read_text())
answer = Path("answer.md").read_text()
model = ModelClient(
    base_url=os.environ["ARENA_MODEL_BASE_URL"],
    model=os.environ["ARENA_MODEL"],
    api_key=Path(os.environ["ARENA_MODEL_API_KEY_FILE"]).read_text().strip(),
    parameters=json.loads(os.environ.get("ARENA_PARAMETERS", "{}")),
)
result = judge_answer(metric, example, answer, model.complete)
Path("judgement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
```

Это пример вызова для одного уже решённого public-примера, не команда автоматического запуска кампании. `example.json` — одна выбранная запись публичного JSONL. Значения модели и параметров берутся из кампании и совпадают с решателем; endpoint — только внутренний. Текст `metric.instruction` задаёт оцениваемую задачу без исходников. `example.input` получает только решатель.

`judge_answer` выполняет ровно три запроса на критерий и возвращает средний балл, отдельные голоса и обоснования. Обёртка вокруг `model.complete` может сохранить публичные запросы, ответы и время для [трейсов MLflow](mlflow.md). Не записывай заголовки авторизации. Пустой штатный ответ решателя оценивается кодом как 0 без вызова судьи; инфраструктурный сбой или невалидное судейство остаются ошибкой.

Храни результат каждого примера отдельно и сохраняй уже полученные ответы. Повторная доставка в MLflow не должна вызывать решатель или судью снова. Итог сравниваемого набора — среднее баллов всех заранее выбранных примеров; при ошибках явно укажи незавершённость, а не усредняй только успешные.

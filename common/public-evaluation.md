# Тот же судья на public

Решатель запускается существующим харнесом с закреплёнными настройками. Сохрани один корректно полученный ответ на выбранный пример, его трейс и обязательные замеры ресурсов. Все обращения харнеса к модели должны иметь шесть часов на попытку и до пяти повторов при технической ошибке, как у приватного оценщика; подробности и ожидание всего прогона — в [правилах долгих запусков](long-runs.md). Для оценки используй ту же функцию, которую вызывает приватный сервис:

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
    timeout=float(os.environ.get("ARENA_MODEL_TIMEOUT", "21600")),
    retry_delay=float(os.environ.get("ARENA_RETRY_DELAY", "30")),
)
result = {"status": "failed"}
try:
    if answer.strip():
        result.update(judge_answer(metric, example, answer, model))
    else:
        result.update({"score": 0.0, "criteria": {}, "failure": "candidate_failure"})
    result["status"] = "complete"
finally:
    result["resources"] = {
        "judge": {"usage": model.usage, "wall_seconds": model.elapsed_seconds}
    }
    Path("judgement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
```

Это пример вызова для одного уже решённого public-примера, не команда автоматического запуска кампании. `example.json` — одна выбранная запись публичного JSONL. Значения модели и параметров берутся из кампании и совпадают с решателем; endpoint — только внутренний. Текст `metric.instruction` задаёт оцениваемую задачу без исходников. `example.input` получает только решатель.

`judge_answer` получает три корректные оценки на критерий и возвращает средний балл, отдельные голоса и обоснования. При ошибке каждого запроса предусмотрено до пяти повторов; дополнительные попытки не создают дополнительные голоса. Успешный ответ модели должен содержать корректный usage. `ModelClient.usage` и `elapsed_seconds` накапливаются за время жизни клиента, включая повторы и паузы; в примере клиент создан для одного примера, а при переиспользовании сохраняй разность счётчиков до и после. Ресурсы записываются и при исчерпанных повторах, а ошибка продолжает распространяться; балл сбойному примеру не придумывается. Обёртка вокруг клиента может сохранить публичные запросы, ответы и время для [трейсов MLflow](mlflow.md), сохранив ту же проверку ответа и единый бюджет повторов. Не записывай заголовки авторизации. Пустой ответ модельного API повторяется как ошибка; штатный пустой окончательный ответ собственного харнеса оценивается кодом как 0 без вызова судьи. Исчерпанные повторы остаются явной ошибкой.

Храни результат каждого примера отдельно и сохраняй уже полученные ответы. Повторная доставка в MLflow не должна вызывать решатель или судью снова. Итог сравниваемого набора — среднее баллов всех заранее выбранных примеров; при ошибках явно укажи незавершённость, а не усредняй только успешные.

## Замеры и выбор кандидата

Для каждого примера обязательны отдельные записи `resources.solver` и `resources.judge`. Каждая содержит `wall_seconds` и вложенный `usage` с полями `prompt_tokens`, `completion_tokens`, `total_tokens`, `attempts`, `unreported_attempts`. Время решателя измеряется снаружи харнеса от начала подготовки до получения окончательного ответа, включая все вызовы инструментов, модели и повторы. Usage собирается из ответов модельного сервера для всех обращений харнеса; собственное заявление кандидата о расходе не является измерением. Код выше измеряет только судью; запуск public-харнеса обязан дополнить запись ресурсами решателя.

Неизвестный usage не заменяется нулём. Если неудачная попытка не вернула корректный usage, увеличь `unreported_attempts`; известные токены храни как неполную сумму. Для сравнения кандидатов суммируй замеры по одному и тому же завершённому набору в `solver_tokens`, `solver_seconds`, `solver_unreported_attempts`, а также отдельно `judge_tokens`, `judge_seconds`, `judge_unreported_attempts`.

Для выбора победителя используй общую с приватным оценщиком функцию:

```python
from evaluator.ranking import choose_winner

winner = choose_winner(candidate_scores)
```

Каждая строка `candidate_scores` содержит `candidate_id`, `score`, `solver_tokens`, `solver_seconds` и `solver_unreported_attempts`. Выше ранжируется больший балл; при равенстве полный учёт токенов предпочтительнее неполного, среди полных замеров выбирается меньший `solver_tokens`, затем меньшее время и стабильный порядок `candidate_id`. Если у обоих кандидатов учёт токенов неполный, сравнивается время. Судейские затраты не влияют на победителя, но сохраняются для контроля расходов кампании.

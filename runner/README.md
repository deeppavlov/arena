# Пример адаптера

`main.py` — минимальный фиксированный харнес: подготовка контекста и один вызов модели. Оператор может заменить image реализацией своего харнеса до начала кампании. Граница контейнера и stdin/stdout описаны в [контракте runner](../common/runner-contract.md).

Пример читает `/candidate/prompt.md`. Необязательный `/candidate/prepare.py` получает через stdin `{"input":"...","assets_dir":"/assets"}` и возвращает `{"context":"Подготовленный контекст"}`. Доступны стандартные Python-модули; дополнительные зависимости оператор устанавливает в image до фиксации кампании. Настройки модели и ключ остаются в gateway.

Для последующей сборки из корня репозитория:

```bash
docker build -f runner/Dockerfile -t arena-runner:local .
docker image inspect arena-runner:local --format '{{.Id}}'
```

Полученный `sha256:...` укажите в `ARENA_RUNNER_IMAGE`. На другом Docker-хосте образ заранее доставляется с тем же ID либо указывается через registry digest. Воспроизводимый релиз также фиксирует digest базового Python image. Контейнеры и права описаны в [инструкции развёртывания](../deploy/README.md).

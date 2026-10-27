# Свидетельства подготовки второй кампании SC-71

Датированные свидетельства подготовки и ремонта 24–27 октября 2026 года. Действующая точка входа — [preparation.md](../../../../../cases/sc71/campaigns/sc71-second-20261024/preparation.md), условия — [campaign.md](../../../../../cases/sc71/campaigns/sc71-second-20261024/campaign.md). Архив не заменяет живые проверки перед поиском.

| Файл | Что подтверждено |
| --- | --- |
| [local-checks.json](evidence/local-checks.json) | Ruff, оба Pyright, тесты ядра/кейса и smoke |
| [hermes-mock-check.json](evidence/hermes-mock-check.json) | Пять сценариев настоящего Hermes с подставным API |
| [build-check.json](evidence/build-check.json) | Build v2, повторное использование квитанции и readonly assets |
| [network-check.jsonl](evidence/network-check.jsonl) | Новая изоляция researcher/worker и интернет через proxy |
| [private-readiness-researcher.json](evidence/private-readiness-researcher.json) | Авторизованная защита новой серии из исследовательской среды |
| [acceptance-summary.json](evidence/acceptance-summary.json) | Три реальных синтетических сценария и отдельные расходы |
| [acceptance-wait.json](evidence/acceptance-wait.json) | Расчёт ожидания после первого сценария |
| [mlflow-trace-read.json](evidence/mlflow-trace-read.json) | Чтение полного synthetic train payload из Hermes |
| [acceptance-mlflow.json](evidence/acceptance-mlflow.json) | Экспорт и проверка трёх синтетических сценариев в обоих MLflow |
| [researcher-preflight.json](evidence/researcher-preflight.json) | Полная конфигурация исследовательского контейнера с техническим entrypoint: mounts, журнал, readiness и чтение MLflow без поиска |
| [http-acceptance-20261025.json](evidence/http-acceptance-20261025.json) | Полный синтетический train HTTP-путь после исправления каталога слотов: исполнение, результат, журнал и чтение MLflow |
| [repair-checks-20261025.json](evidence/repair-checks-20261025.json) | Обязательные проверки после исправления launcher: 1775 тестов, Ruff, оба Pyright и smoke |
| [judge-http-acceptance-20261026.json](evidence/judge-http-acceptance-20261026.json) | Полный HTTP-путь с настоящим решателем и девятью голосами модельной метрики через общий gateway из разных сетей |
| [gateway-repair-checks-20261026.json](evidence/gateway-repair-checks-20261026.json) | Проверки исправления сетевого маршрута, включая пять падений основного pytest в тестах с короткими таймаутами |
| [real-judge-repair-20261026.json](evidence/real-judge-repair-20261026.json) | Production train после ремонта: девять голосов судьи для сохранённого ответа S02, без нового вызова решателя |
| [waiter-repair-20261027.json](evidence/waiter-repair-20261027.json) | Исправление терминального статуса public `failed`, сохранение четырёх статусов и завершение зависшего обхода без перезапуска исследователя |
| [waiter-repair-checks-20261027.json](evidence/waiter-repair-checks-20261027.json) | Ruff, оба Pyright, smoke и 1779 тестов прошли; пять отдельных регрессий скрипта ожидания также проверены |

27 октября по команде пользователя исправлен созданный исследователем `board/runs/wait_public.py` в канонической кампании на GPU6: добавлен терминальный статус `failed`. Пять старых процессов ожидания завершены, разрешённые ответы API для S20/S21/S23/S25 сохранены; новые проверки не отправлялись. Отдельный зависший дочерний процесс рекурсивного обхода файловой системы завершён без остановки основного исследователя и смены внешнего run. Прежний скрипт и рабочий план сохранены в `board/runs/operator-waiter-repair-20261027`; операторская передача контекста — `board/findings/operator-waiter-repair-20261027.md`. На момент ремонта журнал содержал 32 кандидата, 64 завершённых измерения и 68 свободных мест. Ошибочный пункт рабочего плана о досрочном private заменён продолжением поиска по прежним условиям.

Закрытые вопросы, эталоны, результаты и секреты сюда не переносились. Старые private-состояние и результаты сохранены; новая серия назначена пользователем. При исходной подготовке 24–25 октября поиск и оценивание настоящих кандидатов не запускались; 26 октября сохранённый train-ответ проверен после ремонта по команде пользователя.

# Свидетельства подготовки второй кампании SC-71

Датированные свидетельства подготовки 24–25 октября 2026 года. Действующая точка входа — [preparation.md](../../../../../cases/sc71/campaigns/sc71-second-20261024/preparation.md), условия — [campaign.md](../../../../../cases/sc71/campaigns/sc71-second-20261024/campaign.md). Архив не заменяет живые проверки перед поиском.

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

Закрытые вопросы, эталоны, результаты и секреты сюда не переносились. Старые private-состояние и результаты сохранены; новая серия назначена пользователем. Поиск и оценивание настоящих кандидатов при подготовке не запускались.

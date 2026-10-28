# Свидетельства подготовки SC-35

Синтетическая приёмка новой установки на GPU6 выполнена 28 октября 2026 года. Действующие условия и готовность находятся в [каталоге кампании](../../../../../cases/sc35/campaigns/sc35-first-20261028/README.md). Архив не запускает поиск и не заменяет живые проверки перед стартом.

По отдельной команде пользователя поиск [запущен 28 октября в 11:17:41 МСК](launch-20261028.md). Свидетельства подготовки ниже описывают состояние до запуска; текущий прогресс находится в каноническом журнале на GPU6.

28 октября по запросу пользователя [исправлена буферизация операторского MLflow proxy](../../../../development/sc35-mlflow-proxy-20261028.md), вызывавшая OOM при крупных выгрузках. Исследователь и оценщики продолжили работу без перезапуска.

В тот же день исправление перенесено в [общий конфиг проекта с поведенческой регрессией](../../../../development/mlflow-proxy-regression-20261028.md); установка SC35 переведена на этот источник, очереди доставки исправны. [Серверное свидетельство](evidence/mlflow-common-config-20261028.json) фиксирует общий runtime mount и сохранение остальных контейнеров.

| Свидетельство | Что проверено |
| --- | --- |
| [local-checks.json](evidence/local-checks.json) | Проверки исходников, кейса и замороженного пакета |
| [environment.json](evidence/environment.json), [versions.json](evidence/versions.json), [image-versions.json](evidence/image-versions.json) | Среда, пути и закреплённые образы |
| [hermes-check.json](evidence/hermes-check.json), [hermes-common-check.json](evidence/hermes-common-check.json) | Настоящий Hermes с подставным API, инструменты, делегирование и отказ без лишних повторов |
| [content-check.json](evidence/content-check.json) | Настоящий контейнер baseline: четыре этапа, 113 статусов и обработка ошибки API |
| [http-acceptance.json](evidence/http-acceptance.json) | Синтетический `submit → status/result → журнал → MLflow` с рабочими образами и метрикой |
| [mlflow-read.json](evidence/mlflow-read.json) | Полный payload синтетического train-трейса, прочитанный из фактической исследовательской среды |
| [service-check.json](evidence/service-check.json), [extra-permissions.json](evidence/extra-permissions.json), [workspace-check.json](evidence/workspace-check.json) | Права UID 10001, слоты, mounts и пустой журнал настоящей кампании |
| [network-check.json](evidence/network-check.json) | Разрешённые API, запрет доступа к закрытым контурам и интернет через proxy |
| [search-readiness.json](evidence/search-readiness.json) | Авторизованная серверная защита private, условия 100/10/3 и release ID |
| [synthetic-build.json](evidence/synthetic-build.json) | Build SC-35 Hermes без входа примера и модельных вызовов |
| [upstream-check.json](evidence/upstream-check.json), [services.json](evidence/services.json) | Доступность закреплённой модели через метаданные и состояние сервисов |
| [final-preflight.json](evidence/final-preflight.json), [journal-final.json](evidence/journal-final.json), [search-readiness-final.json](evidence/search-readiness-final.json), [mlflow-read-final.json](evidence/mlflow-read-final.json) | Повторная проверка после синхронизации документов и создания маркеров READY, без запуска поиска |

HTTP-приёмка использовала вымышленный DOCX, отдельные состояние и журнал, подставной upstream и ожидаемый score 100. Указанные 18 токенов — синтетический usage ответа mock API. Реальные модели не генерировали ответы, кандидаты настоящей кампании не регистрировались, реальный private-допуск не расходовался. В архиве нет секретов, закрытых примеров и закрытых результатов.

# Исторические свидетельства подготовки

`readiness.json` и `mlflow-verification.json` сохранены без изменения байтов. Это результаты проверки исходного развёртывания 19 октября 2026 года, до запуска поиска. Поля `ready`, число решений, проверки контейнеров и адреса описывают тот момент. Они не являются текущим мониторингом GPU6 и не подтверждают готовность локального рефакторинга.

Digest замороженной метрики: `e60f2fb93a0dd6d8703ed7d40ed44c54db943287ae687891fe87697b12396e29`. Пакет в `../metric/` перенесён без изменения байтов.

## Карта локального переноса

| Прежний путь | Текущий путь от корня репозитория |
| --- | --- |
| `runner/hermes/` | `cases/sc71/harness/hermes/` |
| `deploy/sc71/` | `cases/sc71/deploy/gpu6/` |
| `deploy/sc71-mlflow/` | `cases/sc71/deploy/gpu6/mlflow/` |
| `deploy/sc71-acceptance/` | `cases/sc71/acceptance/` |
| `deploy/sc71-mlflow/export_acceptance.py` | `cases/sc71/acceptance/export_acceptance.py` |
| `runs/sc71-first-20261019/` | `cases/sc71/campaigns/sc71-first-20261019/` |
| `cases/sc71/prompts/start.md` | `cases/sc71/campaigns/sc71-first-20261019/start.md` |
| `deploy/sc71/readiness.json` | `cases/sc71/campaigns/sc71-first-20261019/evidence/readiness.json` |
| `deploy/sc71-mlflow/verification.json` | `cases/sc71/campaigns/sc71-first-20261019/evidence/mlflow-verification.json` |
| `tests/test_sc71_*.py`, `tests/test_acceptance_tracking.py`, `tests/test_hermes_runner.py` | `cases/sc71/tests/` |
| `artifacts/sc-71-candidates/` | `cases/sc71/proposals/original/` |

Абсолютные серверные пути внутри JSON, исторического отчёта и board не переписаны: они указывают на фактически проверенные файлы. Прежний серверный launcher — `/cephfs/home/malyshev_sa/arena-sc71-first-20261019/release/deploy/sc71/launch.sh`; текущая кампания уже запущена, повторно выполнять его не требуется.

## Версии исходной установки

Исторический образ evaluator/gateway/monitor: `sha256:ba01c4008398f4193a8e84538bfb884a2f0ec63a8d4351b32d2d8ded88109909`.

Исторический образ Hermes: `sha256:a3f7696d01f77aa4d2789ac26b3c7e3414707b4e697c0e1b3913c6a99442a07d`. Upstream Hermes: `86fa9e6ca1de7ee71132d2be0103f3e37d1fa872`.

Упоминания `e930bda`, `3acabc1`, `33e12e9`, `fd63a4c` и других локальных коммитов в прежних свидетельствах — метки происхождения на момент подготовки. Эти коммиты были отменены по указанию пользователя; они не являются зависимостью текущей структуры и не должны использоваться как инструкции `checkout`. Для работающей кампании сохраняются исходные серверные файлы, digest образов, контрольные суммы данных и замороженной метрики. Новый код не выдаётся за исходники этих образов.

Локальные проверки после рефакторинга относятся только к новому дереву файлов. Модельная приёмка и расходы в исторических JSON не пересчитывались и не повторялись.

[source-manifest.json](source-manifest.json) фиксирует SHA256 исходников локального staged-снимка после подготовки и до разделения кейса. Он не заявляет точного совпадения с build context старых образов: образы собирались раньше этого снимка. Старые пути сохранены как идентификаторы происхождения; локальные Git-коммиты для проверки этих hashes не нужны.

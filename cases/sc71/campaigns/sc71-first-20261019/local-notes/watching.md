**Да, после успешного завершения команды запуска терминал можно закрыть.** Скрипт запускает арену в фоновом Docker-контейнере через `up -d` — ждать окончания всей кампании не нужно. Это видно в [launch.sh](/home/kaneki/Documents/DeepPavlov/arena/deploy/sc71/launch.sh:48). `tmux` или `nohup` для этой команды не нужны.

А терминал с `ssh -L ...` нужен, пока ты смотришь MLflow через туннель. Закроешь его — пропадёт доступ к интерфейсу, сама арена продолжит работать.

Кроме MLflow можно смотреть так. Сначала подключись:

```bash
ssh gpu6
```

**Счётчик решений и текущие проверки**, обновление раз в минуту:

```bash
watch -n 60 'docker exec arena-sc71-researcher-1 python -m evaluator.search --journal /workspace/runs/sc71-first-20261019/search.sqlite3 status'
```

Поля: `proposed` — сколько решений предложено, `remaining_proposals` — сколько осталось, `running` — ID выполняющихся проверок. **100/100 решений ещё не означает завершение всех проверок.**

**История кандидатов, гипотезы и результаты измерений:**

```bash
docker exec arena-sc71-researcher-1 python -m evaluator.search \
  --journal /workspace/runs/sc71-first-20261019/search.sqlite3 export
```

**Общее состояние запуска и ошибки экспорта в MLflow:**

```bash
docker logs -f --tail 5 arena-sc71-research-monitor-1
```

`Ctrl+C` в просмотре логов или `watch` прекращает только просмотр.

Нюанс: подробный консольный вывод самого Hermes сейчас отключён, поэтому логи `arena-sc71-researcher-1` почти пустые до завершения. Для текущего прогресса полезнее журнал поиска выше.
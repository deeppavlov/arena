# Необязательный кеш исходников

Для сборки без загрузки Git-истории положите сюда `source.tar`, полученный
из закреплённого upstream:

```bash
git -C /path/to/hermes-agent archive --format=tar 86fa9e6ca1de7ee71132d2be0103f3e37d1fa872 --output=/path/to/arena/cases/sc71/harness/hermes/source-cache/source.tar
```

Dockerfile проверяет SHA256
`1ca6c5bb31f35069a13218a1401320ebd10f96bcdd80567e28e934a48642c630`
перед распаковкой. При отсутствии архива выполняется shallow fetch того же
коммита. Сам архив в Git не сохраняется.

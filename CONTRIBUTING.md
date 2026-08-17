# Contributing

Contributions should preserve the strict leave-two-out protocol and keep test
evaluation separate from model selection. Please add or update tests for any
change to data splitting, structural priors, ranking, or checkpoint loading.

Before opening a pull request, run:

```bash
ruff check .
pytest
```

Do not commit datasets, extracted feature tensors, checkpoints, run logs, or
credentials. Changes that alter reported behavior should document the old and
new protocol explicitly.


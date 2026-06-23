# SCC Dataset

Исходный файл:

- `acct_0921-0923`

Выборка для варианта:

- `acct_0921-0923_uid50728_qe7`

Фильтр:

- `UID = 50728`
- `JobName = qe7`

Размер выборки:

- 1563 строки данных
- 1564 строки с заголовком

Состояния задач в выборке:

- `COMPLETED`: 1542
- `FAILED`: 11
- `CANCELLED by 50728`: 10

Распределение запрошенных ресурсов:

- `ReqNodes = 2`: 1559 задач
- `ReqNodes = 1`: 4 задачи
- `ReqCPUS = 56`: 1561 задача
- `ReqCPUS = 28`: 2 задачи

## Генерация SLURM-заявок

Код анализа и генерации:

- `src/analyze/main.py`
- `src/analyze/analyze.py`
- `src/analyze/graphs.py`
- `src/generate/main.py`
- `src/generate/generate.py`
- `src/generate/template.py`

Логика генерации:

1. По фактическим длительностям исторических задач строится логнормальное распределение задач:

   ```text
   runtime_seconds = ElapsedRaw
   runtime_mu = mean(ln(runtime_seconds))
   runtime_sigma = stdev(ln(runtime_seconds))
   ```

2. Для каждой исторической задачи отдельно считается ошибка прогноза времени:

   ```text
   delta_seconds = TimelimitRaw * 60 - ElapsedRaw
   ```

3. По положительным значениям `delta_seconds` подбираются параметры логнормального распределения ошибки прогноза:

   ```text
   error_mu = mean(ln(delta_seconds))
   error_sigma = stdev(ln(delta_seconds))
   ```

4. При генерации новой заявки:

   - фактическая длительность задачи берётся из `lognormal(runtime_mu, runtime_sigma)`;
   - ошибка прогноза берётся из `lognormal(error_mu, error_sigma)`;
   - запрошенное время считается как `sampled_elapsed_seconds + sampled_error_seconds`;
   - в batch-файл записывается `sleep`, масштабированный для локального тестового кластера.

Посмотреть параметры распределения:

```bash
python3 -m src.analyze.main --no-graphs
```

Построить графики анализа:

```bash
python3 -m src.analyze.main
```

Сгенерировать 20 batch-файлов:

```bash
python3 -m src.generate.main --count 20
```

Файлы будут созданы в `generated/slurm_jobs/`, а манифест генерации — в `generated/slurm_jobs/manifest.csv`.

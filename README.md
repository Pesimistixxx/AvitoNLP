    R --> S
    S --> O[50 item_id]
```

### Построение индексов

`CandidateModel.fit()`:

1. Нормализует текстовые поля.
2. Строит BM25 по `title + params + description`.
3. Строит character TF-IDF по `title + params`.
4. Считает MiniLM-эмбеддинги по `title + params`.
5. Собирает статистику кликов по паре `search_query + search_location_id`.

MiniLM не дообучается: используется готовая open-source модель [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

### Инференс

`CandidateModel.predict()`:

1. Получает до 150 кандидатов от каждого поискового канала.
2. Добавляет локальные результаты, если запрос не связан с доставкой.
3. Объединяет ранги через RRF.
4. Добавляет до 15 объявлений из точной истории запроса и локации.
5. Удаляет повторы и возвращает 50 `item_id`.

## Структура проекта

```text
.
├── app/
│   ├── data_io/
│   │   ├── __init__.py
│   │   └── data_io.py
│   └── model/
│       ├── __init__.py
│       └── model.py
├── data/
│   ├── train.parquet
│   ├── benchmark_queries.parquet
│   └── benchmark_items.parquet
├── artifacts/                  # модель и кеш эмбеддингов
├── output/                     # answer.csv
├── config.py
├── main.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

`config.py` определяет каталоги входных и выходных данных. `main.py` запускает весь pipeline: загрузка → построение индексов → инференс → сохранение `answer.csv`.

## Локальный запуск

Требуется Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Перед запуском нужно положить три Parquet-файла в `data/`. Первый запуск скачает MiniLM и рассчитает эмбеддинги всех объявлений.

## Запуск в Docker

Собрать образ:

```bash
docker build -t avito-nlp .
```

Запустить pipeline:

```bash
mkdir -p output artifacts

docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  avito-nlp
```

Данные не копируются в Docker-образ. Каталоги `data`, `output` и `artifacts` подключаются как volumes.

## Кеш

После первого запуска в `artifacts/` появятся:

```text
artifacts/
├── huggingface/                        # файлы MiniLM
├── item_embeddings_<model_hash>.npy  # эмбеддинги корпуса
└── item_embeddings_<model_hash>.json # fingerprint данных
```

Если модель и тексты объявлений не изменились, эмбеддинги загружаются из кеша через memory mapping. При изменении модели, `item_id` или текстов fingerprint изменится, и кеш будет пересчитан. BM25 и TF-IDF пока строятся заново при каждом запуске.

## Результат

После запуска создаётся `output/answer.csv`:

```csv
query_id,answer
00WuFMaXSFZBxSzT,1382564bf8994a83 121fa7f5e765ce00
```

Файл содержит ровно две колонки:

- `query_id` — идентификатор запроса;
- `answer` — до 50 уникальных `item_id`, разделённых пробелом.

## Используемые инструменты

- pandas и PyArrow — загрузка Parquet и сохранение CSV;
- scikit-learn — CountVectorizer и character TF-IDF;
- NumPy/SciPy — матричные операции и разреженные индексы;
- Sentence Transformers — локальный запус multilingual MiniLM.

Внешние API во время инференса не используются. Доступ к интернету нужен только для первичного скачивания open-source модели.
import logging

import pandas as pd

from config import INPUT_DATA_DIR, OUTPUT_DATA_DIR
from app.data_io import load_data, save_data
from app.model import CandidateModel

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Последовательно выполнить весь pipeline кандидатогенерации."""
    # Единая настройка логирования.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Загружаем исходные таблицы с полной схемой признаков.
    logging.info("Загрузка данных из %s", INPUT_DATA_DIR)
    train, queries, items = load_data(INPUT_DATA_DIR)

    # Строим поисковые индексы в памяти.
    logging.info("Обучение модели кандидатогенерации")
    model = CandidateModel()
    model.fit(train, items)

    # Получаем до 50 item_id для каждого benchmark-запроса.
    logging.info("Запуск инференса для %d запросов", len(queries))
    predictions = model.predict(queries)

    # Только main.py отвечает за формирование конкурсного CSV.
    answer = pd.DataFrame(
        {
            "query_id": queries["query_id"].astype(str),
            "answer": [" ".join(item_ids) for item_ids in predictions],
        }
    )
    save_data(answer, OUTPUT_DATA_DIR)
    logging.info("Ответ сохранён: %s", OUTPUT_DATA_DIR / "answer.csv")


if __name__ == "__main__":
    main()

"""Гибридная модель кандидатогенерации для поиска услуг Авито.

Модель объединяет BM25, character TF-IDF и dense-поиск MiniLM. Ранги
каналов сводятся через Reciprocal Rank Fusion (RRF), после чего к ним
добавляются три сигнала из данных:

* точная история кликов по паре `query + location`;
* мягкое усиление предсказанных `item_microcat_id`;
* квоты по локации и категории услуг.

Модель MiniLM не дообучается. `fit()` строит поисковые индексы,
статистику кликов и маппинг запросов на микрокатегории.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


LOGGER = logging.getLogger(__name__)


class Normalizer:
    """Готовит текст для лексического и нейросетевого поиска."""

    @staticmethod
    def lexical(values: pd.Series, limit: int | None = None) -> np.ndarray:
        """Нормализовать серию текстов для BM25 и TF-IDF.

        Args:
            values: Серия со строками; NaN заменяются пустыми строками.
            limit: Максимальное число символов. `None` не ограничивает длину.

        Returns:
            NumPy-массив строк в нижнем регистре, без пунктуации и
            повторных пробелов.
        """
        text = values.fillna("").astype(str)
        if limit is not None:
            text = text.str.slice(0, limit)
        return (
            text.str.lower()
            .str.replace("ё", "е", regex=False)
            .str.replace(r"[^0-9a-zа-я]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .to_numpy()
        )

    @staticmethod
    def dense(values: pd.Series, limit: int | None = None) -> np.ndarray:
        """Подготовить текст для MiniLM без агрессивной нормализации.

        Args:
            values: Серия со строками; NaN заменяются пустыми строками.
            limit: Максимальное число символов. `None` не ограничивает длину.

        Returns:
            NumPy-массив строк без повторных пробелов.
        """
        text = values.fillna("").astype(str)
        if limit is not None:
            text = text.str.slice(0, limit)
        return text.str.replace(r"\s+", " ", regex=True).str.strip().to_numpy()


class CandidateModel:
    """Гибридная модель поиска с мягкими бизнес-ограничениями."""

    def __init__(
        self,
        model_name_or_path: str = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        artifacts_dir: str | Path = "artifacts",
        local_files_only: bool = False,
        embedding_batch_size: int = 64,
        channel_candidates: int = 150,
        result_max_size: int = 50,
        local_quota: int = 35,
        service_category_id: int = 114,
        service_quota: int = 48,
        history_limit: int = 10,
    ) -> None:
        """Задать параметры кандидатогенерации.

        Args:
            model_name_or_path: Имя Hugging Face-модели или путь к локальной копии.
            artifacts_dir: Каталог модели и кеша эмбеддингов.
            local_files_only: Если `True`, запретить скачивание MiniLM.
                При `False` модель сначала тоже ищется в локальном кеше
                и скачивается только при её отсутствии.
            embedding_batch_size: Число текстов в одном батче MiniLM.
            channel_candidates: Глубина списка каждого поискового канала.
            result_max_size: Максимальное число `item_id` в ответе.
            local_quota: Целевое число кандидатов из локации запроса.
            service_category_id: ID категории услуг. В этом датасете это `114`.
            service_quota: Целевое число кандидатов категории услуг.
            history_limit: Максимум точных исторических кандидатов.

        Raises:
            ValueError: Если квоты отрицательны или больше `result_max_size`.
        """
        if not 0 <= local_quota <= result_max_size:
            raise ValueError("local_quota должна быть в диапазоне [0, result_max_size]")
        if not 0 <= service_quota <= result_max_size:
            raise ValueError("service_quota должна быть в диапазоне [0, result_max_size]")

        self.model_name_or_path = model_name_or_path
        self.artifacts_dir = Path(artifacts_dir)
        self.local_files_only = local_files_only
        self.embedding_batch_size = embedding_batch_size
        self.channel_candidates = channel_candidates
        self.result_max_size = result_max_size
        self.local_quota = local_quota
        self.service_category_id = service_category_id
        self.service_quota = service_quota
        self.history_limit = history_limit
        self.is_fitted = False

    @staticmethod
    def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
        """Вернуть индексы лучших оценок без полной сортировки.

        Args:
            scores: Одномерный массив оценок.
            count: Число нужных индексов.

        Returns:
            Индексы по убыванию оценки.
        """
        count = min(count, len(scores))
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        positions = np.argpartition(scores, len(scores) - count)[-count:]
        return positions[np.argsort(scores[positions])[::-1]]

    @staticmethod
    def _positions_by_value(values: np.ndarray) -> dict[int, np.ndarray]:
        """Сгруппировать позиции массива по целочисленному значению.

        Args:
            values: Например, `item_location_id` или `item_microcat_id`.

        Returns:
            Словарь `{value: positions}`.
        """
        grouped: defaultdict[int, list[int]] = defaultdict(list)
        for position, value in enumerate(values):
            grouped[int(value)].append(position)
        return {
            value: np.asarray(positions, dtype=np.int64)
            for value, positions in grouped.items()
        }

    @staticmethod
    def _add_rrf(
        fused_scores: np.ndarray,
        ranked_indices: np.ndarray,
        weight: float,
        rrf_k: int = 60,
    ) -> None:
        """Добавить ранжированный список в RRF-оценки.

        Args:
            fused_scores: Общие оценки, изменяемые на месте.
            ranked_indices: Индексы от лучшего к худшему.
            weight: Вес поискового канала.
            rrf_k: Сглаживающая константа.

        Returns:
            `None`.
        """
        ranks = np.arange(1, len(ranked_indices) + 1, dtype=np.float32)
        fused_scores[ranked_indices] += weight / (rrf_k + ranks)

    @staticmethod
    def _check_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
        """Проверить наличие обязательных колонок.

        Args:
            data: Проверяемый DataFrame.
            required: Множество обязательных колонок.
            name: Имя таблицы для текста ошибки.

        Returns:
            `None`.

        Raises:
            ValueError: Если одна из колонок отсутствует.
        """
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"В {name} нет колонок: {sorted(missing)}")

    def _fit_bm25(self, documents: list[str]) -> None:
        """Построить разреженную BM25-матрицу без отдельной зависимости.

        Args:
            documents: Нормализованные тексты объявлений.

        Returns:
            `None`. Создаёт `bm25_vectorizer` и `item_bm25`.
        """
        self.bm25_vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=150_000,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        counts = self.bm25_vectorizer.fit_transform(documents).tocsr()
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = max(float(document_lengths.mean()), 1.0)

        document_frequency = np.diff(counts.tocsc().indptr)
        idf = np.log1p(
            (counts.shape[0] - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)

        k1, b = 1.5, 0.75
        row_norm = k1 * (1.0 - b + b * document_lengths / average_length)
        repeated_norm = np.repeat(row_norm, np.diff(counts.indptr))
        term_frequency = counts.data.copy()
        counts.data = (
            idf[counts.indices]
            * term_frequency
            * (k1 + 1.0)
            / (term_frequency + repeated_norm)
        ).astype(np.float32)
        self.item_bm25 = counts

    def _fit_query_knowledge(self, train: pd.DataFrame) -> None:
        """Построить маппинг train-запросов на микрокатегории.

        Args:
            train: Таблица кликов с нормализованным `normalized_query`.

        Returns:
            `None`. Создаёт `query_microcats`, `reference_queries` и
            char TF-IDF-индекс train-запросов.
        """
        # Один item может встречаться в train много раз. Дедупликация не даёт
        # одному популярному объявлению полностью определить микрокатегорию.
        unique_pairs = train.drop_duplicates(["normalized_query", "item_id"])
        microcat_counts = (
            unique_pairs.groupby(["normalized_query", "item_microcat_id"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(
                ["normalized_query", "count"], ascending=[True, False]
            )
            .groupby("normalized_query", sort=False)
            .head(3)
        )
        self.query_microcats = {
            query: list(
                zip(
                    group["item_microcat_id"].astype(int),
                    group["count"].astype(int),
                )
            )
            for query, group in microcat_counts.groupby(
                "normalized_query", sort=False
            )
        }

        # Character n-граммы устойчивы к опечаткам и формам русских слов, поэтому ими
        # ищем похожий train-запрос, когда точного совпадения нет.
        self.reference_queries = np.asarray(
            list(self.query_microcats), dtype=object
        )
        self.query_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.reference_query_matrix = self.query_vectorizer.fit_transform(
            self.reference_queries
        )

    def _predict_microcats(self, query_texts: np.ndarray) -> list[list[int]]:
        """Предсказать до трёх микрокатегорий для каждого запроса.

        Args:
            query_texts: Нормализованные benchmark-запросы.

        Returns:
            Список микрокатегорий для каждого запроса.
        """
        query_matrix = self.query_vectorizer.transform(query_texts)
        predictions: list[list[int]] = []

        for start in range(0, len(query_texts), 64):
            stop = min(start + 64, len(query_texts))
            similarities = (
                query_matrix[start:stop] @ self.reference_query_matrix.T
            ).toarray()

            for query, row in zip(query_texts[start:stop], similarities):
                # Точный train-запрос надёжнее поиска соседей.
                exact = self.query_microcats.get(str(query))
                if exact is not None:
                    predictions.append([microcat for microcat, _ in exact])
                    continue

                neighbours = self._top_indices(row, min(20, len(row)))
                votes: defaultdict[int, float] = defaultdict(float)
                for neighbour in neighbours:
                    similarity = float(row[neighbour])
                    if similarity <= 0:
                        continue
                    reference_query = self.reference_queries[neighbour]
                    for microcat, count in self.query_microcats[reference_query]:
                        # sqrt не позволяет очень частому train-запросу подавить всех соседей.
                        votes[microcat] += similarity * float(np.sqrt(count))

                predictions.append(
                    sorted(votes, key=votes.get, reverse=True)[:3]
                )

        return predictions

    def _embedding_fingerprint(
        self,
        items: pd.DataFrame,
        dense_documents: list[str],
    ) -> str:
        """Рассчитать fingerprint модели и текстов корпуса.

        Args:
            items: Корпус с `item_id`.
            dense_documents: Тексты для MiniLM.

        Returns:
            SHA-256 fingerprint кеша.
        """
        digest = hashlib.sha256(self.model_name_or_path.encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(
                items["item_id"], index=False
            ).values.tobytes()
        )
        digest.update(
            pd.util.hash_pandas_object(
                pd.Series(dense_documents), index=False
            ).values.tobytes()
        )
        return digest.hexdigest()

    def _fit_dense(self, items: pd.DataFrame, dense_documents: list[str]) -> None:
        """Загрузить MiniLM или рассчитать эмбеддинги корпуса.

        Args:
            items: Корпус объявлений.
            dense_documents: Тексты в порядке строк `items`.

        Returns:
            `None`. Создаёт `encoder` и `item_embeddings`.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_cache = self.artifacts_dir / "huggingface"
        # Сначала загружаем только локальные файлы. Это убирает долгие сетевые
        # HEAD-запросы Hugging Face при повторном или полностью offline-запуске.
        try:
            self.encoder = SentenceTransformer(
                self.model_name_or_path,
                cache_folder=str(model_cache),
                local_files_only=True,
            )
        except OSError:
            if self.local_files_only:
                raise
            LOGGER.info("Локальный кеш MiniLM не найден; загрузка с Hugging Face")
            self.encoder = SentenceTransformer(
                self.model_name_or_path,
                cache_folder=str(model_cache),
            )
        self.encoder.max_seq_length = 128

        model_key = hashlib.sha1(
            self.model_name_or_path.encode("utf-8")
        ).hexdigest()[:10]
        embeddings_path = self.artifacts_dir / f"item_embeddings_{model_key}.npy"
        metadata_path = embeddings_path.with_suffix(".json")
        fingerprint = self._embedding_fingerprint(items, dense_documents)

        if embeddings_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                LOGGER.info("Загрузка кеша MiniLM: %s", embeddings_path)
                # Memory mapping экономит оперативную память на матрице корпуса.
                self.item_embeddings = np.load(embeddings_path, mmap_mode="r")
                return

        LOGGER.info("Расчёт MiniLM эмбеддингов для %d объявлений", len(items))
        embeddings = self.encoder.encode(
            dense_documents,
            batch_size=self.embedding_batch_size,
            # Для нормализованных векторов dot product равен cosine similarity.
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(embeddings_path, embeddings)
        metadata_path.write_text(
            json.dumps({"fingerprint": fingerprint}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.item_embeddings = np.load(embeddings_path, mmap_mode="r")

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> "CandidateModel":
        """Построить поисковые индексы и статистику train.

        Args:
            train: Пары `запрос — выбранное объявление`.
            items: Корпус объявлений для поиска.

        Returns:
            Текущий экземпляр `CandidateModel`, готовый к `predict()`.

        Raises:
            ValueError: Если нет обязательных колонок или `item_id` повторяются.
        """
        self._check_columns(
            train,
            {
                "search_query",
                "search_location_id",
                "item_id",
                "item_microcat_id",
            },
            "train",
        )
        self._check_columns(
            items,
            {
                "item_id",
                "item_title_raw",
                "item_description_raw",
                "item_infm_params_text",
                "item_category_id",
                "item_microcat_id",
                "item_location_id",
                "item_rating_reviews_count",
            },
            "items",
        )
        if items["item_id"].duplicated().any():
            raise ValueError("В benchmark_items есть повторы item_id")

        self.item_ids = items["item_id"].astype(str).to_numpy()
        self.locations = items["item_location_id"].to_numpy(dtype=np.int64)
        self.categories = items["item_category_id"].to_numpy(dtype=np.int64)
        self.microcats = items["item_microcat_id"].to_numpy(dtype=np.int64)
        self.location_positions = self._positions_by_value(self.locations)
        self.microcat_positions = self._positions_by_value(self.microcats)
        self.service_positions = np.flatnonzero(
            self.categories == self.service_category_id
        )

        normalized_train = train.copy()
        normalized_train["normalized_query"] = Normalizer.lexical(
            normalized_train["search_query"]
        )
        self._fit_query_knowledge(normalized_train)

        title = Normalizer.lexical(items["item_title_raw"])
        params = Normalizer.lexical(items["item_infm_params_text"], limit=500)
        description = Normalizer.lexical(items["item_description_raw"], limit=300)
        # Заголовок повторяется дважды: для коротких запросов он надёжнее длинного описания.
        lexical_documents = [
            f"{item_title} {item_title} {item_params} {item_description}"
            for item_title, item_params, item_description in zip(
                title, params, description
            )
        ]

        LOGGER.info("Построение BM25 и char TF-IDF индексов")
        self._fit_bm25(lexical_documents)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.item_chars = self.char_vectorizer.fit_transform(
            [
                f"{item_title} {item_params}"
                for item_title, item_params in zip(title, params)
            ]
        )

        # Dense-текст оставлен идентичным baseline 0.54: только title + params.
        # Длинное рекламное описание не размывает смысл, а fingerprint кеша
        # MiniLM не меняется.
        dense_title = Normalizer.dense(items["item_title_raw"])
        dense_params = Normalizer.dense(items["item_infm_params_text"], limit=500)
        dense_documents = [
            ". ".join(part for part in parts if part)
            for parts in zip(dense_title, dense_params)
        ]
        self._fit_dense(items, dense_documents)

        # История намеренно точная: совпадение query + location редко, но является
        # высокоточным сигналом и поэтому получает первые позиции.
        item_position = {
            item_id: position for position, item_id in enumerate(self.item_ids)
        }
        history = normalized_train[
            normalized_train["item_id"].isin(item_position)
        ].copy()
        history["position"] = history["item_id"].map(item_position)
        history = (
            history.groupby(
                ["normalized_query", "search_location_id", "position"],
                sort=False,
            )
            .size()
            .rename("clicks")
            .reset_index()
            .sort_values(
                ["normalized_query", "search_location_id", "clicks"],
                ascending=[True, True, False],
            )
        )
        self.history_map = {
            (query, int(location)): group["position"].astype(int).tolist()
            for (query, location), group in history.groupby(
                ["normalized_query", "search_location_id"], sort=False
            )
        }

        # Число отзывов используется только как очень слабый tie-breaker.
        reviews = items["item_rating_reviews_count"].fillna(0).clip(lower=0)
        self.quality = np.log1p(reviews.to_numpy(dtype=np.float32))
        if self.quality.max() > 0:
            self.quality /= self.quality.max()

        self.is_fitted = True
        return self

    def _apply_microcat_boost(
        self,
        fused_scores: np.ndarray,
        predicted_microcats: list[int],
    ) -> None:
        """Мягко усилить кандидатов предсказанных микрокатегорий.

        Args:
            fused_scores: RRF-оценки, изменяемые на месте.
            predicted_microcats: До трёх `item_microcat_id` по убыванию
                уверенности.

        Returns:
            `None`.
        """
        # Множители, а не жёсткий фильтр: ошибочная микрокатегория не должна
        # полностью удалить релевантное объявление из другой категории.
        for microcat, multiplier in zip(
            predicted_microcats, (1.45, 1.25, 1.10)
        ):
            positions = self.microcat_positions.get(microcat)
            if positions is not None:
                fused_scores[positions] *= multiplier

    def _select_with_quotas(
        self,
        fused_scores: np.ndarray,
        normalized_query: str,
        location: int,
    ) -> list[int]:
        """Выбрать финальные позиции с квотами локации и услуг.

        Args:
            fused_scores: Общие RRF-оценки всех объявлений.
            normalized_query: Нормализованный запрос для поиска истории.
            location: `search_location_id`.

        Returns:
            До `result_max_size` уникальных позиций корпуса.
        """
        history = list(
            dict.fromkeys(
                self.history_map.get((normalized_query, location), [])[
                    : self.history_limit
                ]
            )
        )
        # Даже точная история подчиняется страховочной квоте: сначала сохраняем
        # клики по услугам, затем не более двух редких кликов из других категорий.
        non_service_limit = self.result_max_size - self.service_quota
        selected = [
            position
            for position in history
            if self.categories[position] == self.service_category_id
        ]
        selected.extend(
            position
            for position in history
            if self.categories[position] != self.service_category_id
        )
        selected = selected[: self.history_limit]
        if non_service_limit < self.history_limit:
            service_history = [
                position
                for position in selected
                if self.categories[position] == self.service_category_id
            ]
            other_history = [
                position
                for position in selected
                if self.categories[position] != self.service_category_id
            ][:non_service_limit]
            selected = service_history + other_history
        selected_set = set(selected)

        def add_ranked(pool: np.ndarray, target_count: int, kind: str) -> None:
            """Добавить лучшие позиции пула до достижения заданной квоты.

            Args:
                pool: Допустимые позиции объявлений в корпусе.
                target_count: Требуемое число кандидатов выбранного типа.
                kind: Тип счётчика: `local`, `service` или `all`.

            Returns:
                `None`. Список `selected` изменяется на месте.
            """
            if len(pool) == 0 or len(selected) >= self.result_max_size:
                return

            ranked = pool[
                self._top_indices(
                    fused_scores[pool],
                    min(len(pool), max(self.channel_candidates, 200)),
                )
            ]
            for position in ranked:
                position = int(position)
                if position in selected_set:
                    continue
                selected.append(position)
                selected_set.add(position)

                if kind == "local":
                    current_count = sum(
                        self.locations[index] == location for index in selected
                    )
                elif kind == "service":
                    current_count = sum(
                        self.categories[index] == self.service_category_id
                        for index in selected
                    )
                else:
                    current_count = len(selected)

                if current_count >= target_count or len(selected) >= self.result_max_size:
                    break

        local_positions = self.location_positions.get(
            location, np.empty(0, dtype=np.int64)
        )
        local_target = min(self.local_quota, len(local_positions))

        # В train локация выбранного объявления совпадает с запросом примерно в 83%
        # случаев. Сначала набираем локальные услуги. Если их мало, локальная квота
        # может быть недобрана: категория услуг статистически является более сильным
        # сигналом, поэтому она имеет приоритет над локальностью.
        local_service_positions = local_positions[
            self.categories[local_positions] == self.service_category_id
        ]
        add_ranked(local_service_positions, local_target, "local")

        # 497658 из 497673 положительных train-строк относятся к категории 114.
        # Квота 48/50 удаляет категориальный шум, но оставляет два fallback-места
        # для редких исключений вместо опасного жёсткого фильтра.
        service_target = min(self.service_quota, len(self.service_positions))
        add_ranked(self.service_positions, service_target, "service")

        # После выполнения сервисной квоты свободные fallback-места сначала отдаём
        # локальным объявлениям, даже если они относятся к редкой другой категории.
        add_ranked(local_positions, local_target, "local")

        all_positions = np.arange(len(self.item_ids), dtype=np.int64)
        add_ranked(all_positions, self.result_max_size, "all")
        return selected[: self.result_max_size]

    def predict(self, queries: pd.DataFrame) -> list[list[str]]:
        """Найти до 50 кандидатов для каждого запроса.

        Args:
            queries: Benchmark-запросы в исходном порядке.

        Returns:
            Список списков `item_id`; порядок совпадает с `queries`.

        Raises:
            RuntimeError: Если `fit()` ещё не вызван.
            ValueError: Если нет обязательных колонок.
        """
        if not self.is_fitted:
            raise RuntimeError("Сначала вызовите CandidateModel.fit()")
        self._check_columns(
            queries,
            {
                "search_query",
                "search_location_id",
                "search_is_delivery_search",
                "search_infm_params_text",
            },
            "queries",
        )

        query = Normalizer.lexical(queries["search_query"])
        filters = Normalizer.lexical(
            queries["search_infm_params_text"], limit=500
        )
        predicted_microcats = self._predict_microcats(query)
        lexical_queries = [
            f"{query_text} {query_text} {query_filters}"
            for query_text, query_filters in zip(query, filters)
        ]

        query_bm25 = self.bm25_vectorizer.transform(lexical_queries).tocsr()
        # BM25 учитывает наличие терма в коротком запросе, а не число его повторов.
        query_bm25.data.fill(1.0)
        query_chars = self.char_vectorizer.transform(lexical_queries)

        dense_query = Normalizer.dense(queries["search_query"])
        dense_filters = Normalizer.dense(
            queries["search_infm_params_text"], limit=500
        )
        dense_queries = [
            ". ".join(part for part in parts if part)
            for parts in zip(dense_query, dense_filters)
        ]
        query_embeddings = self.encoder.encode(
            dense_queries,
            batch_size=self.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        predictions: list[list[str]] = []
        inference_batch_size = 16

        for start in range(0, len(queries), inference_batch_size):
            stop = min(start + inference_batch_size, len(queries))
            bm25_scores = (
                query_bm25[start:stop] @ self.item_bm25.T
            ).toarray()
            char_scores = (
                query_chars[start:stop] @ self.item_chars.T
            ).toarray()
            dense_scores = (
                query_embeddings[start:stop] @ self.item_embeddings.T
            )

            for offset in range(stop - start):
                query_index = start + offset
                fused = 1e-8 * self.quality.copy()
                channel_scores = (
                    (bm25_scores[offset], 1.0),
                    (char_scores[offset], 0.8),
                    (dense_scores[offset], 1.2),
                )

                # Глобальные списки не дают локальной эвристике удалить исполнителей,
                # работающих удалённо или обслуживающих соседние локации.
                for scores, weight in channel_scores:
                    ranked = self._top_indices(
                        scores, self.channel_candidates
                    )
                    self._add_rrf(fused, ranked, weight)

                row = queries.iloc[query_index]
                location = int(row["search_location_id"])
                local_positions = self.location_positions.get(
                    location, np.empty(0, dtype=np.int64)
                )

                # Для delivery-запроса локальность не усиливается. Сейчас все benchmark-
                # запросы имеют delivery=0, но ветка оставлена для общей корректности.
                if not bool(row["search_is_delivery_search"]):
                    for scores, weight in channel_scores:
                        local_rank = self._top_indices(
                            scores[local_positions],
                            min(self.channel_candidates, len(local_positions)),
                        )
                        self._add_rrf(
                            fused,
                            local_positions[local_rank],
                            0.6 * weight,
                        )

                self._apply_microcat_boost(
                    fused, predicted_microcats[query_index]
                )
                selected = self._select_with_quotas(
                    fused,
                    normalized_query=str(query[query_index]),
                    location=location,
                )

                expected_size = min(self.result_max_size, len(self.item_ids))
                if len(selected) != expected_size or len(selected) != len(set(selected)):
                    raise RuntimeError("Не удалось сформировать уникальный top-N")
                predictions.append(self.item_ids[selected].tolist())

            LOGGER.info("Обработано запросов: %d/%d", stop, len(queries))

        return predictions

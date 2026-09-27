"""Гибридная кандидатогенерация V4 для поиска услуг Авито.

Версия объединяет пять независимых текстовых каналов, географический
поиск по координатам, статистику query/filter -> microcat, query profile по
историческим кликам и точное совпадение поисковых фильтров с параметрами.
Финальные 50 кандидатов выбираются с квотами, потому что Recall@50 зависит
от покрытия, а не от порядка внутри ответа.

MiniLM не дообучается. `fit()` строит разреженные индексы, статистику
кликов и географические центры; эмбеддинги объявлений кешируются на диске.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


LOGGER = logging.getLogger(__name__)


class Normalizer:
    """Подготавливает текст для лексического и нейросетевого поиска."""

    @staticmethod
    def lexical(values: pd.Series, limit: int | None = None) -> np.ndarray:
        """Нормализовать серию текстов для BM25 и TF-IDF.

        Args:
            values: Серия со строками; пропуски заменяются пустыми строками.
            limit: Максимальное число исходных символов. `None` не обрезает текст.

        Returns:
            NumPy-массив строк в нижнем регистре без пунктуации и лишних пробелов.
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
            values: Серия со строками; пропуски заменяются пустыми строками.
            limit: Максимальное число исходных символов. `None` не обрезает текст.

        Returns:
            NumPy-массив строк без повторных пробелов.
        """
        text = values.fillna("").astype(str)
        if limit is not None:
            text = text.str.slice(0, limit)
        return text.str.replace(r"\s+", " ", regex=True).str.strip().to_numpy()


class CandidateModel:
    """Гибридная модель поиска с microcat- и geo-квотами."""

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
        service_category_id: int = 114,
        history_limit: int = 8,
        geo_min_quota: int = 32,
        geo_max_quota: int = 45,
        filter_quota: int = 10,
        profile_quota: int = 6,
        quality_weight: float = 0.002,
        geo_weight: float = 0.001,
    ) -> None:
        """Задать параметры кандидатогенерации.

        Args:
            model_name_or_path: Имя Hugging Face-модели или локальный путь.
            artifacts_dir: Каталог модели и кеша эмбеддингов.
            local_files_only: Запретить скачивание MiniLM при отсутствии кеша.
            embedding_batch_size: Число текстов в одном батче MiniLM.
            channel_candidates: Глубина ранжированного списка каждого канала.
            result_max_size: Число кандидатов в финальном ответе.
            service_category_id: Категория услуг; в данных задачи это `114`.
            history_limit: Максимум точных исторических кандидатов.
            geo_min_quota: Минимальная квота географически близких объявлений.
            geo_max_quota: Максимальная квота географически близких объявлений.
            filter_quota: Квота объявлений с полным совпадением фильтра.
            profile_quota: Квота кандидатов query-profile канала.
            quality_weight: Вес сглаженного prior качества объявления.
            geo_weight: Дополнительный вес близости внутри geo-пула.

        Raises:
            ValueError: Если квоты несовместимы с размером результата.
        """
        if not 0 <= geo_min_quota <= geo_max_quota <= result_max_size:
            raise ValueError(
                "Ожидается 0 <= geo_min_quota <= geo_max_quota <= result_max_size"
            )
        if result_max_size <= 0:
            raise ValueError("result_max_size должен быть положительным")
        if not 0 <= filter_quota <= result_max_size:
            raise ValueError("filter_quota должна быть в диапазоне результата")
        if not 0 <= profile_quota <= result_max_size:
            raise ValueError("profile_quota должна быть в диапазоне результата")

        self.model_name_or_path = model_name_or_path
        self.artifacts_dir = Path(artifacts_dir)
        self.local_files_only = local_files_only
        self.embedding_batch_size = embedding_batch_size
        self.channel_candidates = channel_candidates
        self.result_max_size = result_max_size
        self.service_category_id = service_category_id
        self.history_limit = history_limit
        self.geo_min_quota = geo_min_quota
        self.geo_max_quota = geo_max_quota
        self.filter_quota = filter_quota
        self.profile_quota = profile_quota
        self.quality_weight = quality_weight
        self.geo_weight = geo_weight
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
            values: Например, массив `item_location_id` или `item_microcat_id`.

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
        """Добавить ранжированный список в общие RRF-оценки.

        Args:
            fused_scores: Общие оценки, изменяемые на месте.
            ranked_indices: Позиции объявлений от лучшего к худшему.
            weight: Вес поискового канала.
            rrf_k: Сглаживающая константа RRF.

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
            name: Имя таблицы для сообщения об ошибке.

        Returns:
            `None`.

        Raises:
            ValueError: Если одна из обязательных колонок отсутствует.
        """
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"В {name} нет колонок: {sorted(missing)}")

    @staticmethod
    def _build_bm25(
        documents: list[str],
        ngram_range: tuple[int, int],
        max_features: int,
    ) -> tuple[CountVectorizer, csr_matrix]:
        """Построить BM25-матрицу средствами scikit-learn.

        Args:
            documents: Нормализованные документы одного поля.
            ngram_range: Диапазон словных n-грамм.
            max_features: Максимальный размер словаря.

        Returns:
            Обученный `CountVectorizer` и BM25-взвешенная CSR-матрица.
        """
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        counts = vectorizer.fit_transform(documents).tocsr()
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
        return vectorizer, counts

    @staticmethod
    def _top_microcat_map(
        train: pd.DataFrame,
        keys: list[str],
        limit: int,
    ) -> dict[object, list[tuple[int, int]]]:
        """Собрать частотный microcat-маппинг для заданного ключа.

        Args:
            train: Нормализованные положительные пары train.
            keys: Колонки ключа, например query или query + filter.
            limit: Максимум микрокатегорий для одного ключа.

        Returns:
            Словарь с парами `(item_microcat_id, unique_item_count)`.
        """
        unique_pairs = train.drop_duplicates(keys + ["item_id"])
        counts = (
            unique_pairs.groupby(keys + ["item_microcat_id"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(keys + ["count"], ascending=[True] * len(keys) + [False])
            .groupby(keys, sort=False)
            .head(limit)
        )

        result: dict[object, list[tuple[int, int]]] = {}
        group_key: str | list[str] = keys[0] if len(keys) == 1 else keys
        for key, group in counts.groupby(group_key, sort=False):
            result[key] = list(
                zip(
                    group["item_microcat_id"].astype(int),
                    group["count"].astype(int),
                )
            )
        return result

    def _fit_query_knowledge(self, train: pd.DataFrame) -> None:
        """Построить query/filter-маппинги и индекс похожих запросов.

        Args:
            train: Таблица с `normalized_query` и `normalized_filter`.

        Returns:
            `None`. Сохраняет статистику микрокатегорий и char TF-IDF-индекс.
        """
        self.query_microcats = self._top_microcat_map(
            train, ["normalized_query"], limit=5
        )
        nonempty_filters = train[train["normalized_filter"].ne("")]
        self.filter_microcats = self._top_microcat_map(
            nonempty_filters, ["normalized_filter"], limit=10
        )
        self.query_filter_microcats = self._top_microcat_map(
            nonempty_filters,
            ["normalized_query", "normalized_filter"],
            limit=5,
        )

        # Character n-граммы устойчивы к опечаткам и русской морфологии.
        self.reference_queries = np.asarray(list(self.query_microcats), dtype=object)
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

    def _fit_query_profiles(
        self,
        normalized_train: pd.DataFrame,
        raw_train: pd.DataFrame,
    ) -> None:
        """Собрать текстовые профили запросов из исторически выбранных объявлений.

        Args:
            normalized_train: Train с колонкой `normalized_query`.
            raw_train: Исходный train с текстами объявлений.

        Returns:
            `None`. Создаёт профиль каждого train-запроса и массив в порядке
            `reference_queries`.
        """
        pair_counts = (
            normalized_train.groupby(["normalized_query", "item_id"], sort=False)
            .size()
            .rename("clicks")
            .reset_index()
            .sort_values(
                ["normalized_query", "clicks"], ascending=[True, False]
            )
            .groupby("normalized_query", sort=False)
            .head(3)
        )
        item_texts = raw_train[
            ["item_id", "item_title_raw", "item_infm_params_text"]
        ].drop_duplicates("item_id")
        profiles = pair_counts.merge(item_texts, on="item_id", how="left")
        titles = Normalizer.dense(profiles["item_title_raw"])
        params = Normalizer.dense(profiles["item_infm_params_text"], limit=180)
        profiles["profile_part"] = [
            ". ".join(part for part in parts if part)
            for parts in zip(titles, params)
        ]
        self.query_profiles = (
            profiles.groupby("normalized_query", sort=False)["profile_part"]
            .agg(". ".join)
            .to_dict()
        )
        self.reference_profiles = np.asarray(
            [self.query_profiles.get(str(query), "") for query in self.reference_queries],
            dtype=object,
        )

    @staticmethod
    def _add_probability_evidence(
        scores: defaultdict[int, float],
        evidence: list[tuple[int, float]],
        weight: float,
    ) -> None:
        """Добавить нормированное microcat-свидетельство.

        Args:
            scores: Накопленные оценки, изменяемые на месте.
            evidence: Пары `(microcat, confidence)`.
            weight: Доверие к источнику.

        Returns:
            `None`.
        """
        total = sum(value for _, value in evidence)
        if total <= 0:
            return
        for microcat, value in evidence:
            scores[int(microcat)] += weight * float(value) / total

    def _predict_microcats(
        self,
        query_texts: np.ndarray,
        filter_texts: np.ndarray,
    ) -> tuple[list[list[tuple[int, float]]], list[str]]:
        """Предсказать микрокатегории и построить query-profile тексты.

        Args:
            query_texts: Нормализованные benchmark-запросы.
            filter_texts: Нормализованные поисковые фильтры.

        Returns:
            Списки `(microcat, probability)` и тексты query profile.
        """
        query_matrix = self.query_vectorizer.transform(query_texts)
        predictions: list[list[tuple[int, float]]] = []
        profile_queries: list[str] = []

        for start in range(0, len(query_texts), 64):
            stop = min(start + 64, len(query_texts))
            similarities = (
                query_matrix[start:stop] @ self.reference_query_matrix.T
            ).toarray()

            for query, query_filter, row in zip(
                query_texts[start:stop],
                filter_texts[start:stop],
                similarities,
            ):
                scores: defaultdict[int, float] = defaultdict(float)
                neighbours = np.empty(0, dtype=np.int64)

                exact_pair = self.query_filter_microcats.get(
                    (str(query), str(query_filter))
                )
                if exact_pair is not None:
                    self._add_probability_evidence(scores, exact_pair, weight=3.0)

                query_evidence = self.query_microcats.get(str(query))
                if query_evidence is None:
                    neighbours = self._top_indices(row, min(20, len(row)))
                    neighbour_votes: defaultdict[int, float] = defaultdict(float)
                    for neighbour in neighbours:
                        similarity = float(row[neighbour])
                        if similarity <= 0:
                            continue
                        reference = self.reference_queries[neighbour]
                        for microcat, count in self.query_microcats[reference]:
                            neighbour_votes[microcat] += similarity * float(
                                np.sqrt(count)
                            )
                    query_evidence = list(neighbour_votes.items())

                self._add_probability_evidence(scores, query_evidence, weight=2.0)
                filter_evidence = self.filter_microcats.get(str(query_filter), [])
                self._add_probability_evidence(scores, filter_evidence, weight=1.5)

                best = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:5]
                total = sum(score for _, score in best)
                predictions.append(
                    [
                        (microcat, score / total)
                        for microcat, score in best
                    ]
                    if total > 0
                    else []
                )

                # Для известного запроса используем его клики. Для нового —
                # профиль ближайшего train-запроса, но только при ненулевой
                # char-схожести. Исходный запрос остаётся первым и защищает от
                # полного дрейфа к соседнему интенту.
                profile = self.query_profiles.get(str(query), "")
                if not profile:
                    for neighbour in neighbours:
                        if float(row[neighbour]) <= 0:
                            break
                        candidate_profile = str(self.reference_profiles[neighbour])
                        if candidate_profile:
                            profile = candidate_profile
                            break
                profile_queries.append(
                    f"{query}. {profile}" if profile else ""
                )

        return predictions, profile_queries

    @staticmethod
    def _haversine_km(
        center_latitude: float,
        center_longitude: float,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """Рассчитать расстояние от центра до массива координат.

        Args:
            center_latitude: Широта центра в градусах.
            center_longitude: Долгота центра в градусах.
            latitudes: Широты объявлений в градусах.
            longitudes: Долготы объявлений в градусах.

        Returns:
            Расстояния в километрах; невалидные координаты остаются `NaN`.
        """
        center_lat = np.radians(center_latitude)
        center_lon = np.radians(center_longitude)
        item_lat = np.radians(latitudes)
        item_lon = np.radians(longitudes)
        value = (
            np.sin((item_lat - center_lat) / 2.0) ** 2
            + np.cos(center_lat)
            * np.cos(item_lat)
            * np.sin((item_lon - center_lon) / 2.0) ** 2
        )
        return (12_742.0 * np.arcsin(np.minimum(1.0, np.sqrt(value)))).astype(
            np.float32
        )

    def _fit_geography(self, train: pd.DataFrame) -> None:
        """Оценить центры поисковых локаций и локальность микрокатегорий.

        Args:
            train: Положительные пары с координатами выбранного объявления.

        Returns:
            `None`. Создаёт центры локаций и сглаженные near-25km priors.
        """
        geo = train[
            [
                "search_location_id",
                "item_microcat_id",
                "item_latitude",
                "item_longitude",
            ]
        ].copy()
        geo["latitude"] = pd.to_numeric(geo["item_latitude"], errors="coerce")
        geo["longitude"] = pd.to_numeric(geo["item_longitude"], errors="coerce")
        valid = geo["latitude"].between(40, 82) & geo["longitude"].between(10, 190)
        geo = geo[valid].copy()

        centers = geo.groupby("search_location_id").agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
        self.location_centers = {
            int(location): (float(row.latitude), float(row.longitude))
            for location, row in centers.iterrows()
        }

        center_lat = geo["search_location_id"].map(centers["latitude"])
        center_lon = geo["search_location_id"].map(centers["longitude"])
        geo["near"] = self._haversine_km(
            center_lat.to_numpy(),
            center_lon.to_numpy(),
            geo["latitude"].to_numpy(),
            geo["longitude"].to_numpy(),
        ) <= 25.0
        self.global_near_rate = float(geo["near"].mean())

        # Сглаживание не позволяет редкой локации или microcat получить квоту
        # из нескольких случайных кликов.
        location_stats = geo.groupby("search_location_id")["near"].agg(["sum", "count"])
        microcat_stats = geo.groupby("item_microcat_id")["near"].agg(["sum", "count"])
        location_prior = 100.0
        microcat_prior = 200.0
        self.location_near_rate = (
            (location_stats["sum"] + location_prior * self.global_near_rate)
            / (location_stats["count"] + location_prior)
        ).to_dict()
        self.microcat_near_rate = (
            (microcat_stats["sum"] + microcat_prior * self.global_near_rate)
            / (microcat_stats["count"] + microcat_prior)
        ).to_dict()
        self.geo_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _geo_candidates(self, location: int) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть сервисные объявления в адаптивном радиусе.

        Args:
            location: `search_location_id` запроса.

        Returns:
            Позиции объявлений и их proximity-score от нуля до единицы.
        """
        cached = self.geo_cache.get(location)
        if cached is not None:
            return cached

        center = self.location_centers.get(location)
        if center is None:
            exact = self.location_positions.get(
                location, np.empty(0, dtype=np.int64)
            )
            exact = exact[self.categories[exact] == self.service_category_id]
            result = (exact, np.ones(len(exact), dtype=np.float32))
            self.geo_cache[location] = result
            return result

        distances = self._haversine_km(
            center[0],
            center[1],
            self.item_latitudes,
            self.item_longitudes,
        )
        required = max(self.channel_candidates, self.result_max_size)
        positions = np.empty(0, dtype=np.int64)
        for radius in (25.0, 50.0, 100.0):
            positions = np.flatnonzero(
                (self.categories == self.service_category_id)
                & np.isfinite(distances)
                & (distances <= radius)
            )
            if len(positions) >= required:
                break

        proximity = np.exp(-distances[positions] / 25.0).astype(np.float32)
        result = (positions, proximity)
        self.geo_cache[location] = result
        return result

    def _filter_candidates(self, filter_text: str) -> np.ndarray:
        """Найти объявления со всеми значимыми словами поискового фильтра.

        Args:
            filter_text: Нормализованный `search_infm_params_text`.

        Returns:
            Позиции сервисных объявлений с полным token coverage. Пустой
            массив означает, что фильтр пуст, слишком общий или содержит
            неизвестные корпусу значения.
        """
        cached = self.filter_cache.get(filter_text)
        if cached is not None:
            return cached

        stop_words = {"вид", "тип", "услуга", "услуги", "услуг"}
        analyzer = self.params_vectorizer.build_analyzer()
        tokens = sorted(
            {
                token
                for token in analyzer(filter_text)
                if token not in stop_words
            }
        )
        vocabulary = self.params_vectorizer.vocabulary_
        if not tokens or any(token not in vocabulary for token in tokens):
            result = np.empty(0, dtype=np.int64)
            self.filter_cache[filter_text] = result
            return result

        columns = [vocabulary[token] for token in tokens]
        matched_terms = np.asarray(
            self.item_params_bm25[:, columns].getnnz(axis=1)
        ).ravel()
        result = np.flatnonzero(
            (matched_terms == len(columns))
            & (self.categories == self.service_category_id)
        )
        self.filter_cache[filter_text] = result
        return result

    def _embedding_fingerprint(
        self,
        items: pd.DataFrame,
        dense_documents: list[str],
    ) -> str:
        """Рассчитать fingerprint модели и dense-текстов корпуса.

        Args:
            items: Корпус с `item_id`.
            dense_documents: Тексты для MiniLM.

        Returns:
            SHA-256 fingerprint кеша.
        """
        digest = hashlib.sha256(self.model_name_or_path.encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(items["item_id"], index=False).values.tobytes()
        )
        digest.update(
            pd.util.hash_pandas_object(
                pd.Series(dense_documents), index=False
            ).values.tobytes()
        )
        return digest.hexdigest()

    def _fit_dense(self, items: pd.DataFrame, dense_documents: list[str]) -> None:
        """Загрузить MiniLM и кеш эмбеддингов или рассчитать их заново.

        Args:
            items: Корпус объявлений.
            dense_documents: Тексты в порядке строк `items`.

        Returns:
            `None`. Создаёт `encoder` и `item_embeddings`.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_cache = self.artifacts_dir / "huggingface"
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
                self.item_embeddings = np.load(embeddings_path, mmap_mode="r")
                return

        LOGGER.info("Расчёт MiniLM эмбеддингов для %d объявлений", len(items))
        embeddings = self.encoder.encode(
            dense_documents,
            batch_size=self.embedding_batch_size,
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

    def _build_quality_prior(
        self,
        train: pd.DataFrame,
        items: pd.DataFrame,
    ) -> np.ndarray:
        """Построить сглаженный prior качества объявления.

        Args:
            train: Положительные пары для подсчёта исторической популярности.
            items: Корпус объявлений.

        Returns:
            Оценка качества каждого объявления от нуля до единицы.
        """
        microcats = items["item_microcat_id"]
        reviews = pd.to_numeric(
            items["item_rating_reviews_count"], errors="coerce"
        ).fillna(0).clip(lower=0)
        review_percentile = reviews.groupby(microcats).rank(pct=True).to_numpy(
            dtype=np.float32
        )

        rating = pd.to_numeric(items["item_rating"], errors="coerce")
        valid_rating = rating.between(0, 5)
        global_rating = float(rating[valid_rating].mean())
        rating = rating.where(valid_rating, global_rating).fillna(global_rating)
        bayesian_rating = (
            (rating * reviews + global_rating * 20.0) / (reviews + 20.0) / 5.0
        ).to_numpy(dtype=np.float32)

        click_counts = train["item_id"].value_counts()
        popularity = np.log1p(
            items["item_id"].map(click_counts).fillna(0).to_numpy(dtype=np.float32)
        )
        if popularity.max() > 0:
            popularity /= popularity.max()

        price = pd.to_numeric(items["item_price"], errors="coerce").replace(-1, np.nan)
        price_percentile = price.groupby(microcats).rank(pct=True).fillna(0.5)
        # Внутри microcat пользователи чаще выбирали нижнюю и среднюю часть цен.
        price_prior = (1.0 - price_percentile).to_numpy(dtype=np.float32)

        contact = (
            1.0
            - 0.25 * items["item_is_phone_hidden"].to_numpy(dtype=np.float32)
            - 0.40 * items["item_is_message_forbidden"].to_numpy(dtype=np.float32)
        )
        quality = (
            0.45 * review_percentile
            + 0.25 * bayesian_rating
            + 0.15 * popularity
            + 0.10 * contact
            + 0.05 * price_prior
        )
        return np.clip(quality, 0.0, 1.0).astype(np.float32)

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> "CandidateModel":
        """Построить поисковые индексы и статистику train.

        Args:
            train: Пары `запрос — выбранное объявление`.
            items: Корпус объявлений для поиска.

        Returns:
            Текущий экземпляр `CandidateModel`, готовый к `predict()`.

        Raises:
            ValueError: Если нет колонок, `item_id` повторяются или услуг меньше 50.
        """
        self._check_columns(
            train,
            {
                "search_query",
                "search_location_id",
                "search_infm_params_text",
                "item_id",
                "item_title_raw",
                "item_infm_params_text",
                "item_microcat_id",
                "item_latitude",
                "item_longitude",
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
                "item_price",
                "item_rating",
                "item_rating_reviews_count",
                "item_location_id",
                "item_latitude",
                "item_longitude",
                "item_is_phone_hidden",
                "item_is_message_forbidden",
            },
            "items",
        )
        if items["item_id"].duplicated().any():
            raise ValueError("В benchmark_items есть повторы item_id")

        self.item_ids = items["item_id"].astype(str).to_numpy()
        self.locations = items["item_location_id"].to_numpy(dtype=np.int64)
        self.categories = items["item_category_id"].to_numpy(dtype=np.int64)
        self.microcats = items["item_microcat_id"].to_numpy(dtype=np.int64)
        self.item_latitudes = pd.to_numeric(
            items["item_latitude"], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=True)
        self.item_longitudes = pd.to_numeric(
            items["item_longitude"], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=True)
        valid_coordinates = (
            (self.item_latitudes >= 40)
            & (self.item_latitudes <= 82)
            & (self.item_longitudes >= 10)
            & (self.item_longitudes <= 190)
        )
        self.item_latitudes[~valid_coordinates] = np.nan
        self.item_longitudes[~valid_coordinates] = np.nan

        self.location_positions = self._positions_by_value(self.locations)
        self.microcat_positions = self._positions_by_value(self.microcats)
        self.service_positions = np.flatnonzero(
            self.categories == self.service_category_id
        )
        if len(self.service_positions) < self.result_max_size:
            raise ValueError("В корпусе недостаточно объявлений категории услуг")

        # Берём только реально используемые train-поля: длинные тексты объявления
        # уже представлены в `items`, а их копия заметно увеличивает память.
        normalized_train = train[
            [
                "search_query",
                "search_location_id",
                "search_infm_params_text",
                "item_id",
                "item_microcat_id",
                "item_latitude",
                "item_longitude",
            ]
        ].copy()
        normalized_train["normalized_query"] = Normalizer.lexical(
            normalized_train["search_query"]
        )
        normalized_train["normalized_filter"] = Normalizer.lexical(
            normalized_train["search_infm_params_text"]
        )
        self._fit_query_knowledge(normalized_train)
        self._fit_query_profiles(normalized_train, train)
        self._fit_geography(normalized_train)

        title = Normalizer.lexical(items["item_title_raw"])
        description = Normalizer.lexical(items["item_description_raw"])
        params = Normalizer.lexical(items["item_infm_params_text"])

        # Поля индексируются отдельно: длинное описание не должно уменьшать вес
        # точного совпадения в коротком заголовке.
        LOGGER.info("Построение fielded BM25 индексов")
        self.title_vectorizer, self.item_title_bm25 = self._build_bm25(
            title.tolist(), ngram_range=(1, 2), max_features=100_000
        )
        self.description_vectorizer, self.item_description_bm25 = self._build_bm25(
            description.tolist(), ngram_range=(1, 1), max_features=120_000
        )
        self.params_vectorizer, self.item_params_bm25 = self._build_bm25(
            params.tolist(), ngram_range=(1, 1), max_features=100_000
        )
        self.filter_cache: dict[str, np.ndarray] = {}

        # Char TF-IDF оставляем в проверенном компактном виде baseline. Полный
        # params уже представлен отдельным BM25-каналом, поэтому повторно
        # индексировать его character n-граммами нет смысла.
        del description, params
        char_params = Normalizer.lexical(items["item_infm_params_text"], limit=500)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.item_chars = self.char_vectorizer.fit_transform(
            [
                f"{item_title} {item_params}"
                for item_title, item_params in zip(title, char_params)
            ]
        )
        del title, char_params

        # Dense-текст корпуса совпадает с V3, поэтому существующий кеш
        # MiniLM переиспользуется и повторный расчёт эмбеддингов не требуется.
        dense_title = Normalizer.dense(items["item_title_raw"])
        dense_params = Normalizer.dense(items["item_infm_params_text"], limit=500)
        dense_documents = [
            ". ".join(part for part in parts if part)
            for parts in zip(dense_title, dense_params)
        ]
        self._fit_dense(items, dense_documents)

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
        self.quality = self._build_quality_prior(normalized_train, items)

        self.is_fitted = True
        return self

    def _apply_microcat_boost(
        self,
        fused_scores: np.ndarray,
        predicted_microcats: list[tuple[int, float]],
    ) -> None:
        """Мягко усилить объявления вероятных микрокатегорий.

        Args:
            fused_scores: RRF-оценки, изменяемые на месте.
            predicted_microcats: Пары `(microcat, probability)`.

        Returns:
            `None`.
        """
        for microcat, probability in predicted_microcats:
            positions = self.microcat_positions.get(microcat)
            if positions is not None:
                # Мягкий boost сохраняет шанс исправить ошибку microcat-модели.
                fused_scores[positions] *= 1.0 + 0.70 * probability

    def _geo_quota(
        self,
        location: int,
        predicted_microcats: list[tuple[int, float]],
    ) -> int:
        """Рассчитать адаптивную квоту объявлений в ближайшем радиусе.

        Args:
            location: `search_location_id` запроса.
            predicted_microcats: Вероятные микрокатегории запроса.

        Returns:
            Число мест, зарезервированных для geo-пула.
        """
        location_rate = float(
            self.location_near_rate.get(location, self.global_near_rate)
        )
        if predicted_microcats:
            microcat_rate = sum(
                probability
                * float(
                    self.microcat_near_rate.get(microcat, self.global_near_rate)
                )
                for microcat, probability in predicted_microcats
            )
        else:
            microcat_rate = self.global_near_rate

        near_probability = 0.65 * location_rate + 0.35 * microcat_rate
        quota = int(round(self.result_max_size * near_probability))
        return int(np.clip(quota, self.geo_min_quota, self.geo_max_quota))

    def _select_with_quotas(
        self,
        fused_scores: np.ndarray,
        normalized_query: str,
        location: int,
        predicted_microcats: list[tuple[int, float]],
        geo_positions: np.ndarray,
        filter_positions: np.ndarray,
        profile_positions: np.ndarray,
    ) -> list[int]:
        """Выбрать 50 кандидатов с geo-, filter- и profile-покрытием.

        Args:
            fused_scores: Общие оценки всех объявлений.
            normalized_query: Нормализованный запрос для точной истории.
            location: `search_location_id`.
            predicted_microcats: Вероятные микрокатегории запроса.
            geo_positions: Сервисные объявления в адаптивном радиусе.
            filter_positions: Объявления с полным token coverage фильтра.
            profile_positions: Лучшие кандидаты query-profile канала.

        Returns:
            Ровно `result_max_size` уникальных позиций корпуса.
        """
        history = self.history_map.get((normalized_query, location), [])
        selected = [
            position
            for position in history
            if self.categories[position] == self.service_category_id
        ][: self.history_limit]
        selected = list(dict.fromkeys(selected))
        selected_set = set(selected)
        geo_set = set(int(position) for position in geo_positions)
        filter_set = set(int(position) for position in filter_positions)
        profile_set = set(int(position) for position in profile_positions)

        def add_count(pool: np.ndarray, count: int) -> None:
            """Добавить заданное число лучших ещё не выбранных позиций.

            Args:
                pool: Допустимые позиции объявлений.
                count: Максимум новых позиций.

            Returns:
                `None`. Список `selected` изменяется на месте.
            """
            if count <= 0 or len(pool) == 0:
                return
            ranked = pool[
                self._top_indices(
                    fused_scores[pool],
                    min(len(pool), max(self.channel_candidates, 250)),
                )
            ]
            added = 0
            for position in ranked:
                position = int(position)
                if position in selected_set:
                    continue
                selected.append(position)
                selected_set.add(position)
                added += 1
                if added >= count or len(selected) >= self.result_max_size:
                    break

        # Новые каналы сначала получают места внутри geo-пула: так точный
        # фильтр или click-profile не вытесняет локальных исполнителей.
        filter_geo = np.intersect1d(
            geo_positions, filter_positions, assume_unique=True
        )
        profile_geo = np.intersect1d(
            geo_positions, profile_positions, assume_unique=True
        )
        current_filter = sum(position in filter_set for position in selected)
        add_count(
            filter_geo,
            max(0, min(self.filter_quota, 7) - current_filter),
        )
        current_profile = sum(position in profile_set for position in selected)
        add_count(
            profile_geo,
            max(0, min(self.profile_quota, 4) - current_profile),
        )

        geo_target = min(
            self._geo_quota(location, predicted_microcats), len(geo_positions)
        )
        current_geo = sum(position in geo_set for position in selected)
        remaining_geo = max(0, geo_target - current_geo)

        # Сначала распределяем geo-квоту между вероятными микрокатегориями.
        # Минимум одно место не позволяет редкому второму интенту исчезнуть.
        for microcat, probability in predicted_microcats:
            if remaining_geo <= 0 or len(selected) >= self.result_max_size:
                break
            microcat_positions = self.microcat_positions.get(
                microcat, np.empty(0, dtype=np.int64)
            )
            pool = np.intersect1d(
                geo_positions, microcat_positions, assume_unique=True
            )
            requested = min(
                remaining_geo,
                max(1, int(round(geo_target * probability))),
            )
            before = len(selected)
            add_count(pool, requested)
            added = len(selected) - before
            remaining_geo -= added

        if remaining_geo > 0:
            add_count(geo_positions, remaining_geo)

        # Добираем глобальную часть filter/profile квот, но сохраняем три
        # fallback-места для независимого текстового поиска.
        available = max(0, self.result_max_size - len(selected) - 3)
        current_filter = sum(position in filter_set for position in selected)
        add_count(
            filter_positions,
            min(available, max(0, self.filter_quota - current_filter)),
        )
        available = max(0, self.result_max_size - len(selected) - 3)
        current_profile = sum(position in profile_set for position in selected)
        add_count(
            profile_positions,
            min(available, max(0, self.profile_quota - current_profile)),
        )

        # После geo-пула оставляем минимум три места глобальному текстовому
        # поиску: это страховка для удалённых услуг и ошибки microcat-прогноза.
        intent_budget = max(0, self.result_max_size - len(selected) - 3)
        if intent_budget > 0 and predicted_microcats:
            intent_positions = np.concatenate(
                [
                    self.microcat_positions.get(
                        microcat, np.empty(0, dtype=np.int64)
                    )
                    for microcat, _ in predicted_microcats
                ]
            )
            intent_positions = intent_positions[
                self.categories[intent_positions] == self.service_category_id
            ]
            add_count(np.unique(intent_positions), intent_budget)

        add_count(
            self.service_positions,
            self.result_max_size - len(selected),
        )
        return selected[: self.result_max_size]

    @staticmethod
    def _binary_query_matrix(
        vectorizer: CountVectorizer,
        texts: list[str],
    ) -> csr_matrix:
        """Преобразовать запросы в бинарную матрицу для BM25-документов.

        Args:
            vectorizer: Обученный словарь соответствующего поля.
            texts: Нормализованные тексты запросов.

        Returns:
            Бинарная CSR-матрица запросов.
        """
        matrix = vectorizer.transform(texts).tocsr()
        matrix.data.fill(1.0)
        return matrix

    def predict(self, queries: pd.DataFrame) -> list[list[str]]:
        """Найти 50 кандидатов для каждого benchmark-запроса.

        Args:
            queries: Benchmark-запросы в исходном порядке.

        Returns:
            Список списков `item_id`; порядок совпадает с `queries`.

        Raises:
            RuntimeError: Если `fit()` ещё не вызван или top-50 некорректен.
            ValueError: Если в запросах отсутствуют обязательные колонки.
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
        filters = Normalizer.lexical(queries["search_infm_params_text"])
        predicted_microcats, profile_queries = self._predict_microcats(
            query, filters
        )
        query_texts = query.tolist()
        params_queries = [
            f"{query_text} {query_filter}".strip()
            for query_text, query_filter in zip(query, filters)
        ]
        char_queries = [
            f"{query_text} {query_text} {query_filter}".strip()
            for query_text, query_filter in zip(query, filters)
        ]

        title_query = self._binary_query_matrix(
            self.title_vectorizer, query_texts
        )
        description_query = self._binary_query_matrix(
            self.description_vectorizer, query_texts
        )
        params_query = self._binary_query_matrix(
            self.params_vectorizer, params_queries
        )
        char_query = self.char_vectorizer.transform(char_queries)

        dense_query = Normalizer.dense(queries["search_query"])
        dense_filters = Normalizer.dense(queries["search_infm_params_text"], limit=500)
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
        profile_embeddings = self.encoder.encode(
            profile_queries,
            batch_size=self.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        has_profile = np.asarray([bool(text) for text in profile_queries])
        profile_embeddings[~has_profile] = 0.0

        predictions: list[list[str]] = []
        inference_batch_size = 16

        for start in range(0, len(queries), inference_batch_size):
            stop = min(start + inference_batch_size, len(queries))
            title_scores = (
                title_query[start:stop] @ self.item_title_bm25.T
            ).toarray()
            description_scores = (
                description_query[start:stop] @ self.item_description_bm25.T
            ).toarray()
            params_scores = (
                params_query[start:stop] @ self.item_params_bm25.T
            ).toarray()
            char_scores = (
                char_query[start:stop] @ self.item_chars.T
            ).toarray()
            dense_scores = query_embeddings[start:stop] @ self.item_embeddings.T
            profile_scores = (
                profile_embeddings[start:stop] @ self.item_embeddings.T
            )

            for offset in range(stop - start):
                query_index = start + offset
                fused = self.quality_weight * self.quality.copy()
                channel_scores: tuple[tuple[np.ndarray, float], ...] = (
                    (title_scores[offset], 1.20),
                    (description_scores[offset], 0.90),
                    (params_scores[offset], 1.00),
                    (char_scores[offset], 0.80),
                    (dense_scores[offset], 1.20),
                )
                if has_profile[query_index]:
                    channel_scores += ((profile_scores[offset], 1.00),)

                for scores, weight in channel_scores:
                    ranked = self._top_indices(scores, self.channel_candidates)
                    self._add_rrf(fused, ranked, weight)

                row = queries.iloc[query_index]
                location = int(row["search_location_id"])
                filter_positions = self._filter_candidates(
                    str(filters[query_index])
                )
                if has_profile[query_index]:
                    profile_positions = self._top_indices(
                        profile_scores[offset], self.channel_candidates
                    )
                    profile_positions = profile_positions[
                        self.categories[profile_positions]
                        == self.service_category_id
                    ]
                else:
                    profile_positions = np.empty(0, dtype=np.int64)

                # Внутри exact-filter пула повторно объединяем каналы. Иначе
                # при широком фильтре квота выбирала бы объявления почти только
                # по популярности, а не по тексту запроса.
                for scores, weight in channel_scores:
                    filter_rank = self._top_indices(
                        scores[filter_positions],
                        min(self.channel_candidates, len(filter_positions)),
                    )
                    self._add_rrf(
                        fused,
                        filter_positions[filter_rank],
                        0.40 * weight,
                    )

                geo_positions = np.empty(0, dtype=np.int64)
                if not bool(row["search_is_delivery_search"]):
                    geo_positions, proximity = self._geo_candidates(location)
                    for scores, weight in channel_scores:
                        geo_rank = self._top_indices(
                            scores[geo_positions],
                            min(self.channel_candidates, len(geo_positions)),
                        )
                        self._add_rrf(
                            fused,
                            geo_positions[geo_rank],
                            0.70 * weight,
                        )
                    fused[geo_positions] += self.geo_weight * proximity

                self._apply_microcat_boost(
                    fused, predicted_microcats[query_index]
                )
                selected = self._select_with_quotas(
                    fused,
                    normalized_query=str(query[query_index]),
                    location=location,
                    predicted_microcats=predicted_microcats[query_index],
                    geo_positions=geo_positions,
                    filter_positions=filter_positions,
                    profile_positions=profile_positions,
                )

                if (
                    len(selected) != self.result_max_size
                    or len(selected) != len(set(selected))
                    or np.any(
                        self.categories[selected] != self.service_category_id
                    )
                ):
                    raise RuntimeError(
                        "Не удалось сформировать 50 уникальных объявлений услуг"
                    )
                predictions.append(self.item_ids[selected].tolist())

            LOGGER.info("Обработано запросов: %d/%d", stop, len(queries))

        return predictions

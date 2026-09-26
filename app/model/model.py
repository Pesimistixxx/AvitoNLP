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
    """Готовит текст отдельно для лексического и нейросетевого поиска."""

    @staticmethod
    def lexical(values: pd.Series, limit: int | None = None) -> np.ndarray:
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
        text = values.fillna("").astype(str)
        if limit is not None:
            text = text.str.slice(0, limit)
        return text.str.replace(r"\s+", " ", regex=True).str.strip().to_numpy()


class CandidateModel:
    """Гибридный поиск BM25 + char TF-IDF + BGE-M3 с объединением через RRF."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        artifacts_dir: str | Path = "artifacts",
        embedding_batch_size: int = 32,
        channel_candidates: int = 200,
        result_max_size: int = 50,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.artifacts_dir = Path(artifacts_dir)
        self.embedding_batch_size = embedding_batch_size
        self.channel_candidates = channel_candidates
        self.result_max_size = result_max_size
        self.is_fitted = False

    @staticmethod
    def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
        """Вернуть индексы максимальных значений без полной сортировки массива."""
        count = min(count, len(scores))
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        positions = np.argpartition(scores, len(scores) - count)[-count:]
        return positions[np.argsort(scores[positions])[::-1]]

    @staticmethod
    def _positions_by_value(values: np.ndarray) -> dict[int, np.ndarray]:
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
        """Добавить ранжированный список в Reciprocal Rank Fusion."""
        ranks = np.arange(1, len(ranked_indices) + 1, dtype=np.float32)
        fused_scores[ranked_indices] += weight / (rrf_k + ranks)

    @staticmethod
    def _check_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"В {name} нет колонок: {sorted(missing)}")

    def _fit_bm25(self, documents: list[str]) -> None:
        """Построить разреженную матрицу весов BM25 без отдельной зависимости."""
        self.bm25_vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=200_000,
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

    def _embedding_fingerprint(
        self,
        items: pd.DataFrame,
        dense_documents: list[str],
    ) -> str:
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
        """Загрузить BGE-M3 и получить нормализованные эмбеддинги корпуса."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_cache = self.artifacts_dir / "huggingface"
        self.encoder = SentenceTransformer(
            self.model_name_or_path,
            cache_folder=str(model_cache),
        )

        model_key = hashlib.sha1(
            self.model_name_or_path.encode("utf-8")
        ).hexdigest()[:10]
        embeddings_path = self.artifacts_dir / f"item_embeddings_{model_key}.npy"
        metadata_path = embeddings_path.with_suffix(".json")
        fingerprint = self._embedding_fingerprint(items, dense_documents)

        if embeddings_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                LOGGER.info("Загрузка кеша BGE-M3: %s", embeddings_path)
                self.item_embeddings = np.load(embeddings_path, mmap_mode="r")
                return

        LOGGER.info("Расчёт BGE-M3 эмбеддингов для %d объявлений", len(items))
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

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> "CandidateModel":
        """Построить три поисковых индекса и статистику прошлых кликов."""
        self._check_columns(
            train,
            {"search_query", "search_location_id", "item_id"},
            "train",
        )
        self._check_columns(
            items,
            {
                "item_id",
                "item_title_raw",
                "item_description_raw",
                "item_infm_params_text",
                "item_location_id",
                "item_rating_reviews_count",
            },
            "items",
        )
        if items["item_id"].duplicated().any():
            raise ValueError("В benchmark_items есть повторы item_id")

        self.item_ids = items["item_id"].astype(str).to_numpy()
        self.locations = items["item_location_id"].to_numpy(dtype=np.int64)
        self.location_positions = self._positions_by_value(self.locations)

        title = Normalizer.lexical(items["item_title_raw"])
        params = Normalizer.lexical(items["item_infm_params_text"], limit=500)
        description = Normalizer.lexical(items["item_description_raw"], limit=800)
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
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.item_chars = self.char_vectorizer.fit_transform(
            [f"{item_title} {item_params}" for item_title, item_params in zip(title, params)]
        )

        dense_title = Normalizer.dense(items["item_title_raw"])
        dense_params = Normalizer.dense(items["item_infm_params_text"], limit=500)
        dense_description = Normalizer.dense(items["item_description_raw"], limit=800)
        dense_documents = [
            ". ".join(part for part in parts if part)
            for parts in zip(dense_title, dense_params, dense_description)
        ]
        self._fit_dense(items, dense_documents)

        # Точные исторические совпадения запроса и локации получают первые места.
        item_position = {
            item_id: position for position, item_id in enumerate(self.item_ids)
        }
        history = train[train["item_id"].isin(item_position)].copy()
        history["normalized_query"] = Normalizer.lexical(history["search_query"])
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

        reviews = items["item_rating_reviews_count"].fillna(0).clip(lower=0)
        self.quality = np.log1p(reviews.to_numpy(dtype=np.float32))
        if self.quality.max() > 0:
            self.quality /= self.quality.max()

        self.is_fitted = True
        return self

    def predict(self, queries: pd.DataFrame) -> list[list[str]]:
        """Вернуть ровно result_size уникальных item_id для каждого запроса."""
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
        filters = Normalizer.lexical(queries["search_infm_params_text"], limit=500)
        lexical_queries = [
            f"{query_text} {query_text} {query_filters}"
            for query_text, query_filters in zip(query, filters)
        ]

        query_bm25 = self.bm25_vectorizer.transform(lexical_queries).tocsr()
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
            bm25_scores = (query_bm25[start:stop] @ self.item_bm25.T).toarray()
            char_scores = (query_chars[start:stop] @ self.item_chars.T).toarray()
            dense_scores = query_embeddings[start:stop] @ self.item_embeddings.T

            for offset in range(stop - start):
                query_index = start + offset
                fused = 1e-8 * self.quality.copy()
                channel_scores = (
                    (bm25_scores[offset], 1.0),
                    (char_scores[offset], 0.8),
                    (dense_scores[offset], 1.3),
                )

                # Глобальные списки сохраняют полноту для удалённых исполнителей.
                for scores, weight in channel_scores:
                    ranked = self._top_indices(scores, self.channel_candidates)
                    self._add_rrf(fused, ranked, weight)

                # Локальные списки усиливают объявления из локации запроса.
                row = queries.iloc[query_index]
                location = int(row["search_location_id"])
                local_positions = self.location_positions.get(
                    location, np.empty(0, dtype=np.int64)
                )
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

                history_key = (query[query_index], location)
                selected = list(
                    dict.fromkeys(self.history_map.get(history_key, [])[:15])
                )
                selected_set = set(selected)

                for position in self._top_indices(
                    fused, min(len(fused), self.result_max_size + len(selected))
                ):
                    position = int(position)
                    if position not in selected_set:
                        selected.append(position)
                        selected_set.add(position)
                    if len(selected) == self.result_max_size:
                        break

                predictions.append(self.item_ids[selected].tolist())

            LOGGER.info("Обработано запросов: %d/%d", stop, len(queries))

        return predictions

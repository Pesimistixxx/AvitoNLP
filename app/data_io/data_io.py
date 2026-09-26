from pathlib import Path
import pandas as pd


def load_data(data_dir: Path) -> list[pd.DataFrame]:
    """Загрузить train, benchmark-запросы и корпус объявлений."""
    train = pd.read_parquet(data_dir / "train.parquet")
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")
    return [train, queries, items]


def save_data(data: pd.DataFrame, output_dir: Path) -> None:
    """Сохранить итоговый DataFrame в output/answer.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "answer.csv", index=False, encoding="utf-8")
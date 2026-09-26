FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/artifacts/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    "numpy>=2,<3" \
    "pandas>=2.2,<4" \
    "pyarrow>=17,<24" \
    "scipy>=1.13,<2" \
    "scikit-learn>=1.5,<2" \
    "sentence-transformers>=5,<6"

COPY app ./app
COPY config.py main.py ./

RUN mkdir -p data output artifacts

CMD ["python", "-u", "main.py"]
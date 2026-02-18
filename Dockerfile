FROM dolfinx/dolfinx:stable

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    openmpi-bin libopenmpi-dev \
    python3-mpi4py \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

COPY backend ./backend

CMD ["bash", "-lc", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

FROM dolfinx/dolfinx:stable

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    openmpi-bin libopenmpi-dev \
    python3-mpi4py \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm==3.7.1

COPY backend ./backend

CMD ["bash", "-lc", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
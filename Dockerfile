FROM dolfinx/dolfinx:stable

WORKDIR /opt/app

# System deps sometimes needed for gmsh/pyvista/meshio usage
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Copy your app (adjust if your repo layout differs)
COPY backend ./backend

# Render provides PORT
CMD ["bash", "-lc", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

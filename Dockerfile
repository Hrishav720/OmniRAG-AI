# ---------- OmniRAG backend (FastAPI) ----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    OMNIRAG_DATA_DIR=/data

# Libraries needed by docling / opencv at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# Install a matching CPU-only PyTorch / TorchVision pair
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-backend.txt .
RUN pip install -r requirements-backend.txt

# Verify that dependency installation did not break compatibility
RUN python -c "import torch, torchvision; print('Torch:', torch.__version__); print('TorchVision:', torchvision.__version__); from sentence_transformers import CrossEncoder; print('CrossEncoder import OK')"


# Bake the models into the image so the first request isn't a long download
RUN python -c "from sentence_transformers import CrossEncoder; \
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "from docling.utils.model_downloader import download_models; download_models()" \
    || echo "docling model pre-download skipped; they will download on first PDF"

COPY main.py api.py ./

# Run as a non-root user; /data holds the FAISS indexes
RUN useradd -m appuser && mkdir -p /data && chown -R appuser /data /opt/hf
USER appuser
VOLUME /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','8000'))" || exit 1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

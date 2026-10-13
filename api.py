from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shutil
import tempfile
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, List

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from main import (
    DOCLING_EXTENSIONS,
    EVAL_METRICS,
    GRAPH_RECURSION_LIMIT,
    _get_docling_converter,
    ensure_knowledge_collection,
    ingest_files_into_collection,
    list_knowledge_collections,
    embeddings,
    eval_answer,
    graph,
    reranker,
    route_after_critic,
    route_after_load,
    route_after_sufficiency,
    route_cache,
)

load_dotenv()

APP_TITLE = "OmniRAG API"
APP_VERSION = "2.1.0"

# Security controls are environment-driven; no credentials are hard-coded.
API_KEY = os.getenv("OMNIRAG_API_KEY", "").strip()
MAX_UPLOAD_FILE_BYTES = int(os.getenv("OMNIRAG_MAX_UPLOAD_FILE_BYTES", str(25 * 1024 * 1024)))
MAX_UPLOAD_TOTAL_BYTES = int(os.getenv("OMNIRAG_MAX_UPLOAD_TOTAL_BYTES", str(100 * 1024 * 1024)))
MAX_FILES_PER_REQUEST = int(os.getenv("OMNIRAG_MAX_FILES_PER_REQUEST", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("OMNIRAG_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("OMNIRAG_RATE_LIMIT_MAX_REQUESTS", "20"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "OMNIRAG_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)

# Keep this in one place so the API and UI can expose the same supported set.
SUPPORTED_EXTENSIONS = sorted({".csv", *DOCLING_EXTENSIONS})

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # stop reverse proxies (nginx) from buffering the stream
}


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(default_factory=list)


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


# ── Warm-up: pay model-loading cost at startup, not on the first question ────

def _warmup() -> None:
    try:
        embeddings.embed_query("warmup")
        reranker.predict([("warmup", "warmup")])
        converter = _get_docling_converter()
        try:
            from docling.datamodel.base_models import InputFormat

            converter.initialize_pipeline(InputFormat.PDF)
        except Exception:
            pass  # older docling versions: model loads lazily on first PDF
    except Exception as exc:  # warm-up must never stop the API from starting
        print(f"[omnirag] warm-up skipped: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="API for the OmniRAG agentic RAG pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect data-bearing endpoints. Set OMNIRAG_API_KEY in the API's .env."""
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured. Set OMNIRAG_API_KEY in the environment.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


@app.middleware("http")
async def rate_limit_requests(request: Request, call_next):
    """Simple per-client in-process rate limiter for expensive/write API routes."""
    protected_paths = ("/query", "/evaluate", "/collections")
    if request.method in {"GET", "POST", "PUT", "DELETE"} and request.url.path.startswith(protected_paths):
        # Reject obviously oversized multipart requests before the upload parser
        # spools the body; _save_uploads below also enforces per-file and total
        # limits while copying individual files.
        content_length = request.headers.get("content-length")
        if request.method == "POST" and request.url.path.startswith("/query") and content_length:
            try:
                max_body_bytes = MAX_UPLOAD_TOTAL_BYTES + 2 * 1024 * 1024
                if int(content_length) > max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body exceeds the {MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MB total upload limit."},
                    )
            except ValueError:
                pass
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _RATE_LIMIT_LOCK:
            bucket = _RATE_LIMIT_BUCKETS[client]
            cutoff = now - RATE_LIMIT_WINDOW_SECONDS
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
                retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please retry later."},
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)
    return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_upload(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. Supported: {supported}",
        )
    return suffix


def _save_uploads(files: List[UploadFile]) -> tuple[str, list[str], list[str]]:
    """Save uploads while enforcing per-file, total-size and file-count limits."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files. Maximum allowed per request is {MAX_FILES_PER_REQUEST}.",
        )
    workdir = tempfile.mkdtemp(prefix="omnirag_")
    paths: list[str] = []
    names: list[str] = []
    total_bytes = 0
    try:
        for i, upload in enumerate(files):
            _validate_upload(upload.filename)
            original = Path(upload.filename).name
            safe = re.sub(r"[^\w.\- ]", "_", original) or f"file_{i}"
            target = Path(workdir) / safe
            if target.exists():
                target = Path(workdir) / f"{i}_{safe}"
            file_bytes = 0
            with open(target, "wb") as out:
                while True:
                    block = upload.file.read(1024 * 1024)
                    if not block:
                        break
                    file_bytes += len(block)
                    total_bytes += len(block)
                    if file_bytes > MAX_UPLOAD_FILE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{original}' exceeds the {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB per-file limit.",
                        )
                    if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Combined upload exceeds the {MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MB total limit.",
                        )
                    out.write(block)
            paths.append(str(target))
            names.append(original)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return workdir, paths, names


def _clean_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return question


def _event(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str) + "\n"


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str)
            else part.get("text", "") if isinstance(part, dict) and part.get("type", "text") == "text"
            else ""
            for part in content
        )
    return ""


# Which node runs after which (so the UI can show "what's happening now").
_STATIC_NEXT = {
    "load_cache_node": "retrieval_node",
    "chunking_node": "vectorstore_node",
    "vectorstore_node": "save_cache_node",
    "save_cache_node": "retrieval_node",
    "retrieval_node": "rerank_node",
    "rerank_node": "filter_chunks_node",
    "filter_chunks_node": "check_context_sufficiency_node",
    "increase_attempts_node": "retrieval_node",
    "generation_node": "critic_node",
    "increase_critic_attempts": "retrieval_node",
    "answer_not_found": None,
}
_SUFFICIENCY_NEXT = {
    "generate_answer": "generation_node",
    "increase_attempts_node": "increase_attempts_node",
}
_CRITIC_NEXT = {
    "finish": None,
    "increase_critic_attempts": "increase_critic_attempts",
    "answer_not_found": "answer_not_found",
}


def _next_node(node: str, state: dict) -> str | None:
    """Reuse the backend's own routers so the prediction can't drift from the graph."""
    try:
        if node == "compute_hash_node":
            return route_cache(state)
        if node == "load_documents_node":
            return route_after_load(state)
        if node == "check_context_sufficiency_node":
            return _SUFFICIENCY_NEXT.get(route_after_sufficiency(state))
        if node == "critic_node":
            return _CRITIC_NEXT.get(route_after_critic(state))
    except Exception:
        return None
    return _STATIC_NEXT.get(node)


def _step_detail(node: str, state: dict) -> str:
    n = lambda key: len(state.get(key) or [])  # noqa: E731
    if node == "load_cache_node":
        return "cache hit"
    if node == "load_documents_node":
        text = f"{n('documents')} document(s) loaded"
        if state.get("skipped_files"):
            text += f", {n('skipped_files')} skipped"
        return text
    if node == "chunking_node":
        return f"{n('chunks')} chunks"
    if node == "retrieval_node":
        return f"{n('retrieved_docs')} candidates"
    if node == "rerank_node":
        return f"top {n('reranked_docs')} kept"
    if node == "filter_chunks_node":
        return f"{n('filtered_docs')} of {n('reranked_docs')} relevant"
    if node == "check_context_sufficiency_node":
        ok = str(state.get("sufficient", "")).lower() == "yes"
        return "enough context" if ok else "needs more context"
    if node == "increase_attempts_node":
        return f"attempt {state.get('retrieval_attempts', 0)}"
    if node == "critic_node":
        verdict = str(state.get("critic_verdict", "")).lower()
        return f"{verdict} ({state.get('critic_reason', '')})"
    if node == "increase_critic_attempts":
        return f"attempt {state.get('critic_attempts', 0)}"
    return ""


def _run_pipeline(
    file_paths: list[str], file_names: list[str], question: str,
    collection_name: str | None = None,
) -> Iterator[dict]:
    """Run the existing LangGraph pipeline against a named persistent collection."""
    started = time.perf_counter()
    final: dict = {"file_paths": file_paths, "question": question}
    if collection_name:
        final["collection_name"] = collection_name
    executed: list[str] = []
    last_t = 0.0
    generation_done = False

    yield {"type": "start", "file_names": file_names}

    for mode, payload in graph.stream(
        dict(final),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            # generation_node publishes native Gemini chunks through
            # get_stream_writer(), so forward only those events to the UI.
            if not isinstance(payload, dict) or payload.get("type") != "token":
                continue
            if generation_done:  # critic sent us back: a fresh draft is starting
                generation_done = False
                yield {"type": "reset"}
            yield payload
            continue

        for node, update in payload.items():
            if isinstance(update, dict):
                final.update(update)
            now = time.perf_counter() - started
            executed.append(node)
            if node == "generation_node":
                generation_done = True
            yield {
                "type": "step",
                "node": node,
                "detail": _step_detail(node, final),
                "next": _next_node(node, final),
                "t": round(now, 2),
                "dt": round(now - last_t, 2),
            }
            last_t = now

    docs = final.get("filtered_docs") or []
    sources = sorted({d.metadata.get("source") for d in docs if d.metadata.get("source")})
    source_chunks = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        pages = metadata.get("page_numbers") or []
        if not pages and metadata.get("page_number") is not None:
            pages = [metadata["page_number"]]
        source_chunks.append({
            "id": f"S{index}",
            "citation": f"[S{index}]",
            "source": metadata.get("source") or "Unknown source",
            "page_numbers": pages,
            "headings": metadata.get("headings") or [],
            "chunk_id": metadata.get("chunk_id"),
            "content": doc.page_content,
        })

    yield {
        "type": "final",
        "success": True,
        "question": question,
        "file_names": file_names,
        "collection_name": collection_name or "",
        "answer": final.get("answer", "No answer returned."),
        "critic_verdict": final.get("critic_verdict", "unknown"),
        "critic_reason": final.get("critic_reason", ""),
        "retrieval_attempts": final.get("retrieval_attempts", 0),
        "critic_attempts": final.get("critic_attempts", 0),
        "final_verdict": final.get("final_verdict", ""),
        "from_cache": final.get("from_cache", False),
        "skipped_files": final.get("skipped_files", []),
        "executed_steps": executed,
        # Keep these strings unchanged for the existing DeepEval request flow.
        "evaluation_contexts": [d.page_content for d in docs],
        # Structured records power citations and the Sources UI.
        "source_chunks": source_chunks,
        "sources": sources,
        "elapsed_total": round(time.perf_counter() - started, 3),
    }


# ── System routes ─────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "query": "/query",
        "query_stream": "/query/stream",
        "evaluate": "/evaluate",
        "evaluate_stream": "/evaluate/stream",
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "service": APP_TITLE,
        "llm_provider": "OpenAI",
        "vector_store": "FAISS",
    }


@app.get("/supported-formats", tags=["System"])
def supported_formats():
    return {
        "count": len(SUPPORTED_EXTENSIONS),
        "extensions": [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    }


# ── Persistent collections ────────────────────────────────────────────────────

@app.get("/collections", tags=["Collections"], dependencies=[Depends(require_api_key)])
def get_collections():
    return {"collections": list_knowledge_collections()}


@app.post("/collections", tags=["Collections"], dependencies=[Depends(require_api_key)])
def create_collection(request: CreateCollectionRequest):
    try:
        return {"success": True, "collection": ensure_knowledge_collection(request.name)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── RAG ───────────────────────────────────────────────────────────────────────

@app.post("/query/stream", tags=["RAG"], dependencies=[Depends(require_api_key)])
def query_stream(
    question: str = Form(...),
    collection_name: str = Form("default"),
    files: List[UploadFile] = File(default=[]),
):
    """Incrementally ingest supplied files, then stream a query over the collection."""
    question = _clean_question(question)
    workdir, paths, names = _save_uploads(files)
    try:
        ingestion = ingest_files_into_collection(collection_name, paths, names)
    except ValueError as exc:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        raise

    def generate():
        try:
            for event in _run_pipeline([], names, question, collection_name=collection_name):
                if event.get("type") == "final":
                    event["collection_ingestion"] = ingestion
                yield _event(event)
        except Exception as exc:
            yield _event({
                "type": "error",
                "detail": f"RAG pipeline failed: {type(exc).__name__}: {exc}",
            })
        finally:
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    return StreamingResponse(generate(), media_type="application/x-ndjson", headers=STREAM_HEADERS)


@app.post("/query", tags=["RAG"], dependencies=[Depends(require_api_key)])
def query(
    question: str = Form(...),
    collection_name: str = Form("default"),
    files: List[UploadFile] = File(default=[]),
):
    """Same collection workflow, returning one JSON response at the end."""
    question = _clean_question(question)
    workdir, paths, names = _save_uploads(files)
    try:
        ingestion = ingest_files_into_collection(collection_name, paths, names)
        final = None
        for event in _run_pipeline([], names, question, collection_name=collection_name):
            if event["type"] == "final":
                final = event
        if final is None:
            raise RuntimeError("pipeline produced no result")
        final.pop("type", None)
        final["collection_ingestion"] = ingestion
        return final
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


# ── Evaluation (DeepEval, user-triggered) ─────────────────────────────────────

def _validated_eval_inputs(request: EvaluateRequest) -> tuple[str, str, list[str]]:
    question = request.question.strip()
    answer = request.answer.strip()
    contexts = [c for c in request.contexts if isinstance(c, str) and c.strip()]

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")
    if not contexts:
        raise HTTPException(status_code=400, detail="No retrieved context available for evaluation.")
    return question, answer, contexts


def _eval_state(question: str, answer: str, contexts: list[str]) -> dict:
    # eval_answer() reads the same fields the graph state carries.
    return {
        "question": question,
        "answer": answer,
        "filtered_docs": [Document(page_content=c) for c in contexts],
    }


@app.post("/evaluate/stream", tags=["Evaluation"], dependencies=[Depends(require_api_key)])
def evaluate_stream(request: EvaluateRequest):
    """Evaluate an answer; each metric is streamed the moment it finishes."""
    question, answer, contexts = _validated_eval_inputs(request)
    events: "queue.Queue[dict | None]" = queue.Queue()
    started = time.perf_counter()

    def worker():
        try:
            result = eval_answer(
                _eval_state(question, answer, contexts),
                progress_cb=lambda name, res: events.put({
                    "type": "metric",
                    "name": name,
                    "result": res,
                    "t": round(time.perf_counter() - started, 2),
                }),
            )
            events.put({
                "type": "final",
                "evaluation": result["evaluation"],
                "elapsed_total": round(time.perf_counter() - started, 3),
            })
        except Exception as exc:
            events.put({
                "type": "error",
                "detail": f"Evaluation failed: {type(exc).__name__}: {exc}",
            })
        finally:
            events.put(None)

    def generate():
        threading.Thread(target=worker, daemon=True).start()
        yield _event({"type": "start", "metrics": EVAL_METRICS})
        while True:
            item = events.get()
            if item is None:
                break
            yield _event(item)

    return StreamingResponse(generate(), media_type="application/x-ndjson", headers=STREAM_HEADERS)


@app.post("/evaluate", tags=["Evaluation"], dependencies=[Depends(require_api_key)])
def evaluate(request: EvaluateRequest):
    """Evaluate an already generated answer (all metrics run in parallel)."""
    started = time.perf_counter()
    question, answer, contexts = _validated_eval_inputs(request)

    try:
        evaluation = eval_answer(_eval_state(question, answer, contexts))["evaluation"]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "success": True,
        "evaluation": evaluation,
        "scores": {name: res.get("score") for name, res in evaluation.items()},
        "elapsed_total": round(time.perf_counter() - started, 3),
    }

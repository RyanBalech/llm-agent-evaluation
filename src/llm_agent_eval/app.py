"""RepoLens HTTP API and same-origin web interface. Run one worker per data directory."""
from __future__ import annotations

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .repository import normalize_repository, validate_ref
from .service import CapacityError, IndexService

PACKAGE = Path(__file__).parent


class IndexRequest(BaseModel):
    repository: str = Field(min_length=3, max_length=250)
    revision: str = Field(default="", max_length=151)
    dense: bool = False

    @field_validator("repository")
    @classmethod
    def repository_format(cls, value):
        return normalize_repository(value)

    @field_validator("revision")
    @classmethod
    def revision_format(cls, value):
        return validate_ref(value.strip())


class SearchRequest(BaseModel):
    index_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    query: str = Field(min_length=3, max_length=8000)
    method: Literal["bm25", "dense", "hybrid"] = "bm25"
    context_budget_chars: int = Field(default=40000, ge=1000, le=80000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Describe the issue in at least three characters")
        return value.strip()


def create_app(data_dir: Path | None = None, *, token: str | None = None) -> FastAPI:
    access_token = token if token is not None else os.getenv("REPOLENS_TOKEN", "")
    public_demo = os.getenv("REPOLENS_PUBLIC_DEMO") == "1"
    explanations_enabled = os.getenv("REPOLENS_ENABLE_EXPLANATIONS") == "1"

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = IndexService(
            data_dir or Path(os.getenv("REPOLENS_DATA_DIR", ".repolens")),
            enable_dense=os.getenv("REPOLENS_ENABLE_DENSE") == "1")
        app.state.explanations = None
        if explanations_enabled:
            from .explanations import ExplanationJobs
            app.state.explanations = ExplanationJobs(
                app.state.service, device=os.getenv("REPOLENS_EXPLAIN_DEVICE", "cpu"),
                public=public_demo)
        if public_demo:
            app.state.service.submit("demo")
            if os.getenv("REPOLENS_DEMO_REVISION"):
                existing = app.state.service.list()
                if not any(r["repository"] == "RyanBalech/llm-agent-evaluation"
                           and r["status"] != "failed" for r in existing):
                    app.state.service.submit("RyanBalech/llm-agent-evaluation",
                                             os.environ["REPOLENS_DEMO_REVISION"])
        yield
        if app.state.explanations is not None:
            app.state.explanations.close()
        app.state.service.close()

    app = FastAPI(title="RepoLens", version="0.1.0", lifespan=lifespan,
                  description="Repository issue localization with source-linked retrieval evidence")
    hosts = ["localhost", "127.0.0.1", "[::1]", "testserver"]
    hosts.extend(h.strip() for h in os.getenv("REPOLENS_HOSTS", "").split(",") if h.strip())
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        origin = request.headers.get("origin")
        if (request.method not in {"GET", "HEAD", "OPTIONS"} and origin
                and urlsplit(origin).netloc != request.headers.get("host")):
            return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not public_demo:
            response.headers["X-Frame-Options"] = "DENY"
        if request.url.path not in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors "
                + ("'self' https://huggingface.co" if public_demo else "'none'"))
        response.headers["Cache-Control"] = "no-store"
        return response

    def service(request: Request):
        if access_token and not public_demo:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {access_token}"):
                raise HTTPException(401, "Enter the server access token")
        return request.app.state.service

    def require_write_access(request: Request):
        if public_demo:
            supplied = request.headers.get("authorization", "")
            if not access_token or not hmac.compare_digest(supplied, f"Bearer {access_token}"):
                raise HTTPException(403, "Public demo repositories are managed by the operator")

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse({"detail": "Index not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=429 if isinstance(exc, CapacityError) else 409)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    dependency = Depends(service)

    @app.get("/api/config")
    def config(s: IndexService = dependency):
        return {"dense_enabled": s.enable_dense, "max_indexes": 20, "max_pending": 3,
                "public_demo": public_demo, "explanations_enabled": explanations_enabled}

    @app.get("/api/indexes")
    def indexes(s: IndexService = dependency):
        return s.list()

    @app.post("/api/indexes", status_code=202)
    def index(payload: IndexRequest, request: Request, s: IndexService = dependency):
        require_write_access(request)
        return s.submit(payload.repository, payload.revision, dense=payload.dense)

    @app.post("/api/demo", status_code=202)
    def demo(s: IndexService = dependency):
        return s.submit("demo")

    @app.get("/api/indexes/{index_id}")
    def index_status(index_id: str, s: IndexService = dependency):
        return s.get(index_id)

    @app.delete("/api/indexes/{index_id}", status_code=204)
    def delete_index(index_id: str, request: Request, s: IndexService = dependency):
        require_write_access(request)
        s.delete(index_id)

    @app.post("/api/search")
    def search(payload: SearchRequest, s: IndexService = dependency):
        return s.search(payload.index_id, payload.query, payload.method,
                        payload.context_budget_chars, payload.top_k)

    @app.post("/api/explanations", status_code=202)
    def explain(payload: SearchRequest, request: Request, s: IndexService = dependency):
        if request.app.state.explanations is None:
            raise HTTPException(409, "Local explanations are disabled on this server")
        if s.get(payload.index_id)["status"] != "ready":
            raise HTTPException(409, "Wait for the index to become ready")
        return request.app.state.explanations.submit(payload.model_dump())

    @app.get("/api/explanations/{job_id}")
    def explanation_status(job_id: str, request: Request, s: IndexService = dependency):
        if request.app.state.explanations is None:
            raise HTTPException(409, "Local explanations are disabled on this server")
        return request.app.state.explanations.get(job_id)

    @app.get("/api/evaluation")
    def evaluation(s: IndexService = dependency):
        result = json.loads((PACKAGE / "evidence" / "baseline.json").read_text())
        studies = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((PACKAGE / "evidence").glob("study_*.json"))]
        limitations = ["12 of 20 tasks are Django; this is a small development slice.",
                       "File localization does not measure successful bug fixes.",
                       "Latency is from the archived environment, not this server."]
        if not studies:
            limitations.append("Dense and hybrid full-study results are not yet available.")
        return {"baseline": result, "label": "Archived 20-task SWE-bench Verified experiment",
                "studies": studies,
                "limitations": limitations,
                "source": "https://github.com/RyanBalech/llm-agent-evaluation/blob/main/"
                          "results/published/swebench_verified_bm25_20.json"}

    app.mount("/static", StaticFiles(directory=PACKAGE / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(PACKAGE / "static" / "index.html")

    return app


def main():
    import argparse

    import uvicorn
    parser = argparse.ArgumentParser(description="Run the RepoLens application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if (args.host not in {"127.0.0.1", "localhost", "::1"}
            and not os.getenv("REPOLENS_TOKEN") and os.getenv("REPOLENS_PUBLIC_DEMO") != "1"):
        parser.error("Set REPOLENS_TOKEN before listening on a network interface")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

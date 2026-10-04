"""Main FastAPI application for LA_ML (Linear Algebra in Machine Learning)."""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.math_core import (
    compute_transform,
    compute_pca,
    compute_least_squares,
    compute_neural_layer,
    compute_svd,
    compute_low_rank_approx,
    compute_embeddings_similarity,
    compute_vector_analogy,
    compute_svm_classification,
    compute_self_attention,
    compute_hessian_landscape
)
from app.challenges import (
    CHALLENGE_TYPES,
    generate_challenge,
    verify_challenge
)

app = FastAPI(
    title="LA_ML: Linear Algebra in Machine Learning",
    description="Интерактивный учебный модуль по линейной алгебре и ML",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    css_dir = os.path.join(STATIC_DIR, "css")
    js_dir = os.path.join(STATIC_DIR, "js")
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")


# Pydantic Request Models
class TransformRequest(BaseModel):
    matrix: List[List[float]] = Field(..., examples=[[[1.0, 0.0], [0.0, 1.0]]])
    bias: Optional[List[float]] = Field(default=[0.0, 0.0], examples=[[0.0, 0.0]])


class PCARequest(BaseModel):
    points: List[List[float]] = Field(..., examples=[[[1.0, 2.0], [2.0, 4.0], [3.0, 5.0]]])
    n_components: Optional[int] = Field(default=1, ge=1, le=2)


class LeastSquaresRequest(BaseModel):
    points: List[List[float]] = Field(..., examples=[[[1.0, 2.0], [2.0, 3.5], [3.0, 5.2]]])
    fit_intercept: Optional[bool] = Field(default=True)


class NeuralLayerRequest(BaseModel):
    matrix: List[List[float]] = Field(..., examples=[[[1.0, 0.0], [0.0, 1.0]]])
    bias: List[float] = Field(default=[0.0, 0.0])
    points: List[Dict[str, Any]] = Field(default=[])
    activation: str = Field(default="none")


class SVDRequest(BaseModel):
    matrix: List[List[float]] = Field(..., examples=[[[1.0, 2.0], [0.0, 2.0]]])


class LowRankRequest(BaseModel):
    matrix: List[List[float]]
    rank_k: int = Field(default=1, ge=1)


class EmbeddingQueryRequest(BaseModel):
    query: List[float] = Field(..., examples=[[1.0, 0.5]])
    documents: List[Dict[str, Any]]
    top_k: Optional[int] = Field(default=3, ge=1)


class AnalogyRequest(BaseModel):
    vec_a: List[float]
    vec_b: List[float]
    vec_c: List[float]
    vocabulary: Optional[List[Dict[str, Any]]] = Field(default=[])


class SVMRequest(BaseModel):
    weights: List[float] = Field(..., examples=[[1.0, 1.0]])
    bias: float = Field(default=0.0)
    points: List[Dict[str, Any]] = Field(default=[])


class AttentionRequest(BaseModel):
    tokens: List[str]
    Q: List[List[float]]
    K: List[List[float]]
    V: List[List[float]]
    scale_factor: Optional[float] = None


class HessianRequest(BaseModel):
    eigenvals: List[float] = Field(default=[5.0, 0.5])
    angle_deg: float = Field(default=30.0)
    lr: float = Field(default=0.15)
    momentum_beta: float = Field(default=0.85)
    start_point: List[float] = Field(default=[-3.0, 2.5])
    steps: Optional[int] = Field(default=35, ge=5, le=100)


class VerifySubmissionRequest(BaseModel):
    submission: Dict[str, Any]


@app.get("/")
async def root():
    """Serve the main interactive SPA page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "LA_ML Backend running. Static UI not yet created."}


@app.post("/api/transform")
async def api_transform(req: TransformRequest):
    """Compute linear transformation properties."""
    try:
        return compute_transform(req.matrix, req.bias)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pca")
async def api_pca(req: PCARequest):
    """Compute PCA eigenvalues, eigenvectors, and projections."""
    try:
        if len(req.points) < 2:
            raise HTTPException(status_code=400, detail="At least 2 points are required for PCA.")
        return compute_pca(req.points, req.n_components or 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/least-squares")
async def api_least_squares(req: LeastSquaresRequest):
    """Compute OLS linear regression and subspace projection."""
    try:
        if len(req.points) < 2:
            raise HTTPException(status_code=400, detail="At least 2 points are required for Least Squares.")
        return compute_least_squares(req.points, req.fit_intercept if req.fit_intercept is not None else True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/neural-layer")
async def api_neural_layer(req: NeuralLayerRequest):
    """Simulate dense layer matrix transformation + activation."""
    try:
        return compute_neural_layer(req.matrix, req.bias, req.points, req.activation)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/svd")
async def api_svd(req: SVDRequest):
    """Compute SVD decomposition and 2D unit circle geometric stages."""
    try:
        return compute_svd(req.matrix)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/low-rank")
async def api_low_rank(req: LowRankRequest):
    """Compute Eckart-Young rank-k approximation and LoRA parameter compression."""
    try:
        return compute_low_rank_approx(req.matrix, req.rank_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/embeddings/similarity")
async def api_embeddings_similarity(req: EmbeddingQueryRequest):
    """Compute Cosine Similarity, distance metrics, and Top-K retrieval."""
    try:
        return compute_embeddings_similarity(req.query, req.documents, req.top_k or 3)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/embeddings/analogy")
async def api_embeddings_analogy(req: AnalogyRequest):
    """Compute vector analogy A - B + C = Target (Word2Vec / semantic search)."""
    try:
        return compute_vector_analogy(req.vec_a, req.vec_b, req.vec_c, req.vocabulary)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/svm")
async def api_svm(req: SVMRequest):
    """Compute linear classification, normal vector, margin and hyperplane metrics."""
    try:
        return compute_svm_classification(req.weights, req.bias, req.points)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/attention")
async def api_attention(req: AttentionRequest):
    """Compute Transformer Self-Attention matrix decomposition Attention(Q, K, V)."""
    try:
        return compute_self_attention(req.tokens, req.Q, req.K, req.V, req.scale_factor)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/hessian-landscape")
async def api_hessian_landscape(req: HessianRequest):
    """Compute Hessian condition number kappa and simulate SGD vs Momentum trajectories."""
    try:
        return compute_hessian_landscape(
            req.eigenvals,
            req.angle_deg,
            req.lr,
            req.momentum_beta,
            req.start_point,
            req.steps or 35
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/challenges/list")
async def api_challenges_list():
    """Return available practical challenges."""
    return CHALLENGE_TYPES


@app.get("/api/challenges/generate/{challenge_id}")
async def api_challenge_generate(challenge_id: str):
    """Generate a challenge instance."""
    try:
        return generate_challenge(challenge_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/challenges/verify/{challenge_id}")
async def api_challenge_verify(challenge_id: str, req: VerifySubmissionRequest):
    """Verify challenge solution."""
    try:
        return verify_challenge(challenge_id, req.submission)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import threading
    import time
    import webbrowser
    import uvicorn

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("   LA_ML - Linear Algebra in Machine Learning")
    print("   Сервер запущен на: http://127.0.0.1:8000")
    print("   (Страница откроется в браузере автоматически)")
    print("=" * 60 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)

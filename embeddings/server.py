"""
Embeddings microservice — mxbai-embed-large-v1 (1024-dim)
Replaces all-MiniLM-L6-v2 for higher quality semantic search.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
PORT = int(os.environ.get("PORT", 8765))

model: SentenceTransformer = None
EMBED_DIM: int = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, EMBED_DIM
    log.info(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    EMBED_DIM = model.get_sentence_embedding_dimension()
    log.info(f"Model ready. dim={EMBED_DIM}")
    yield

app = FastAPI(lifespan=lifespan)

class EmbedRequest(BaseModel):
    texts: list[str]
    # mxbai needs a prefix for docs at index time — caller can pass it
    # For queries, prefix is added server-side via ?query=true
    is_query: bool = False

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": model is not None, "dim": EMBED_DIM}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not model:
        raise HTTPException(503, "Model not loaded yet")
    if not req.texts:
        return EmbedResponse(embeddings=[], model=MODEL_NAME, dim=EMBED_DIM)
    texts = req.texts
    if req.is_query:
        # mxbai query prefix improves retrieval quality
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    vecs = model.encode(texts, normalize_embeddings=True).tolist()
    return EmbedResponse(embeddings=vecs, model=MODEL_NAME, dim=len(vecs[0]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

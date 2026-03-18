#!/usr/bin/env python3
"""
FastAPI embeddings service using OpenAI text-embedding-3-small (1536-dim).
Matches the vectors already stored in pgvector.
"""
import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME     = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM      = 1536

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Vault Embeddings Service",
    description="OpenAI text-embedding-3-small proxy for vault-search",
    version="2.0.0",
)


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    normalize: bool = Field(default=True)
    is_query: bool = Field(default=False)


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "dimension": EMBED_DIM}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    try:
        response = client.embeddings.create(
            model=MODEL_NAME,
            input=request.texts,
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Embedded {len(request.texts)} text(s) via OpenAI {MODEL_NAME}")
        return EmbedResponse(
            embeddings=embeddings,
            model=MODEL_NAME,
            dimension=EMBED_DIM,
        )
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

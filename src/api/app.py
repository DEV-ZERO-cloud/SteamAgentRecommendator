from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.game import Game
from models.recommendation import Recommendation
from models.user import User, UserQuery
from pipeline.pipeline_recommendation import pipeline

# ── Configuración ─────────────────────────────────────────────────────────────
CSV_PATH = Path(
    os.getenv(
        "STEAM_CSV",
        str(Path(__file__).resolve().parents[2] / "src" / "data" / "steam_rpg_games.csv"),
    )
)

# Intentar cargar datos al arrancar (si el CSV ya existe)
"""if CSV_PATH.exists():
    try:
        engine.load_data(CSV_PATH)
    except Exception as exc:
        print(f"[Startup] No se pudo cargar el CSV automáticamente: {exc}")
else:
    print(f"[Startup] CSV no encontrado en {CSV_PATH}. Usa POST /train para cargar datos.")"""

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Steam RPG Recommender Agent",
    description="Agente híbrido de recomendación de RPGs en Steam (embeddings + reglas).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RecommendResponse(BaseModel):
    user_id: str
    query: str
    recommendations: list[Recommendation]
    total_candidates: int


class TrainResponse(BaseModel):
    status: str
    games_loaded: int
    index_vectors: int

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/train")
def train_model():
    pipeline.knowledge_engine.build(force=True)
    return {"games_loaded": len(pipeline.knowledge_engine.games)}

@app.post("/semantic_engine/search")
async def search_recommendations(request: RecommendationRequest):
    # Split the query string into a list of tags
    tags = [tag.strip() for tag in request.query.split(",")]
    
    results = pipeline.semantic_engine.search(
        tags=tags,
        top_k=request.top_k
    )

    return {
        "query": request.query,
        "results": results
    }

@app.get("/games", response_model=list[Game])
def list_games(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tag: Optional[str] = Query(None, description="Filtrar por tag (parcial, case-insensitive)"),
):
    games = pipeline.knowledge_engine.games
    if tag:
        tag_lower = tag.lower()
        games = [g for g in games if any(tag_lower in t.lower() for t in g.tags)]

    start = (page - 1) * page_size
    return games[start: start + page_size]


@app.get("/games/{app_id}", response_model=Game)
def get_game(app_id: int):
    game = pipeline.knowledge_engine.id_to_game.get(app_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Juego {app_id} no encontrado.")
    return game

"""@app.get("/tags", response_model=list[str])
def list_tags(limit: int = Query(100, ge=1, le=500)):
    #Devuelve los tags más frecuentes en el catálogo.
    counter: Counter = Counter()
    for g in engine._games:
        counter.update(g.tags)
    return [tag for tag, _ in counter.most_common(limit)]"""
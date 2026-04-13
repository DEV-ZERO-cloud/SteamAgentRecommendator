"""
app.py – API REST del agente recomendador de RPGs en Steam.

Endpoints:
  GET  /health              – estado del servicio
  POST /recommend           – recomendaciones para un usuario
  GET  /games               – lista paginada de juegos en el catálogo
  GET  /games/{app_id}      – detalle de un juego específico
  GET  /tags                – lista de tags disponibles
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.game import Game
from src.models.recommendation import Recommendation
from src.models.user import User, UserPreferences
from src.recommender.engine import RecommendationEngine

# ── Configuración ─────────────────────────────────────────────────────────────
CSV_PATH = os.getenv(
    "STEAM_CSV",
    str(Path(__file__).resolve().parents[2] / "src" / "data" / "steam_rpg_games.csv"),
)

engine = RecommendationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el dataset al arrancar la API."""
    print(CSV_PATH)
    csv = Path(CSV_PATH)
    if csv.exists():
        engine.load_data(csv)
    else:
        print(f"[API] ADVERTENCIA: CSV no encontrado en {CSV_PATH}. "
              "El motor arrancará vacío – coloca el CSV o define la variable STEAM_CSV.")
    yield


app = FastAPI(
    title="Steam RPG Recommender Agent",
    description="Agente híbrido de recomendación de RPGs en Steam (embeddings + reglas).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas de request/response ───────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id: str = "guest"
    name: str = "Jugador"
    preferences: UserPreferences = UserPreferences()
    played_app_ids: list[int] = []
    top_n: int = 10


class RecommendResponse(BaseModel):
    user_id: str
    query: str
    recommendations: list[Recommendation]
    total_candidates: int


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "games_loaded": len(engine._games),
        "index_ready": engine._index is not None,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not engine._games:
        raise HTTPException(
            status_code=503,
            detail="El motor aún no tiene datos. Verifica que el CSV esté disponible.",
        )

    user = User(
        user_id        = req.user_id,
        name           = req.name,
        preferences    = req.preferences,
        played_app_ids = req.played_app_ids,
    )

    recs = engine.recommend(user, top_n=min(req.top_n, 20))

    query = req.preferences.free_query or " ".join(req.preferences.preferred_tags) or "RPG general"

    return RecommendResponse(
        user_id          = req.user_id,
        query            = query,
        recommendations  = recs,
        total_candidates = len(engine._games),
    )


@app.get("/games", response_model=list[Game])
def list_games(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tag: Optional[str] = Query(None, description="Filtrar por tag (parcial, case-insensitive)"),
):
    games = engine._games
    if tag:
        tag_lower = tag.lower()
        games = [g for g in games if any(tag_lower in t.lower() for t in g.tags)]

    start = (page - 1) * page_size
    return games[start: start + page_size]


@app.get("/games/{app_id}", response_model=Game)
def get_game(app_id: int):
    idx = engine._id_to_idx.get(app_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Juego {app_id} no encontrado.")
    return engine._games[idx]


@app.get("/tags", response_model=list[str])
def list_tags(limit: int = Query(100, ge=1, le=500)):
    """Devuelve los tags más frecuentes en el catálogo."""
    from collections import Counter
    counter: Counter = Counter()
    for g in engine._games:
        counter.update(g.tags)
    return [tag for tag, _ in counter.most_common(limit)]
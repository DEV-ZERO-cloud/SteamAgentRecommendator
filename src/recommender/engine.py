"""
engine.py – Motor principal del agente recomendador de RPGs en Steam.

Pipeline:
  1. Carga el dataset CSV y construye/carga índice FAISS
  2. Recibe una consulta de usuario (UserPreferences)
  3. Ejecuta búsqueda semántica (embedding query → FAISS)
  4. Aplica FilterStrategy → RankStrategy → DiversifyStrategy
  5. Genera explicaciones y devuelve lista de Recommendation
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.models.game import Game
from src.models.recommendation import Recommendation
from src.models.user import User, UserPreferences
from src.recommender.explanation import generate_explanation
from src.recommender.score import (
    combine_scores,
    compute_preference_score,
    compute_rating_bonus,
)
from src.recommender.strategy import DiversifyStrategy, FilterStrategy, RankStrategy

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[2]
DATA_DIR       = BASE_DIR / "src" / "data"
EMBEDDINGS_DIR = BASE_DIR / "src" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH  = EMBEDDINGS_DIR / "games.index"
META_PATH   = EMBEDDINGS_DIR / "games_meta.pkl"

# Modelo local de HuggingFace (sin necesidad de Ollama)
# Modelos recomendados (en orden de calidad/tamaño):
#   - "sentence-transformers/all-MiniLM-L6-v2"   (rápido, 80 MB)
#   - "sentence-transformers/all-mpnet-base-v2"  (mejor calidad, 420 MB)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K_FAISS   = 200   # candidatos semánticos antes del filtro
TOP_N_RESULT  = 10    # resultados finales a devolver


# ── Helpers de texto ─────────────────────────────────────────────────────────
def _game_to_text(g: Game) -> str:
    """Crea el texto que se embeds para representar un juego."""
    tags_str = ", ".join(g.tags[:15])
    return (
        f"{g.name}. {g.description[:300]} "
        f"Géneros: {', '.join(g.genres)}. Tags: {tags_str}. "
        f"Desarrollado por {g.developer}."
    )


def _parse_list_field(value) -> list[str]:
    """Parsea un campo que puede ser lista JSON, string separado por comas o NaN."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                return json.loads(value)
            except Exception:
                pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


# ── Motor ────────────────────────────────────────────────────────────────────
class RecommendationEngine:
    def __init__(self, csv_path: Optional[str | Path] = None):
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._games: list[Game] = []
        self._id_to_idx: dict[int, int] = {}

        if csv_path:
            self.load_data(Path(csv_path))

    # ── Carga de datos ────────────────────────────────────────────────────────
    def load_data(self, csv_path: Path) -> None:
        print(f"[Engine] Cargando dataset: {csv_path}")
        df = pd.read_csv(csv_path, sep="|")
        self._games = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing juegos"):
            try:
                # Normalización de campos ─────────────────────────────────────
                categories = _parse_list_field(row.get("Categories", []))
                tags    = _parse_list_field(row.get("Tags", []))
                genres  = _parse_list_field(row.get("Genres", []))

                positive = float(row.get("Positive", 0) or 0)
                negative = float(row.get("Negative", 0) or 0)
                total   = int(positive+negative)

                recommendations = float(row.get("Recommendations"))

                # Rating derivado: pos_ratio ponderado por volumen (0-10)
                if total > 0:
                    confidence = min(1.0, np.log10(total+ 1) / 4)
                    raw_rating = positive * 10
                    rating = raw_rating * confidence + 5.0 * (1 - confidence)
                else:
                    rating = 0.0

                release_year = None
                rd = str(row.get("Release date", "") or "")
                if len(rd) >= 4 and rd[:4].isdigit():
                    release_year = int(rd[:4])

                game = Game(
                    app_id = int(row.get("AppID")),
                    name   = str(row.get("Name", "Unknown")),
                    required_ages = int(row.get("Required age")),
                    price         = float(row.get("Price", 0) or 0),
                    positive_reviews = positive,
                    negative_reviews = negative,
                    recommendations_quantity = recommendations,
                    categories    = categories,
                    genres        = genres,
                    tags          = tags,
                    description   = str(row.get("short_description", row.get("description", "")) or ""),
                    rating        = round(rating, 2),
                    total_reviews = total,
                    release_year  = release_year,
                )
                self._games.append(game)
            except Exception as e:
                print(f"[Engine] Error parseando fila: {e}")
                continue

        self._id_to_idx = {g.app_id: i for i, g in enumerate(self._games)}
        print(f"[Engine] {len(self._games)} juegos cargados.")
        self._ensure_index()

    # ── Embeddings & FAISS ────────────────────────────────────────────────────
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[Engine] Cargando modelo: {EMBED_MODEL}")
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _ensure_index(self) -> None:
        """Carga el índice FAISS desde disco o lo construye si no existe."""
        if INDEX_PATH.exists() and META_PATH.exists():
            print("[Engine] Cargando índice FAISS existente…")
            self._index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                saved_ids = pickle.load(f)
            # Verificar que coincide con los juegos cargados
            if len(saved_ids) == len(self._games):
                print("[Engine] Índice FAISS listo.")
                return
            print("[Engine] Índice desactualizado, reconstruyendo…")

        self._build_index()

    def _build_index(self) -> None:
        model = self._load_model()
        texts = [_game_to_text(g) for g in self._games]

        print("[Engine] Generando embeddings (puede tardar unos minutos)…")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner Product = coseno (con normalización)
        self._index.add(embeddings)

        faiss.write_index(self._index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump([g.app_id for g in self._games], f)

        print(f"[Engine] Índice FAISS construido con {len(self._games)} vectores.")

    # ── Búsqueda semántica ────────────────────────────────────────────────────
    def _semantic_search(self, query: str, k: int = TOP_K_FAISS) -> dict[int, float]:
        """Devuelve {app_id: similarity_score} para los k juegos más cercanos."""
        model = self._load_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, k)
        result: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._games):
                result[self._games[idx].app_id] = float(score)
        return result

    # ── Pipeline principal ────────────────────────────────────────────────────
    def recommend(
        self,
        user: User,
        top_n: int = TOP_N_RESULT,
    ) -> list[Recommendation]:
        if not self._games:
            raise RuntimeError("El motor no tiene juegos cargados. Llama a load_data() primero.")

        prefs = user.preferences
        t0 = time.time()

        # 1. Construir texto de consulta semántica
        query_parts = []
        if prefs.free_query:
            query_parts.append(prefs.free_query)
        if prefs.preferred_tags:
            query_parts.append("RPG " + " ".join(prefs.preferred_tags))
        query = " ".join(query_parts) if query_parts else "RPG adventure fantasy story rich"

        # 2. Búsqueda semántica (FAISS)
        semantic_scores = self._semantic_search(query, k=min(TOP_K_FAISS, len(self._games)))
        candidate_ids   = set(semantic_scores.keys())
        candidates      = [g for g in self._games if g.app_id in candidate_ids]

        # 3. Filtro (reglas duras)
        filter_strat = FilterStrategy()
        filtered = filter_strat.apply(
            candidates,
            prefs=prefs,
            played_ids=user.played_app_ids,
        )

        # 4. Ranking híbrido
        rank_strat = RankStrategy()
        ranked = rank_strat.apply(filtered, semantic_scores=semantic_scores, prefs=prefs)

        # 5. Diversificación
        diverse_strat = DiversifyStrategy(max_per_subgenre=2, top_n=top_n * 2)
        diverse = diverse_strat.apply(ranked)

        # 6. Construir objetos Recommendation
        recommendations: list[Recommendation] = []
        for rank_pos, game in enumerate(diverse[:top_n], start=1):
            sem   = semantic_scores.get(game.app_id, 0.0)
            pref  = compute_preference_score(game, prefs)
            rat   = compute_rating_bonus(game)
            final = combine_scores(sem, pref, rat)
            expl  = generate_explanation(game, prefs, sem, pref, final)

            recommendations.append(
                Recommendation(
                    game             = game,
                    final_score      = round(final, 4),
                    semantic_score   = round(sem, 4),
                    preference_score = round(pref, 4),
                    explanation      = expl,
                    rank             = rank_pos,
                )
            )

        elapsed = time.time() - t0
        print(f"[Engine] {len(recommendations)} recomendaciones en {elapsed:.2f}s")
        return recommendations
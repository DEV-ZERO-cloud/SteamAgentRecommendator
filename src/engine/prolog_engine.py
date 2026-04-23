"""
prolog_engine.py – Paso 5 del pipeline de recomendación.

Responsabilidad:
    Recibir los GameScore del ParametersEngine, aplicar las reglas
    simbólicas del motor_logico (Python puro) para filtrar, enriquecer
    y explicar las recomendaciones finales.

Decisiones de diseño:
    1. No hay columna discount en el CSV → Capa 6 inactiva.
    2. Usuario sintético "query_user" cuyos liked_tags vienen de la query.
    3. similar_game sobre el conjunto pequeño de candidatos (<= 50).
    4. motor_logico es Python puro — sin pyDatalog, sin dependencias externas.

Flujo:
    [GameScore, ...] (Paso 4)
        → construir contexto lógico (tags, liked_tags, similar_pairs)
        → aplicar reglas motor_logico
        → devolver [EnrichedScore, ...] filtrados y enriquecidos
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from engine import motor_logico
from engine.parameters_engine import GameScore
from models.game import Game

logger = logging.getLogger(__name__)

SYNTHETIC_USER = "query_user"


# =============================================================================
# Modelo de salida enriquecido
# =============================================================================

@dataclass
class EnrichedScore:
    """GameScore enriquecido con metadatos simbólicos del motor lógico."""
    game_score: GameScore
    explanations: list[str] = field(default_factory=list)
    price_tier: Optional[str] = None
    rating_tier: Optional[str] = None
    tag_overlap: float = 0.0
    is_rpg: bool = False


# =============================================================================
# Motor Prolog
# =============================================================================

class PrologEngine:
    """
    Paso 5 del pipeline: aplica las reglas del motor_logico a los candidatos
    del ParametersEngine y retorna una lista enriquecida de recomendaciones.

    Uso:
        prolog_engine = PrologEngine()
        enriched = prolog_engine.filter(scored_games, games_by_id, query_tags)
    """

    def __init__(
        self,
        disliked_tags: list[str] | None = None,
        max_price: float = 0.0,
    ):
        self.disliked_tags: set[str] = {t.lower().strip() for t in (disliked_tags or [])}
        self.max_price = max_price

    def filter(
        self,
        scored: list[GameScore],
        games_by_id: dict[int, Game],
        query_tags: list[str],
    ) -> list[EnrichedScore]:
        """
        Aplica las capas lógicas y devuelve resultados enriquecidos
        ordenados por (parameter_score DESC, tag_overlap DESC).

        Args:
            scored:      Lista de GameScore del ParametersEngine.
            games_by_id: {app_id: Game} del KnowledgeEngine.
            query_tags:  Tags de la query del usuario.

        Returns:
            Lista de EnrichedScore filtrada y ordenada.
        """
        # ── Construir contexto lógico ─────────────────────────────────────────
        liked_tags: set[str] = {t.lower().strip() for t in query_tags if t.strip()}

        tags_by_gid: dict[str, set[str]] = {}
        price_by_gid: dict[str, float] = {}
        rating_by_gid: dict[str, float] = {}

        for gs in scored:
            gid = str(gs.app_id)
            game = games_by_id.get(gs.app_id)
            if game is None:
                logger.warning("PrologEngine: app_id %d no encontrado.", gs.app_id)
                continue
            tags_by_gid[gid]   = {t.lower().strip() for t in game.tags if t.strip()}
            price_by_gid[gid]  = float(game.price)
            rating_by_gid[gid] = float(game.rating)

        # § 3 — similaridad sobre candidatos
        similar_pairs = motor_logico.compute_similar_pairs(tags_by_gid)

        # § 5 — candidatos válidos (usuario sintético: played_gids vacío)
        played_gids: set[str] = set()
        valid_gids = motor_logico.get_candidates(
            liked_tags, tags_by_gid, similar_pairs, played_gids
        )

        logger.info(
            "PrologEngine: %d candidatos recibidos, %d válidos tras reglas lógicas.",
            len(scored),
            len(valid_gids),
        )

        # Fallback: si ningún candidato pasó el filtro de tags,
        # usar todos (puede ocurrir con queries muy específicas)
        if not valid_gids:
            logger.debug("PrologEngine: fallback — usando todos los candidatos.")
            valid_gids = set(tags_by_gid.keys())

        # ── Aplicar filtros y enriquecer ──────────────────────────────────────
        results: list[EnrichedScore] = []

        for gs in scored:
            gid = str(gs.app_id)
            if gid not in valid_gids:
                continue

            game_tags = tags_by_gid.get(gid, set())
            price     = price_by_gid.get(gid, 0.0)
            rating    = rating_by_gid.get(gid, 0.0)

            # § 7 — filtro is_recommendable (RPG + dislike + precio)
            if not motor_logico.is_recommendable(
                game_tags, self.disliked_tags, self.max_price, price
            ):
                continue

            # § 7 — métricas de calidad
            overlap     = motor_logico.tag_overlap_score(game_tags, liked_tags)
            price_tier  = motor_logico.get_price_tier(price)
            rating_tier = motor_logico.get_rating_tier(rating)
            is_rpg_flag = motor_logico.is_rpg(game_tags)

            # § 8 — explicaciones
            explanations = motor_logico.get_explanations(
                gid, liked_tags, tags_by_gid, similar_pairs, played_gids
            )

            results.append(EnrichedScore(
                game_score=gs,
                explanations=explanations,
                price_tier=price_tier,
                rating_tier=rating_tier,
                tag_overlap=overlap,
                is_rpg=is_rpg_flag,
            ))

        # Orden final: parameter_score DESC, tag_overlap DESC
        results.sort(
            key=lambda e: (e.game_score.parameter_score, e.tag_overlap),
            reverse=True,
        )

        return results

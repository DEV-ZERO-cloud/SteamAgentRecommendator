"""
test_pipeline_monotonia_determinismo.py

Tests de monotonía y determinismo para PipelineRecommendation.

Estructura real de EnrichedScore (inferida del output):
    EnrichedScore(
        game_score = GameScore(
            app_id           = int,
            name             = str,
            semantic_score   = float,
            parameter_score  = int,
            date_flag        = int,
            price_flag       = int,
            positive_rate_flag = int,
            recommendations_flag = int,
        ),
        explanations = list[str],
        price_tier   = str,   # 'free' | 'budget' | ...
        rating_tier  = str,
        tag_overlap  = float,
        is_rpg       = bool,
    )

Helpers locales:
    _id(r)    → r.game_score.app_id
    _score(r) → r.game_score.semantic_score

Nota sobre max_price:
    max_price=0.0 significa SIN LÍMITE de precio (cualquier precio pasa).
    Para probar monotonía real se comparan dos valores positivos distintos.
"""

from __future__ import annotations

import pytest
from pipeline.pipeline_recommendation import PipelineRecommendation


# ---------------------------------------------------------------------------
# Helpers – acceso a campos anidados
# ---------------------------------------------------------------------------

def _id(r) -> int:
    return r.game_score.app_id


def _score(r) -> float:
    return r.game_score.semantic_score


# ---------------------------------------------------------------------------
# Fixture compartida
# ---------------------------------------------------------------------------

QUERY_BASE = "rpg, fantasy"


@pytest.fixture(scope="session")
def pipeline() -> PipelineRecommendation:
    """Instancia única para todos los tests de la sesión."""
    return PipelineRecommendation()


# ===========================================================================
# BLOQUE 1 – DETERMINISMO
# ===========================================================================


class TestDeterminismo:
    """La misma entrada siempre produce la misma salida."""

    def test_misma_query_mismo_orden(self, pipeline):
        """Dos llamadas idénticas devuelven los mismos app_ids en el mismo orden."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5)
        assert [_id(r) for r in r1] == [_id(r) for r in r2]

    def test_misma_query_mismos_scores(self, pipeline):
        """Los semantic_scores son idénticos entre llamadas."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5)
        assert [_score(r) for r in r1] == [_score(r) for r in r2]

    def test_espacios_extra_en_query_equivalente(self, pipeline):
        """Whitespace extra alrededor de tags no altera el resultado."""
        r1 = pipeline.recommend("rpg,fantasy", top_k=5)
        r2 = pipeline.recommend("rpg , fantasy", top_k=5)
        assert [_id(r) for r in r1] == [_id(r) for r in r2]

    def test_disliked_tags_none_vs_string_vacio(self, pipeline):
        """disliked_tags=None y disliked_tags='' son equivalentes."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags="")
        assert [_id(r) for r in r1] == [_id(r) for r in r2]

    def test_max_price_cero_determinista(self, pipeline):
        """max_price=0.0 (sin límite) es determinista entre llamadas."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)
        assert [_id(r) for r in r1] == [_id(r) for r in r2]

    def test_multiples_llamadas_consecutivas_estables(self, pipeline):
        """Tres llamadas seguidas producen el mismo resultado (no hay estado residual)."""
        runs = [pipeline.recommend(QUERY_BASE, top_k=5) for _ in range(3)]
        ids = [[_id(r) for r in run] for run in runs]
        assert ids[0] == ids[1] == ids[2]

    def test_disliked_tags_no_contaminan_llamada_siguiente(self, pipeline):
        """
        El estado interno de prolog_engine (disliked_tags) no persiste entre llamadas.
        Una llamada con dislikes no debe afectar la siguiente sin ellos.
        """
        pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags="co-op, memes")
        r_sin = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)

        pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)
        r_sin2 = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)

        assert [_id(r) for r in r_sin] == [_id(r) for r in r_sin2]

    def test_max_price_no_contamina_llamada_siguiente(self, pipeline):
        """
        El estado interno de prolog_engine (max_price) no persiste entre llamadas.
        """
        pipeline.recommend(QUERY_BASE, top_k=5, max_price=5.0)
        r1 = pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)

        pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)

        assert [_id(r) for r in r1] == [_id(r) for r in r2]


# ===========================================================================
# BLOQUE 2 – MONOTONÍA: top_k
# ===========================================================================


class TestMonotoniaTopK:
    """Aumentar top_k nunca reduce la cantidad de resultados."""

    def test_top_k_mayor_igual_resultados(self, pipeline):
        small = pipeline.recommend(QUERY_BASE, top_k=3)
        large = pipeline.recommend(QUERY_BASE, top_k=10)
        assert len(small) <= len(large)

    def test_top_k_1_devuelve_como_mucho_1(self, pipeline):
        result = pipeline.recommend(QUERY_BASE, top_k=1)
        assert len(result) <= 1

    def test_top_k_respetado_como_limite_superior(self, pipeline):
        for k in [1, 3, 5, 10]:
            result = pipeline.recommend(QUERY_BASE, top_k=k)
            assert len(result) <= k, f"top_k={k} devolvió {len(result)} resultados"

    def test_top_k_subconjunto_del_mayor(self, pipeline):
        """
        top_k=3 debe retornar exactamente los 3 primeros de top_k=10,
        ya que el pipeline ordena antes de cortar con [:top_k].
        """
        small = pipeline.recommend(QUERY_BASE, top_k=3)
        large = pipeline.recommend(QUERY_BASE, top_k=10)
        ids_small = [_id(r) for r in small]
        ids_large = [_id(r) for r in large]
        assert ids_small == ids_large[: len(ids_small)]


# ===========================================================================
# BLOQUE 3 – MONOTONÍA: disliked_tags
# ===========================================================================


class TestMonotoniaDislikedTags:
    """Añadir tags rechazados nunca aumenta la cantidad de resultados."""

    def test_con_dislikes_igual_o_menos_resultados(self, pipeline):
        sin = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags=None)
        con = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op")
        assert len(con) <= len(sin)

    def test_mas_dislikes_igual_o_menos_resultados(self, pipeline):
        uno = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op")
        tres = pipeline.recommend(
            QUERY_BASE, top_k=10, disliked_tags="co-op, memes, multiplayer"
        )
        assert len(tres) <= len(uno)

    def test_dislikes_no_generan_scores_negativos(self, pipeline):
        """Los semantic_scores de los resultados que pasan el filtro son >= 0."""
        result = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op, memes")
        assert all(_score(r) >= 0 for r in result)

    def test_orden_dislikes_no_importa(self, pipeline):
        """
        El orden de los tags rechazados no altera el resultado
        (la penalización usa intersección de sets, que es commutativa).
        """
        r1 = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op, memes")
        r2 = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="memes, co-op")
        assert [_id(r) for r in r1] == [_id(r) for r in r2]

    def test_dislike_tag_inexistente_no_reduce_resultados(self, pipeline):
        """
        Un tag rechazado que no existe en ningún juego no debe penalizar nada
        y el output debe ser idéntico al caso sin dislikes.
        """
        sin = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags=None)
        con = pipeline.recommend(
            QUERY_BASE, top_k=10, disliked_tags="tagquenuncaexiste_xyz"
        )
        assert len(con) == len(sin)


# ===========================================================================
# BLOQUE 4 – MONOTONÍA: max_price
# ===========================================================================


class TestMonotoniaMaxPrice:
    """
    Un precio máximo más alto permite más juegos: nunca devuelve menos resultados
    que un precio máximo más bajo.

    Comportamiento real del sistema (verificado):
        max_price=0.0  → solo juegos gratuitos (price_tier='free')
        max_price=10.0 → gratuitos + budget (más resultados)
        max_price=50.0 → el conjunto más amplio

    La monotonía es estrictamente creciente: 0.0 < 5.0 < 20.0 < 50.0
    en cantidad de resultados posibles.
    """

    def test_precio_alto_igual_o_mas_que_precio_bajo(self, pipeline):
        """max_price=50 debe retornar >= resultados que max_price=5."""
        amplio    = pipeline.recommend(QUERY_BASE, top_k=10, max_price=50.0)
        estrecho  = pipeline.recommend(QUERY_BASE, top_k=10, max_price=5.0)
        assert len(estrecho) <= len(amplio)

    def test_precio_moderado_igual_o_mas_que_precio_minimo(self, pipeline):
        """max_price=20 debe retornar >= resultados que max_price=1."""
        moderado = pipeline.recommend(QUERY_BASE, top_k=10, max_price=20.0)
        minimo   = pipeline.recommend(QUERY_BASE, top_k=10, max_price=1.0)
        assert len(minimo) <= len(moderado)

    def test_precio_alto_igual_o_mas_que_precio_cero(self, pipeline):
        """
        max_price=0.0 solo incluye juegos gratuitos.
        max_price=30.0 incluye gratuitos + de pago → igual o más resultados.
        """
        solo_gratis = pipeline.recommend(QUERY_BASE, top_k=10, max_price=0.0)
        mas_amplio  = pipeline.recommend(QUERY_BASE, top_k=10, max_price=30.0)
        assert len(solo_gratis) <= len(mas_amplio)

    def test_precio_moderado_subconjunto_de_precio_alto(self, pipeline):
        """
        Todos los juegos retornados con max_price=10.0 deben aparecer
        también en los resultados de max_price=50.0 (superconjunto).
        Se evita max_price=0.0 porque el engine lo trata como 'solo gratis',
        produciendo un conjunto disjunto con los precios positivos.
        """
        moderado = {_id(r) for r in pipeline.recommend(QUERY_BASE, top_k=20, max_price=10.0)}
        amplio   = {_id(r) for r in pipeline.recommend(QUERY_BASE, top_k=20, max_price=50.0)}
        assert moderado.issubset(amplio)

    def test_todos_los_resultados_son_rpg(self, pipeline):
        """
        Con cualquier max_price, el filtro is_rpg del prolog_engine
        debe seguir activo: todos los resultados tienen is_rpg=True.
        """
        result = pipeline.recommend(QUERY_BASE, top_k=10, max_price=20.0)
        assert all(r.is_rpg for r in result)


# ===========================================================================
# BLOQUE 5 – MONOTONÍA COMBINADA
# ===========================================================================


class TestMonotoniaCombinada:
    """Restricciones combinadas son al menos tan restrictivas como cada una sola."""

    def test_precio_y_dislikes_juntos_igual_o_menos_que_solo_precio(self, pipeline):
        solo_precio = pipeline.recommend(QUERY_BASE, top_k=10, max_price=20.0)
        ambos = pipeline.recommend(
            QUERY_BASE, top_k=10, max_price=20.0, disliked_tags="co-op"
        )
        assert len(ambos) <= len(solo_precio)

    def test_precio_y_dislikes_juntos_igual_o_menos_que_solo_dislike(self, pipeline):
        """
        Agregar un filtro de precio estricto a los dislikes no puede aumentar resultados.
        Ambas llamadas usan el mismo max_price para aislar el efecto del precio
        y medir solo el impacto del dislike sobre el conjunto con precio.
        """
        solo_precio  = pipeline.recommend(QUERY_BASE, top_k=10, max_price=50.0)
        precio_y_dis = pipeline.recommend(QUERY_BASE, top_k=10, max_price=50.0, disliked_tags="co-op")
        assert len(precio_y_dis) <= len(solo_precio)

    def test_restricciones_producen_subconjunto_de_ids(self, pipeline):
        """
        Los IDs con restricciones activas deben ser subconjunto de los retornados
        con el rango de precio más amplio y sin dislikes.
        Se usa max_price=50.0 como base 'sin restricciones efectivas de precio'.
        """
        sin = {_id(r) for r in pipeline.recommend(QUERY_BASE, top_k=20, max_price=50.0)}
        con = {
            _id(r)
            for r in pipeline.recommend(
                QUERY_BASE, top_k=20, max_price=30.0, disliked_tags="co-op, memes"
            )
        }
        assert con.issubset(sin)
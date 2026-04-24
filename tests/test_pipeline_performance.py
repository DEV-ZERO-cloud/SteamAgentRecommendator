"""
test_pipeline_monotonia_determinismo.py

Tests de monotonía y determinismo para PipelineRecommendation.

  - Monotonía: cambios en parámetros producen cambios consistentes y
    predecibles en el output (más restricciones → igual o menos resultados,
    nunca más).
  - Determinismo: la misma entrada siempre produce la misma salida.

Fixture 'pipeline' se inicializa una sola vez por sesión para evitar
reconstruir los motores en cada test (costoso si el CSV es grande).
"""

from __future__ import annotations

import pytest
from pipeline.pipeline_recommendation import PipelineRecommendation


QUERY_BASE = "rpg, fantasy"


@pytest.fixture(scope="session")
def pipeline() -> PipelineRecommendation:
    """Instancia única para todos los tests de la sesión."""
    return PipelineRecommendation()

# BLOQUE 1 – DETERMINISMO


class TestDeterminismo:
    """La misma entrada siempre produce la misma salida."""

    def test_misma_query_mismo_orden(self, pipeline):
        """Dos llamadas idénticas devuelven los mismos app_ids en el mismo orden."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5)
        assert [r.app_id for r in r1] == [r.app_id for r in r2]

    def test_misma_query_mismos_scores(self, pipeline):
        """Los scores numéricos son idénticos entre llamadas."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5)
        assert [r.score for r in r1] == [r.score for r in r2]

    def test_espacios_extra_en_query_equivalente(self, pipeline):
        """Whitespace extra alrededor de tags no altera el resultado."""
        r1 = pipeline.recommend("rpg,fantasy", top_k=5)
        r2 = pipeline.recommend("rpg , fantasy", top_k=5)
        assert [r.app_id for r in r1] == [r.app_id for r in r2]

    def test_disliked_tags_none_vs_string_vacio(self, pipeline):
        """disliked_tags=None y disliked_tags='' son equivalentes."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags="")
        assert [r.app_id for r in r1] == [r.app_id for r in r2]

    def test_max_price_cero_determinista(self, pipeline):
        """max_price=0.0 (sin límite) es determinista entre llamadas."""
        r1 = pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)
        r2 = pipeline.recommend(QUERY_BASE, top_k=5, max_price=0.0)
        assert [r.app_id for r in r1] == [r.app_id for r in r2]

    def test_multiples_llamadas_consecutivas_estables(self, pipeline):
        """Tres llamadas seguidas producen el mismo resultado (no hay estado residual)."""
        runs = [pipeline.recommend(QUERY_BASE, top_k=5) for _ in range(3)]
        ids = [[r.app_id for r in run] for run in runs]
        assert ids[0] == ids[1] == ids[2]

    def test_disliked_tags_no_contaminan_llamada_siguiente(self, pipeline):
        """
        Estado interno del prolog_engine no persiste entre llamadas.
        Una llamada con disliked_tags no debe afectar la siguiente sin ellos.
        """
        pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags="co-op, memes")
        r_sin = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)

        pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)
        r_sin2 = pipeline.recommend(QUERY_BASE, top_k=5, disliked_tags=None)

        assert [r.app_id for r in r_sin] == [r.app_id for r in r_sin2]

# BLOQUE 2 – MONOTONÍA: top_k

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
        Los resultados de top_k=3 deben ser los mismos 3 primeros de top_k=10,
        ya que el pipeline ordena por score antes de cortar.
        """
        small = pipeline.recommend(QUERY_BASE, top_k=3)
        large = pipeline.recommend(QUERY_BASE, top_k=10)
        ids_small = [r.app_id for r in small]
        ids_large = [r.app_id for r in large]
        assert ids_small == ids_large[: len(ids_small)]


# BLOQUE 3 – MONOTONÍA: disliked_tags


class TestMonotoniaDislikedTags:
    """Añadir tags rechazados nunca aumenta la cantidad de resultados."""

    def test_con_dislikes_igual_o_menos_resultados(self, pipeline):
        sin = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags=None)
        con = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op")
        assert len(con) <= len(sin)

    def test_mas_dislikes_igual_o_menos_resultados(self, pipeline):
        uno = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op")
        tres = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op, memes, multiplayer")
        assert len(tres) <= len(uno)

    def test_dislikes_no_generan_scores_negativos(self, pipeline):
        """Los scores de los resultados que sí pasan nunca son negativos."""
        result = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op, memes")
        assert all(r.score >= 0 for r in result)

    def test_orden_dislikes_no_importa(self, pipeline):
        """
        El orden en que se listan los tags rechazados no altera el resultado
        (la penalización usa intersección de sets).
        """
        r1 = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op, memes")
        r2 = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="memes, co-op")
        assert [r.app_id for r in r1] == [r.app_id for r in r2]

    def test_dislike_tag_inexistente_no_reduce_resultados(self, pipeline):
        """
        Un tag rechazado que no existe en ningún juego no debe cambiar el output.
        """
        sin = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags=None)
        con = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="tagquenuncaexiste_xyz")
        assert len(con) == len(sin)


# BLOQUE 4 – MONOTONÍA: max_price

class TestMonotoniaMaxPrice:
    """Reducir max_price nunca aumenta la cantidad de resultados."""

    def test_sin_limite_vs_con_limite(self, pipeline):
        sin_limite = pipeline.recommend(QUERY_BASE, top_k=10, max_price=0.0)
        con_limite = pipeline.recommend(QUERY_BASE, top_k=10, max_price=20.0)
        assert len(con_limite) <= len(sin_limite)

    def test_precio_mas_bajo_igual_o_menos_resultados(self, pipeline):
        caro = pipeline.recommend(QUERY_BASE, top_k=10, max_price=50.0)
        barato = pipeline.recommend(QUERY_BASE, top_k=10, max_price=5.0)
        assert len(barato) <= len(caro)

    def test_precio_cero_punto_uno_es_muy_restrictivo(self, pipeline):
        """Un precio máximo casi cero debe devolver muy pocos o ningún resultado."""
        result = pipeline.recommend(QUERY_BASE, top_k=10, max_price=0.01)
        sin_limite = pipeline.recommend(QUERY_BASE, top_k=10, max_price=0.0)
        assert len(result) <= len(sin_limite)

    def test_resultados_respetan_max_price(self, pipeline):
        """
        Todos los juegos retornados deben tener price <= max_price.
        Requiere que EnrichedScore exponga el precio del juego.
        """
        max_p = 15.0
        result = pipeline.recommend(QUERY_BASE, top_k=10, max_price=max_p)
        for r in result:
            assert r.price <= max_p, (
                f"Juego {r.app_id} tiene precio {r.price} > max_price {max_p}"
            )

# BLOQUE 5 – MONOTONÍA COMBINADA

class TestMonotoniaCombinada:
    """Restricciones combinadas son al menos tan restrictivas como cada una sola."""

    def test_precio_y_dislikes_juntos_igual_o_menos(self, pipeline):
        solo_precio = pipeline.recommend(QUERY_BASE, top_k=10, max_price=20.0)
        solo_dislike = pipeline.recommend(QUERY_BASE, top_k=10, disliked_tags="co-op")
        ambos = pipeline.recommend(
            QUERY_BASE, top_k=10, max_price=20.0, disliked_tags="co-op"
        )
        assert len(ambos) <= len(solo_precio)
        assert len(ambos) <= len(solo_dislike)

    def test_restricciones_maximas_subset_de_sin_restricciones(self, pipeline):
        """Los IDs con restricciones deben ser subconjunto de los sin restricciones."""
        sin = set(r.app_id for r in pipeline.recommend(QUERY_BASE, top_k=20))
        con = set(
            r.app_id
            for r in pipeline.recommend(
                QUERY_BASE, top_k=20, max_price=30.0, disliked_tags="co-op, memes"
            )
        )
        assert con.issubset(sin)
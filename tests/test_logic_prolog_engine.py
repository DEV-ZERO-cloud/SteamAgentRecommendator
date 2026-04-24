"""
tests/test_logic_prolog_engine.py
Tests del PrologEngine y LogicEngine: carga de reglas simbólicas
"""
from engine.logic_engine import is_rpg, is_recommendable, tag_overlap_score, dislike_penalty
from engine.prolog_engine import PrologEngine, EnrichedScore
from engine.parameters_engine import GameScore
from models.game import Game



def make_game(app_id: int, title: str, tags: list[str], price: float, rating: float) -> Game:
    return Game(
        app_id = app_id,
        name   = title,
        tags   = tags,
        price  = price,
        rating = rating,
    )

GAMES = {
    1: make_game(1, "Baldur's Gate 3",       ["rpg", "turn-based", "fantasy", "open world"], 59.99, 9.5),
    2: make_game(2, "Divinity Original Sin 2",["rpg", "turn-based", "fantasy", "co-op"],     44.99, 9.2),
    3: make_game(3, "Hades",                  ["roguelike", "action", "rpg", "indie"],        24.99, 9.4),
    4: make_game(4, "Dead Space",             ["horror", "survival", "action", "sci-fi"],     39.99, 8.5),
    5: make_game(5, "Stardew Valley",         ["farming", "simulation", "indie", "rpg"],       14.99, 9.1),
}

SCORED = [
    GameScore(app_id=1, name="Baldur's Gate 3",        semantic_score=0.88, parameter_score=0.92, date_flag=1, price_flag=0, positive_rate_flag=1, recommendations_flag=1),
    GameScore(app_id=2, name="Divinity Original Sin 2", semantic_score=0.82, parameter_score=0.87, date_flag=1, price_flag=1, positive_rate_flag=1, recommendations_flag=1),
    GameScore(app_id=3, name="Hades",                   semantic_score=0.70, parameter_score=0.75, date_flag=1, price_flag=1, positive_rate_flag=1, recommendations_flag=1),
    GameScore(app_id=4, name="Dead Space",              semantic_score=0.60, parameter_score=0.65, date_flag=1, price_flag=1, positive_rate_flag=1, recommendations_flag=0),
    GameScore(app_id=5, name="Stardew Valley",          semantic_score=0.55, parameter_score=0.60, date_flag=1, price_flag=1, positive_rate_flag=1, recommendations_flag=1),
]

QUERY_TAGS   = ["rpg", "fantasy", "turn-based", "open world"]
DISLIKED     = ["horror", "survival"]
MAX_PRICE    = 50.0


# ── Tests logic_engine ────────────────────────────────────────────────────────

def test_is_rpg():
    from engine import logic_engine
    rpg_tags = logic_engine._get_rpg_tags()
    non_rpg  = {"farming", "simulation", "sports", "racing"} - rpg_tags

    assert is_rpg({"rpg"})   == True
    if non_rpg:
        assert is_rpg(non_rpg) == False, f"Se esperaba False para {non_rpg}"

def test_is_recommendable_no_rpg():
    from engine import logic_engine
    rpg_tags = logic_engine._get_rpg_tags()
    non_rpg  = {"farming", "simulation", "sports", "racing"} - rpg_tags
    if non_rpg:
        assert is_recommendable(non_rpg, set(), 60.0, 39.99) == False

def test_is_recommendable_dislike_penalty():
    from engine import logic_engine
    rpg_tags_list = list(logic_engine._get_rpg_tags())[:4]
    game_tags  = set(rpg_tags_list)           # todos son RPG
    disliked   = set(rpg_tags_list[:3])       # 3 coincidencias × 0.15 = 0.45
    assert is_recommendable(game_tags, disliked, 0.0, 10.0) == False


def test_is_recommendable_price():
    assert is_recommendable({"rpg"}, set(), 30.0, 59.99) == False

def test_tag_overlap_score():
    score = tag_overlap_score({"rpg", "fantasy", "action"}, {"rpg", "fantasy", "turn-based", "open world"})
    assert score == 2/4  # 2 coincidencias sobre 4 preferidos

def test_dislike_penalty():
    assert dislike_penalty({"rpg", "horror", "survival"}, {"horror", "survival"}) == 0.30


# ── Test PrologEngine.filter ──────────────────────────────────────────────────

def test_prolog_filter_excludes_non_rpg_and_disliked():
    engine = PrologEngine(disliked_tags=DISLIKED, max_price=MAX_PRICE)
    results = engine.filter(SCORED, GAMES, QUERY_TAGS)

    app_ids = [r.game_score.app_id for r in results]

    assert 4 not in app_ids, "Dead Space (horror, no RPG) debe estar excluido"
    assert all(r.is_rpg for r in results), "Todos los resultados deben ser RPG"

def test_prolog_filter_order():
    engine = PrologEngine(disliked_tags=DISLIKED, max_price=MAX_PRICE)
    results = engine.filter(SCORED, GAMES, QUERY_TAGS)

    scores = [r.game_score.parameter_score for r in results]
    assert scores == sorted(scores, reverse=True), "Deben estar ordenados por parameter_score DESC"

def test_prolog_filter_price_cap():
    engine = PrologEngine(disliked_tags=[], max_price=30.0)
    results = engine.filter(SCORED, GAMES, QUERY_TAGS)

    for r in results:
        game = GAMES[r.game_score.app_id]
        assert float(game.price) <= 30.0, f"{game.title} supera el max_price"

def test_prolog_enrichment_fields():
    engine = PrologEngine(disliked_tags=DISLIKED, max_price=MAX_PRICE)
    results = engine.filter(SCORED, GAMES, QUERY_TAGS)

    for r in results:
        assert r.price_tier  is not None
        assert r.rating_tier is not None
        assert 0.0 <= r.tag_overlap <= 1.0
        assert isinstance(r.explanations, list)
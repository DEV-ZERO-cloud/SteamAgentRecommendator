"""
test_engine.py – Tests del motor de recomendación.

Ejecutar:
    pytest tests/test_engine.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.models.user import User, UserPreferences
from src.recommender.explanation import generate_explanation
from src.recommender.score import (
    combine_scores,
    compute_preference_score,
    compute_rating_bonus,
)
from src.recommender.strategy import DiversifyStrategy, FilterStrategy, RankStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _make_game(**kwargs):
    from src.models.game import Game
    defaults = dict(
        app_id=1, name="Test RPG", genres=["RPG"],
        tags=["RPG","Action RPG","Open World"],
        description="A great RPG game.",
        price=29.99, rating=8.5, total_reviews=10000,
        positive_ratio=0.85, release_year=2022,
        developer="TestDev", publisher="TestPub",
        platforms=["Windows"], image_url="",
    )
    defaults.update(kwargs)
    return Game(**defaults)


@pytest.fixture
def sample_games():
    return [
        _make_game(app_id=1, name="Action Hero RPG",   tags=["Action RPG","Hack and Slash"],     price=29.99, rating=8.5),
        _make_game(app_id=2, name="Story Quest",        tags=["JRPG","Anime","Turn-Based"],        price=19.99, rating=9.1),
        _make_game(app_id=3, name="Free Adventure",     tags=["Free to Play","RPG","Open World"],  price=0.0,   rating=7.0),
        _make_game(app_id=4, name="Dark Dungeon",        tags=["Souls-like","Difficult","Dark Fantasy"], price=49.99, rating=6.5),
        _make_game(app_id=5, name="Roguelike Descent",  tags=["Roguelike","Dungeon Crawler"],      price=9.99,  rating=8.0),
    ]


@pytest.fixture
def base_prefs():
    return UserPreferences(preferred_tags=["Action RPG","Open World"])


# ── Score tests ───────────────────────────────────────────────────────────────
class TestScores:
    def test_preference_score_tag_match(self, sample_games, base_prefs):
        game = sample_games[0]  # Action RPG
        score = compute_preference_score(game, base_prefs)
        assert 0.0 < score <= 1.0

    def test_preference_score_price_filter(self, sample_games):
        prefs = UserPreferences(max_price=10.0)
        expensive = sample_games[3]  # $49.99
        assert compute_preference_score(expensive, prefs) == 0.0

    def test_preference_score_free_only(self, sample_games):
        prefs = UserPreferences(free_to_play_only=True)
        paid = sample_games[0]
        free = sample_games[2]
        assert compute_preference_score(paid, prefs) == 0.0
        assert compute_preference_score(free, prefs) > 0.0

    def test_preference_score_min_rating(self, sample_games):
        prefs = UserPreferences(min_rating=7.0)
        low_rated = sample_games[3]  # rating=6.5
        assert compute_preference_score(low_rated, prefs) == 0.0

    def test_rating_bonus_range(self, sample_games):
        for g in sample_games:
            bonus = compute_rating_bonus(g)
            assert 0.0 <= bonus <= 1.0

    def test_combine_scores_weights(self):
        result = combine_scores(1.0, 1.0, 1.0)
        assert abs(result - 1.0) < 1e-6

    def test_combine_scores_clamped(self):
        result = combine_scores(2.0, -1.0, 0.5)
        assert 0.0 <= result <= 1.0


# ── Strategy tests ────────────────────────────────────────────────────────────
class TestFilterStrategy:
    def test_filters_played(self, sample_games, base_prefs):
        strat = FilterStrategy()
        result = strat.apply(sample_games, prefs=base_prefs, played_ids=[1, 2])
        ids = [g.app_id for g in result]
        assert 1 not in ids
        assert 2 not in ids

    def test_filters_by_price(self, sample_games):
        prefs = UserPreferences(max_price=25.0)
        strat = FilterStrategy()
        result = strat.apply(sample_games, prefs=prefs)
        assert all(g.price <= 25.0 for g in result)

    def test_filters_disliked_tags(self, sample_games):
        prefs = UserPreferences(disliked_tags=["Souls-like","Difficult","Dark Fantasy"])
        strat = FilterStrategy()
        result = strat.apply(sample_games, prefs=prefs)
        ids = [g.app_id for g in result]
        assert 4 not in ids  # Dark Dungeon tiene 3 tags no deseados


class TestRankStrategy:
    def test_sorted_desc(self, sample_games, base_prefs):
        sem_scores = {g.app_id: 0.5 for g in sample_games}
        strat = RankStrategy()
        result = strat.apply(sample_games, semantic_scores=sem_scores, prefs=base_prefs)
        scores = [combine_scores(0.5, compute_preference_score(g, base_prefs), compute_rating_bonus(g))
                  for g in result]
        assert scores == sorted(scores, reverse=True)


class TestDiversifyStrategy:
    def test_max_per_subgenre(self, sample_games):
        strat = DiversifyStrategy(max_per_subgenre=1, top_n=10)
        result = strat.apply(sample_games)
        assert len(result) <= len(sample_games)


# ── Explanation tests ─────────────────────────────────────────────────────────
class TestExplanation:
    def test_explanation_not_empty(self, sample_games, base_prefs):
        game = sample_games[0]
        expl = generate_explanation(game, base_prefs, 0.8, 0.7, 0.75)
        assert isinstance(expl, str)
        assert len(expl) > 0

    def test_explanation_contains_score(self, sample_games, base_prefs):
        game = sample_games[1]
        expl = generate_explanation(game, base_prefs, 0.9, 0.8, 0.85)
        assert "%" in expl


# ── CSV + Engine integration ──────────────────────────────────────────────────
class TestEngineIntegration:
    def _make_csv(self, tmp_path: Path, n: int = 20) -> Path:
        import numpy as np
        rows = []
        for i in range(n):
            rows.append({
                "app_id": 100 + i,
                "name": f"RPG Game {i}",
                "genres": json.dumps(["RPG"]),
                "tags": json.dumps(["RPG","Action RPG","Open World"]),
                "short_description": "An epic RPG adventure in a vast fantasy world.",
                "price": float(i * 5 % 60),
                "total_reviews": 5000,
                "positive_ratio": 0.80,
                "release_date": "2022-01-01",
                "developer": "TestDev",
                "publisher": "TestPub",
                "platforms": json.dumps(["Windows"]),
                "header_image": "",
            })
        csv_path = tmp_path / "steam_rpg_games.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_load_and_recommend(self, tmp_path):
        from src.services.engine import RecommendationEngine

        csv = self._make_csv(tmp_path, n=30)
        eng = RecommendationEngine(csv_path=csv)

        assert len(eng._games) == 30

        user = User(
            user_id="u1",
            preferences=UserPreferences(
                preferred_tags=["Action RPG","Open World"],
                free_query="dark fantasy adventure",
            ),
        )
        recs = eng.recommend(user, top_n=5)
        assert len(recs) == 5
        assert all(r.rank == i + 1 for i, r in enumerate(recs))
        assert all(0.0 <= r.final_score <= 1.0 for r in recs)
from pydantic import BaseModel, Field
from typing import Optional


class Game(BaseModel):
    app_id: int
    name: str
    required_ages: int = 0
    positive_reviews: float = 0.0
    negative_reviews: float = 0.0
    recommendations_quantity: float = 0.0
    categories: list[str] = []
    genres: list[str] = []
    tags: list[str] = []
    platforms: Optional[list[str]] = None        # ← campo faltante (usado en score.py y explanation.py)
    price: float = 0.0
    rating: float = 0.0                # 0-10 score derivado de reviews
    total_reviews: int = 0
    positive_ratio: float = 0.0        # porcentaje reseñas positivas (0-1)
    release_year: Optional[int] = None
    short_description: Optional[str] = None

    class Config:
        populate_by_name = True
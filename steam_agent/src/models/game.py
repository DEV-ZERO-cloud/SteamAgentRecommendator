from pydantic import BaseModel, Field
from typing import Optional


class Game(BaseModel):
    app_id: int
    name: str
    genres: list[str] = []
    tags: list[str] = []
    description: str = ""
    price: float = 0.0
    rating: float = 0.0        # 0-10 score derivado de reviews
    total_reviews: int = 0
    positive_ratio: float = 0.0  # porcentaje reseñas positivas (0-1)
    release_year: Optional[int] = None
    developer: str = ""
    publisher: str = ""
    platforms: list[str] = []
    image_url: str = ""

    class Config:
        populate_by_name = True
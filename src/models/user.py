from pydantic import BaseModel, Field
from typing import Optional


class UserPreferences(BaseModel):
    preferred_tags: list[str] = Field(default_factory=list)
    disliked_tags: list[str] = Field(default_factory=list)
    max_price: Optional[float] = None          # None = sin límite
    min_rating: float = 0.0                    # mínimo rating aceptable (0-10)
    preferred_platforms: list[str] = Field(default_factory=list)
    free_to_play_only: bool = False
    free_query: str = ""                       # búsqueda en lenguaje natural


class User(BaseModel):
    user_id: str
    name: str = "Anónimo"
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    played_app_ids: list[int] = Field(default_factory=list)    # juegos ya jugados (excluir)
    wishlist_app_ids: list[int] = Field(default_factory=list)
from pydantic import BaseModel
from src.models.game import Game


class Recommendation(BaseModel):
    game: Game
    final_score: float          # 0-1 score combinado
    semantic_score: float       # similitud semántica (embeddings)
    preference_score: float     # score por preferencias/reglas
    explanation: str            # por qué se recomienda este juego
    rank: int                   # posición en la lista (1-based)
    
    model_config = {"arbitrary_types_allowed": True}
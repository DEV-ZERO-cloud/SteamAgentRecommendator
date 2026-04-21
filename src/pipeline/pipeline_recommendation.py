""""Pipeline de proceso de recomendacion:
    Paso 1: Obtención de la query de la request
    Paso 2: semantic_engine   → búsqueda semántica por tags -> Output {app_id: similarity}
    Paso 3: knowledge_engine  → normaliza IDs a objetos Game -> Output [Game, ...]
    Paso 4: parameters_engine → scoring por filtros -> Output [GameScore, ...]
    Paso 5: prolog_engine     → reglas de inferencia (rules.pl) -> Output [GameScore, ...] [pendiente]
"""
from engine.knowledge_engine import KnowledgeEngine
from engine.semantic_engine import SemanticEngine, SemanticEngineConfig
from engine.parameters_engine import ParametersEngine, ScoreFilters
from pipeline.prepare_data import PrepareData
from dotenv import load_dotenv
import os

load_dotenv()


class PipelineRecommendation:

    def __init__(self):
        csv_path        = os.getenv("CSV_PATH")
        parameters_path = os.getenv("PARAMETERS_PATH", "src/data/parameters.json")

        # Paso 2: motor semántico TF-IDF
        config = SemanticEngineConfig(
            csv_path=csv_path,
            id_column="appid",
            tag_column="tags",
            genre_column="genres",
        )
        prepared = PrepareData(config).prepare()
        self.semantic_engine = SemanticEngine(config, prepared)

        # Paso 3: base de conocimiento CSV → objetos Game
        self.knowledge_engine = KnowledgeEngine(csv_path=csv_path)
        self.knowledge_engine.build()

        # Paso 4: motor de scoring por parameters
        self.parameters_engine = ParametersEngine(parameters_path)

    def recommend(self, query: str, top_k: int = 5, filters: ScoreFilters = None):
        # Paso 1: query viene de la request

        # Paso 2: búsqueda semántica → {app_id: similarity}
        semantic_results = self.semantic_engine.search(tags=query.split(), top_k=200)
        semantic_scores  = dict(semantic_results)

        # Paso 3: IDs → objetos Game
        games = self.knowledge_engine.get_games(list(semantic_scores.keys()))

        # Paso 4: scoring por parameters → [GameScore, ...]
        scored = self.parameters_engine.score(games, semantic_scores, filters)

        # Paso 5: Prolog [pendiente] — por ahora pasa directo
        # scored = prolog_engine.filter(scored)

        return scored[:top_k]


pipeline = PipelineRecommendation()
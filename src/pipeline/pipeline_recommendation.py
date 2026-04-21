""""Pipeline de proceso de recomendacion:
    Paso 1: Obtención de la query de la request
    Paso 2: Hacer busqueda semantica a partir de tags -> Output lista de GameIDS
    Paso 3: Ejecutar motor con reglas de inferencia (rules.pl) -> Output lista de GameIDs
    Paso 4: Ejecutar scripts con script de parameters (parameters.json) 
    para ordenar recomendacions desde score parameters-> Output lista de Recommendations
"""
from engine.semantic_engine import SemanticEngine, SemanticEngineConfig
from engine.knowledge_engine import KnowledgeEngine
from pipeline.prepare_data import PrepareData
from models.explanation import generate_explanation
from dotenv import load_dotenv
import os

load_dotenv()
#-------------------------------------------
class PipelineRecommendation:
    
    def __init__(self):
        csv_path = csv_path=os.getenv("CSV_PATH")
        config = SemanticEngineConfig(
            csv_path=csv_path,
            id_column="appid",
            tag_column="tags",
            genre_column="genres",
        )
        prepared = PrepareData(config).prepare()
        self.semantic_engine = SemanticEngine(config, prepared)
        self.knowledge_engine = KnowledgeEngine(csv_path=csv_path)
        self.knowledge_engine.build()
    

pipeline = PipelineRecommendation()
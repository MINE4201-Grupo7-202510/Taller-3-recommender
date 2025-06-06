from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from typing import List, Dict, Any
import pickle
from joblib import dump, load as joblib_load
import os
import asyncio # Para el Lock
import math # Para math.isnan y math.isinf

# --- Configuración ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MODEL_DIR = "trained_models_cache"

app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="Un API para obtener recomendaciones de películas con opción de reentrenamiento.",
    version="0.2.0" # Nueva versión con explicaciones
)

# --- Configuración de CORS ---
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variables Globales y Lock para Reentrenamiento ---
driver: Driver | None = None 
node2vec_model: Word2Vec | None = None
collab_similarities_dict: Dict[str, Dict[str, float]] = {}
semantic_similarities_dict: Dict[str, Dict[str, int]] = {}
all_movie_ids_from_graph: List[str] = []
all_user_ids_from_graph: List[str] = []

retrain_lock = asyncio.Lock()
is_retraining = False

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main") 

# --- Funciones de Conexión a Neo4j ---
def get_neo4j_driver() -> Driver:
    global driver
    should_create_driver = True
    if driver is not None:
        try:
            driver.verify_connectivity()
            logger.debug("Driver de Neo4j existente verificado exitosamente.")
            should_create_driver = False
        except Exception as e:
            logger.warning(f"Driver de Neo4j existente no pudo verificar conectividad: {e}. Se intentará recrear.")
            try:
                if driver:
                    driver.close()
                    logger.info("Driver de Neo4j anterior cerrado antes de recrear.")
            except Exception as close_exc:
                logger.error(f"Error al intentar cerrar el driver de Neo4j anterior: {close_exc}")
            driver = None

    if should_create_driver:
        try:
            logger.info("Creando/Recreando instancia del driver de Neo4j...")
            new_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            new_driver.verify_connectivity()
            driver = new_driver 
            logger.info("✅ Nueva conexión exitosa a Neo4j")
        except Exception as e:
            logger.error(f"❌ Error al crear/conectar con Neo4j: {e}", exc_info=True)
            raise RuntimeError(f"No se pudo conectar a Neo4j: {e}")
    
    if driver is None: 
        raise RuntimeError("El driver de Neo4j no pudo ser inicializado.")
    return driver


# --- Funciones de Carga/Guardado de Modelos ---
def save_model_data(data_name: str, data: Any):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{data_name}.pkl")
    if data_name == "node2vec_model" and isinstance(data, Word2Vec):
        data.save(os.path.join(MODEL_DIR, "node2vec.model"))
    elif isinstance(data, (list, dict, np.ndarray)):
        path = os.path.join(MODEL_DIR, f"{data_name}.joblib")
        dump(data, path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    logger.info(f"💾 Datos '{data_name}' guardados en {path.replace('.pkl','.joblib') if isinstance(data, (list,dict, np.ndarray)) else path}")


def load_model_data(data_name: str) -> Any:
    pickle_path = os.path.join(MODEL_DIR, f"{data_name}.pkl")
    joblib_path = os.path.join(MODEL_DIR, f"{data_name}.joblib")
    gensim_path = os.path.join(MODEL_DIR, "node2vec.model")

    if data_name == "node2vec_model" and os.path.exists(gensim_path):
        logger.info(f"🔄 Cargando '{data_name}' desde {gensim_path}...")
        return Word2Vec.load(gensim_path)
    elif os.path.exists(joblib_path):
        logger.info(f"🔄 Cargando '{data_name}' desde {joblib_path}...")
        return joblib_load(joblib_path)
    elif os.path.exists(pickle_path):
        logger.info(f"🔄 Cargando '{data_name}' desde {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    logger.info(f"❔ Archivo para '{data_name}' no encontrado en {MODEL_DIR}.")
    return None

# --- Lógica de Entrenamiento y Generación ---
async def perform_full_training_and_save() -> bool:
    global node2vec_model, collab_similarities_dict, semantic_similarities_dict
    global all_movie_ids_from_graph, all_user_ids_from_graph, is_retraining, driver 

    if is_retraining and retrain_lock.locked(): 
        logger.info("perform_full_training_and_save llamado mientras is_retraining es True. Omitiendo.")
        return False

    is_retraining = True
    try:
        logger.info("--- ⏳ Iniciando proceso completo de generación y guardado de modelos... ---")
        
        current_db_driver = get_neo4j_driver()
        if not current_db_driver:
            logger.error("❌ No se pudo obtener el driver de Neo4j. Abortando entrenamiento.")
            return False

        with current_db_driver.session(database="neo4j") as session:
            logger.info("Obteniendo todos los UserIDs y MovieIDs de Neo4j...")
            user_ids_result = session.run("MATCH (u:User) RETURN u.userId AS userId").data()
            all_user_ids_from_graph = [record['userId'] for record in user_ids_result if record['userId'] is not None]
            save_model_data("all_user_ids", all_user_ids_from_graph)
            
            movie_ids_result = session.run("MATCH (m:Movie) RETURN m.movieId AS movieId").data()
            all_movie_ids_from_graph = [record['movieId'] for record in movie_ids_result if record['movieId'] is not None]
            save_model_data("all_movie_ids", all_movie_ids_from_graph)
            logger.info(f"Encontrados y guardados {len(all_user_ids_from_graph)} usuarios y {len(all_movie_ids_from_graph)} películas.")

        logger.info("📊 Generando modelo Node2Vec...")
        def get_edges_for_n2v(tx):
            query = "MATCH (u:User)-[r:RATED]->(m:Movie) RETURN u.userId, m.movieId"
            return tx.run(query).data()

        G = nx.Graph()
        with current_db_driver.session(database="neo4j") as session:
            edges = session.execute_read(get_edges_for_n2v)
            for edge in edges:
                if edge.get('u.userId') and edge.get('m.movieId'): 
                    G.add_edge(f"user_{edge['u.userId']}", f"movie_{edge['m.movieId']}")
        
        temp_node2vec_model = None
        temp_collab_similarities = {}

        if not G.nodes():
            logger.warning("⚠️ El grafo para Node2Vec está vacío.")
        else:
            logger.info(f"Grafo para Node2Vec construido con {len(G.nodes())} nodos y {len(G.edges())} aristas.")
            walks = []
            nodes_for_walks = list(G.nodes())
            for node_idx, node in enumerate(nodes_for_walks):
                if node_idx % 5000 == 0: logger.info(f"Generando caminatas... Nodo {node_idx}/{len(nodes_for_walks)}")
                current_walk = [node]
                for _ in range(10): 
                    neighbors = list(G.neighbors(current_walk[-1]))
                    if not neighbors: break
                    current_walk.append(np.random.choice(neighbors))
                walks.append(current_walk)


            if not walks:
                logger.warning("⚠️ No se generaron caminatas para Node2Vec.")
            else:
                logger.info(f"Entrenando modelo Word2Vec (Node2Vec) en {len(walks)} caminatas...")
                temp_node2vec_model = Word2Vec(walks, vector_size=64, window=5, min_count=1, sg=1, workers=max(1, os.cpu_count() // 2) )
                save_model_data("node2vec_model", temp_node2vec_model)
                logger.info("✅ Modelo Node2Vec entrenado y guardado.")

                _movie_embeddings = {node: temp_node2vec_model.wv[node] for node in temp_node2vec_model.wv.index_to_key if "movie_" in node}
                _user_embeddings = {node: temp_node2vec_model.wv[node] for node in temp_node2vec_model.wv.index_to_key if "user_" in node}

                if _movie_embeddings and _user_embeddings:
                    movie_vectors = np.array(list(_movie_embeddings.values()))
                    movie_ids_sim = [mid for mid in _movie_embeddings.keys() if mid.startswith("movie_")]
                    user_vectors = np.array(list(_user_embeddings.values()))
                    user_ids_sim = [uid for uid in _user_embeddings.keys() if uid.startswith("user_")]

                    if len(user_vectors) > 0 and len(movie_vectors) > 0:
                        logger.info("Calculando matriz de similitud colaborativa...")
                        sim_matrix = cosine_similarity(user_vectors, movie_vectors)
                        temp_collab_similarities = {
                            user_ids_sim[i]: {
                                movie_ids_sim[j]: float(sim_matrix[i, j])
                                for j in range(len(movie_ids_sim))
                            }
                            for i in range(len(user_ids_sim))
                        }
                        save_model_data("collab_similarities", temp_collab_similarities)
                        logger.info("✅ Similitudes colaborativas precomputadas y guardadas.")
                    else:
                        logger.warning("⚠️ No hay suficientes vectores para similitud colaborativa.")
                else:
                    logger.warning("⚠️ No se generaron embeddings para películas o usuarios.")
        
        logger.info("🔎 Generando similitudes semánticas...")
        temp_semantic_similarities = {}
        def get_semantic_sim_for_movie(tx, movie_id_no_prefix):
            query_simple = """
            MATCH (m1:Movie {movieId: $movie_id})
            MATCH (m2:Movie)
            WHERE m1 <> m2
            OPTIONAL MATCH (m1)-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(m2)
            OPTIONAL MATCH (m1)-[:ACTED_BY]->(a:Actor)<-[:ACTED_BY]-(m2)
            OPTIONAL MATCH (m1)-[:DIRECTED_BY]->(d:Director)<-[:DIRECTED_BY]-(m2)
            WITH m2,
                 COUNT(DISTINCT g) AS common_genres,
                 COUNT(DISTINCT a) AS common_actors,
                 COUNT(DISTINCT d) AS common_directors
            WHERE common_genres > 0 OR common_actors > 0 OR common_directors > 0 
            RETURN m2.movieId AS similar_movie_id,
                   (common_genres * 2) + common_actors + common_directors AS similarity_score 
            ORDER BY similarity_score DESC LIMIT 20
            """
            return {record["similar_movie_id"]: record["similarity_score"] for record in tx.run(query_simple, movie_id=movie_id_no_prefix)}


        if not all_movie_ids_from_graph:
            logger.warning("⚠️ No hay movie IDs para similitudes semánticas.")
        else:
            with current_db_driver.session(database="neo4j") as session:
                for idx, movie_id_np in enumerate(all_movie_ids_from_graph):
                    if idx % 2000 == 0: logger.info(f"Calculando similitud semántica... Película {idx}/{len(all_movie_ids_from_graph)}")
                    try:
                        sims = session.execute_read(get_semantic_sim_for_movie, movie_id_np)
                        if sims: 
                            temp_semantic_similarities[movie_id_np] = sims
                    except Exception as e:
                        logger.error(f"Error precalculando similitud semántica para movie_id {movie_id_np}: {e}", exc_info=False) 
            save_model_data("semantic_similarities", temp_semantic_similarities)
            logger.info(f"✅ Similitudes semánticas precomputadas y guardadas para {len(temp_semantic_similarities)} películas.")

        node2vec_model = temp_node2vec_model
        collab_similarities_dict = temp_collab_similarities
        semantic_similarities_dict = temp_semantic_similarities
        
        logger.info("🏁 Proceso completo de generación y guardado de modelos finalizado.")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error durante el proceso de entrenamiento y guardado: {e}", exc_info=True)
        return False
    finally:
        is_retraining = False


# --- Lógica de Inicialización en Startup ---
@app.on_event("startup")
async def initialize_models_and_data():
    global node2vec_model, collab_similarities_dict, semantic_similarities_dict
    global all_movie_ids_from_graph, all_user_ids_from_graph, driver

    logger.info("🚀 Iniciando el sistema de recomendación...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        get_neo4j_driver() 
    except RuntimeError as e:
        logger.error(f"Fallo crítico al inicializar Neo4j driver en startup: {e}. Algunas funcionalidades pueden no estar disponibles.")

    logger.info("--- Intentando cargar datos precalculados ---")
    loaded_users = load_model_data("all_user_ids")
    loaded_movies = load_model_data("all_movie_ids")
    loaded_n2v = load_model_data("node2vec_model")
    loaded_collab_sim = load_model_data("collab_similarities")
    loaded_semantic_sim = load_model_data("semantic_similarities")
    
    needs_initial_generation = not loaded_users or \
                               not loaded_movies or \
                               loaded_n2v is None or \
                               not loaded_collab_sim or \
                               not loaded_semantic_sim
    
    if needs_initial_generation:
        logger.info("--- ⏳ Datos precalculados no encontrados o incompletos. Generando... ---")
        if driver is None: 
             logger.error("No se puede generar modelos porque el driver de Neo4j no está disponible.")
        else:
            await perform_full_training_and_save()
    else:
        all_user_ids_from_graph = loaded_users
        all_movie_ids_from_graph = loaded_movies
        node2vec_model = loaded_n2v
        collab_similarities_dict = loaded_collab_sim
        semantic_similarities_dict = loaded_semantic_sim
        logger.info("✅ Todos los modelos y datos precalculados se cargaron exitosamente desde el caché.")

    if not all_user_ids_from_graph: logger.warning("⚠️ Lista de User IDs está vacía.")
    if not node2vec_model or not collab_similarities_dict: logger.warning("⚠️ Componentes colaborativos no disponibles.")
    if not semantic_similarities_dict: logger.warning("⚠️ Componentes semánticos no disponibles.")
    
    logger.info("🏁 Inicialización del sistema completada.")


@app.on_event("shutdown")
async def shutdown_event():
    global driver
    if driver: 
        try:
            driver.close()
            logger.info("🔌 Conexión a Neo4j cerrada.")
        except Exception as e:
            logger.error(f"Error al cerrar el driver de Neo4j: {e}", exc_info=True)


# --- Endpoint de Reentrenamiento ---
@app.post("/admin/retrain-models", summary="Forzar el reentrenamiento de todos los modelos y datos")
async def force_retrain_models():
    if retrain_lock.locked():
        logger.info("Reentrenamiento solicitado, pero el lock ya está adquirido por otro proceso.")
        raise HTTPException(status_code=429, detail="Otro proceso de reentrenamiento ya está en curso. Inténtalo más tarde.")

    await retrain_lock.acquire()
    try:
        logger.info("🔑 Solicitud de reentrenamiento recibida. Iniciando...")
        
        try:
            get_neo4j_driver()
            if driver is None:
                raise RuntimeError("El driver de Neo4j no está disponible para reentrenamiento.")
        except RuntimeError as e:
            logger.error(f"Fallo al obtener driver de Neo4j para reentrenamiento: {e}")
            raise HTTPException(status_code=500, detail=f"No se pudo conectar a Neo4j para reentrenamiento: {e}")

        success = await perform_full_training_and_save()
        
        if success:
            return {"message": "Reentrenamiento completado y modelos actualizados exitosamente."}
        else:
            raise HTTPException(status_code=500, detail="El proceso de reentrenamiento falló. Revisa los logs.")
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error inesperado en el endpoint de reentrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno durante el reentrenamiento: {str(e)}")
    finally:
        if retrain_lock.locked():
            retrain_lock.release()


# --- Funciones de Recomendación, Sanitización y Explicación ---
def sanitize_value_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None 
    return value

async def _get_movie_details_internal(movie_id: str, db_driver: Driver) -> Dict[str, Any]:
    query = """
    MATCH (m:Movie {movieId: $movie_id})
    OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
    OPTIONAL MATCH (m)-[:ACTED_BY]->(a:Actor)
    OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Director)
    RETURN m.movieId AS movieId, m.title AS title, m.tconst AS tconst,
           COLLECT(DISTINCT g.name) AS genres,
           COLLECT(DISTINCT a.name) AS actors,
           COLLECT(DISTINCT d.name) AS directors
    LIMIT 1
    """
    default_data = {"movieId": movie_id, "title": "Película Desconocida", "tconst": None, "genres": [], "actors": [], "directors": []}
    error_data = {"movieId": movie_id, "title": "Error al cargar detalles", "tconst": None, "genres": [], "actors": [], "directors": []}

    try:
        with db_driver.session(database="neo4j") as session:
            result = session.run(query, movie_id=movie_id)
            record = result.single()

        if not record:
            logger.warning(f"Internal details fetch: Movie with ID '{movie_id}' not found in DB.")
            return default_data
        
        data = dict(record)
        
        for key in ["genres", "actors", "directors"]:
            value = data.get(key)
            if value is not None:
                sanitized_list = [sanitize_value_for_json(item) for item in value]
                data[key] = [item for item in sanitized_list if item is not None]
            else:
                data[key] = []
        
        if data.get("title") is None:
            data["title"] = "Título no disponible"
        
        data["tconst"] = sanitize_value_for_json(data.get("tconst"))
        data["movieId"] = sanitize_value_for_json(data.get("movieId", movie_id))


        return data
    except Neo4jError as e:
        logger.error(f"Neo4j error fetching internal details for movie {movie_id}: {e}", exc_info=True)
        return error_data
    except Exception as e:
        logger.error(f"Unexpected error fetching internal details for movie {movie_id}: {e}", exc_info=True)
        return error_data

def _create_shared_feature_explanation_fragment(current_movie_details: Dict[str, Any], anchor_movie_details: Dict[str, Any]) -> str:
    current_genres = set(current_movie_details.get("genres", []))
    anchor_genres = set(anchor_movie_details.get("genres", []))
    common_genres = list(current_genres.intersection(anchor_genres))

    if common_genres:
        return f"ya que comparten el género '{common_genres[0]}'"

    current_actors = set(current_movie_details.get("actors", []))
    anchor_actors = set(anchor_movie_details.get("actors", []))
    common_actors = list(current_actors.intersection(anchor_actors))
    if common_actors:
        return f"ya que el actor '{common_actors[0]}' participa en ambas"

    current_directors = set(current_movie_details.get("directors", []))
    anchor_directors = set(anchor_movie_details.get("directors", []))
    common_directors = list(current_directors.intersection(anchor_directors))
    if common_directors:
        return f"ya que comparten al director '{common_directors[0]}'"
    
    return "por su conexión temática o de producción"


def get_colaborative_recommendations(user_id_no_prefix: str, top_n: int = 10) -> List[Dict[str, Any]]:
    user_id_with_prefix = f"user_{user_id_no_prefix}"
    if not collab_similarities_dict or user_id_with_prefix not in collab_similarities_dict:
        logger.warning(f"Usuario {user_id_with_prefix} no encontrado en collab_similarities_dict o el dict está vacío.")
        return []
    
    user_similarities = collab_similarities_dict[user_id_with_prefix]
    
    valid_similarities = {
        movie_id_wp: score
        for movie_id_wp, score in user_similarities.items()
        if not (isinstance(score, float) and (math.isnan(score) or math.isinf(score)))
    }

    sorted_movies_with_prefix = sorted(valid_similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    recommendations = []
    for movie_id_wp, score in sorted_movies_with_prefix:
        movie_id_np = movie_id_wp.replace("movie_", "")
        recommendations.append({"movieId": movie_id_np, "score": float(score)}) 
    return recommendations


def get_ontological_recommendations_for_movie(movie_id_no_prefix: str, top_n: int = 10) -> List[Dict[str, Any]]:
    if not semantic_similarities_dict or movie_id_no_prefix not in semantic_similarities_dict:
        logger.warning(f"MovieId {movie_id_no_prefix} no encontrado en semantic_similarities_dict o el dict está vacío.")
        return []
        
    movie_ontological_sims = semantic_similarities_dict[movie_id_no_prefix]
    valid_sims = {
        movie_id: score
        for movie_id, score in movie_ontological_sims.items()
        if not (isinstance(score, float) and (math.isnan(score) or math.isinf(score)))
    }

    sorted_movies = sorted(valid_sims.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    recommendations = []
    for movie_id_np, score in sorted_movies:
        recommendations.append({"movieId": movie_id_np, "score": int(score)}) 
    return recommendations

# --- Endpoints de la API ---
@app.get("/")
async def root():
    return {"message": "Bienvenido al API de Recomendación de Películas! v0.2.0"}

@app.get("/recommend/hybrid/{user_id}", summary="Recomendaciones híbridas para un usuario existente")
async def hybrid_recommend_endpoint(user_id: str, alpha: float = 0.6, top_n: int = 10):
    logger.info(f"Solicitud de recomendación híbrida para user_id: {user_id}, alpha: {alpha}, top_n: {top_n}")
    db_driver = get_neo4j_driver()

    if not node2vec_model or not collab_similarities_dict : 
        logger.warning("Componentes colaborativos no disponibles. No se pueden generar recomendaciones híbridas.")
        return {"user_id": user_id, "alpha": alpha, "recommendations": [], "message": "Modelo colaborativo no disponible."}

    collab_recs_initial = get_colaborative_recommendations(user_id, top_n=top_n * 2) # Get more for potential filtering
    
    if not collab_recs_initial:
        logger.info(f"No se encontraron recomendaciones colaborativas para user_id: {user_id}")
        return {"user_id": user_id, "alpha": alpha, "recommendations": [], "message": "No hay suficientes datos para recomendaciones colaborativas."}

    best_collab_movie_id_np = collab_recs_initial[0]["movieId"] if collab_recs_initial else None

    if not best_collab_movie_id_np: 
        logger.warning(f"No se pudo determinar una película ancla para {user_id}, devolviendo solo colaborativas.")
        # Populate explanations for collab_recs_initial before returning
        explained_collab_recs = []
        for rec in collab_recs_initial[:top_n]:
            movie_details = await _get_movie_details_internal(rec["movieId"], db_driver)
            movie_title = movie_details.get('title', 'esta película')
            rec["explanation"] = f"Te recomendamos '{movie_title}' porque usuarios con gustos similares a los tuyos también la valoraron positivamente."
            explained_collab_recs.append(rec)
        return {"user_id": user_id, "alpha": alpha, "recommendations": explained_collab_recs, "message": "Devolviendo solo colaborativas (no se encontró ancla ontológica)."}
    
    anchor_movie_details = await _get_movie_details_internal(best_collab_movie_id_np, db_driver)
    anchor_title = anchor_movie_details.get('title', 'una película que te gustó')

    user_id_wp = f"user_{user_id}" 
    original_collab_scores_for_user_wp = collab_similarities_dict.get(user_id_wp, {})
    
    ontological_scores_for_anchor_np = semantic_similarities_dict.get(best_collab_movie_id_np, {})
    max_onto_score = max(ontological_scores_for_anchor_np.values()) if ontological_scores_for_anchor_np else 1.0
    if max_onto_score == 0: max_onto_score = 1.0 

    combined_scores_np: Dict[str, float] = {}

    # Calculate combined scores
    all_potential_movie_ids_np = set(m.replace("movie_", "") for m in original_collab_scores_for_user_wp.keys())
    all_potential_movie_ids_np.update(ontological_scores_for_anchor_np.keys())

    for movie_id_np in all_potential_movie_ids_np:
        # Collaborative part
        collab_score_raw = original_collab_scores_for_user_wp.get(f"movie_{movie_id_np}", -2.0) # Use -2 to indicate not present if needed
        collab_score_norm = (collab_score_raw + 1) / 2 if collab_score_raw > -1.5 else 0 # Normalize, or 0 if not in user's collab scores
        
        # Ontological part
        onto_score_raw = ontological_scores_for_anchor_np.get(movie_id_np, 0)
        normalized_onto_score = onto_score_raw / max_onto_score if max_onto_score > 0 else 0
        
        combined_scores_np[movie_id_np] = (alpha * collab_score_norm) + ((1 - alpha) * normalized_onto_score)

    sorted_combined_recs_tuples = sorted(combined_scores_np.items(), key=lambda x: x[1], reverse=True)
    
    final_recommendations = []
    seen_movie_ids = set()

    for movie_id_np, score_val in sorted_combined_recs_tuples:
        if len(final_recommendations) >= top_n: break
        if movie_id_np in seen_movie_ids: continue
        
        current_movie_details = await _get_movie_details_internal(movie_id_np, db_driver)
        movie_title = current_movie_details.get('title', 'esta película')

        # Recalculate individual contributions for explanation
        collab_score_raw = original_collab_scores_for_user_wp.get(f"movie_{movie_id_np}", -2.0)
        collab_score_norm = (collab_score_raw + 1) / 2 if collab_score_raw > -1.5 else 0.0
        
        onto_score_raw = ontological_scores_for_anchor_np.get(movie_id_np, 0)
        normalized_onto_score = onto_score_raw / max_onto_score if max_onto_score > 0 else 0.0
        
        collab_contrib_abs = alpha * collab_score_norm
        onto_contrib_abs = (1 - alpha) * normalized_onto_score
        
        explanation = f"Te recomendamos '{movie_title}' basado en nuestro análisis." # Default

        # Thresholds for "significant contribution"
        collab_significant = collab_contrib_abs > 0.05 # Avoid tiny scores driving explanation
        onto_significant = onto_contrib_abs > 0.05
        
        shared_feature_str = _create_shared_feature_explanation_fragment(current_movie_details, anchor_movie_details)

        if collab_significant and onto_significant:
            if abs(collab_contrib_abs - onto_contrib_abs) < 0.1 * max(collab_contrib_abs, onto_contrib_abs, 0.1): # Contributions are close
                explanation = f"'{movie_title}' es una buena opción, combinando gustos de usuarios similares y su conexión con '{anchor_title}' ({shared_feature_str})."
            elif collab_contrib_abs > onto_contrib_abs:
                explanation = f"'{movie_title}' destaca entre usuarios con gustos parecidos. Además, tiene relación con '{anchor_title}' ({shared_feature_str})."
            else: # onto_contrib_abs > collab_contrib_abs
                explanation = f"Si te gustó '{anchor_title}', '{movie_title}' podría interesarte ({shared_feature_str}). También es apreciada por otros usuarios."
        elif collab_significant:
            explanation = f"Basado en tu historial y usuarios con gustos similares, '{movie_title}' parece una excelente opción."
        elif onto_significant:
            explanation = f"Dado tu posible interés en '{anchor_title}', te sugerimos '{movie_title}' ({shared_feature_str})."
        
        final_recommendations.append({
            "movieId": movie_id_np, 
            "score": round(float(score_val), 4),
            "explanation": explanation
        })
        seen_movie_ids.add(movie_id_np)
            
    if not final_recommendations and collab_recs_initial: 
        logger.warning(f"Hibridación no generó resultados para {user_id}, usando solo colaborativas.")
        explained_collab_recs = []
        for rec in collab_recs_initial[:top_n]:
            movie_details = await _get_movie_details_internal(rec["movieId"], db_driver)
            movie_title = movie_details.get('title', 'esta película')
            rec["explanation"] = f"Te recomendamos '{movie_title}' porque usuarios con gustos similares a los tuyos también la valoraron positivamente."
            explained_collab_recs.append(rec)
        return {"user_id": user_id, "alpha": alpha, "recommendations": explained_collab_recs, "message": "Fallback a recomendaciones colaborativas."}

    return {"user_id": user_id, "alpha": alpha, "recommendations": final_recommendations}


@app.get("/recommend/new_user/{genre_name}", summary="Recomendaciones para un nuevo usuario basadas en género popular")
async def popular_by_genre_recommend_endpoint(genre_name: str, top_n: int = 10):
    logger.info(f"Solicitud de recomendación para nuevo usuario, género: {genre_name}, top_n: {top_n}")
    db_driver = get_neo4j_driver()

    query = """
    MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
    WHERE g.name = $genre_name
    WITH m
    OPTIONAL MATCH (u:User)-[r:RATED]->(m) 
    WITH m, COUNT(r) AS rating_count 
    RETURN m.movieId AS movieId, m.title as title, rating_count
    ORDER BY rating_count DESC, rand() 
    LIMIT $top_n
    """
    recommendations = []
    try:
        with db_driver.session(database="neo4j") as session:
            results = session.run(query, genre_name=genre_name, top_n=top_n)
            for record in results:
                movie_id = sanitize_value_for_json(record["movieId"])
                title_val = sanitize_value_for_json(record["title"])
                rating_count_val = record["rating_count"]
                
                if movie_id is None: 
                    logger.warning(f"Película con movieId None encontrada para género {genre_name}, omitiendo.")
                    continue

                movie_title_for_explanation = title_val if title_val else "esta película"
                
                explanation = f"'{movie_title_for_explanation}' es muy popular en el género '{genre_name}', con {rating_count_val} valoraciones de usuarios. ¡Podría gustarte!"
                if rating_count_val == 0:
                     explanation = f"'{movie_title_for_explanation}' pertenece al género '{genre_name}'. Aunque aún no tiene muchas valoraciones, podría ser una joya oculta."
                elif rating_count_val < 10: # Example threshold
                     explanation = f"'{movie_title_for_explanation}' es una opción interesante en el género '{genre_name}' con {rating_count_val} valoraciones."


                recommendations.append({
                    "movieId": movie_id,
                    "title": title_val if title_val is not None else "Título no disponible", 
                    "rating_count": rating_count_val,
                    "explanation": explanation
                })
    except Exception as e:
        logger.error(f"Error obteniendo populares por género {genre_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la base de datos: {e}")

    if not recommendations:
        logger.info(f"No se encontraron películas para el género '{genre_name}'.")
        
    return {"genre_name": genre_name, "recommendations": recommendations}


@app.get("/movies/{movie_id}", summary="Obtener detalles de una película")
async def get_movie_details_endpoint(movie_id: str):
    db_driver = get_neo4j_driver()
    movie_details_result = await _get_movie_details_internal(movie_id, db_driver)
    
    # Check if essential movieId is missing or if title indicates an error state from internal function
    if not movie_details_result.get("movieId") or \
       movie_details_result.get("title") == "Película Desconocida" or \
       movie_details_result.get("title") == "Error al cargar detalles":
        
        if movie_details_result.get("title") == "Película Desconocida":
            raise HTTPException(status_code=404, detail=f"Película con ID '{movie_id}' no encontrada.")
        else: # Covers "Error al cargar detalles" or missing movieId due to error
            logger.error(f"Failed to fetch details for movie {movie_id} properly. Data: {movie_details_result}")
            raise HTTPException(status_code=500, detail=f"Error interno al obtener detalles para la película '{movie_id}'.")
            
    return movie_details_result


@app.get("/users", summary="Listar todos los User IDs disponibles")
async def list_users_endpoint():
    global all_user_ids_from_graph
    
    if not all_user_ids_from_graph:
        current_driver = get_neo4j_driver() # Ensure driver is available
        if current_driver:
            logger.info("Lista de User IDs vacía, intentando recargar desde Neo4j...")
            try:
                with current_driver.session(database="neo4j") as session:
                    user_ids_result = session.run("MATCH (u:User) RETURN u.userId AS userId").data()
                    all_user_ids_from_graph = [record['userId'] for record in user_ids_result if record['userId'] is not None]
                    if all_user_ids_from_graph:
                        logger.info(f"Recargados {len(all_user_ids_from_graph)} User IDs desde Neo4j.")
                        save_model_data("all_user_ids", all_user_ids_from_graph) 
                    else:
                        logger.warning("No se encontraron User IDs en Neo4j al recargar.")
            except Exception as e:
                logger.error(f"Error recargando User IDs desde Neo4j: {e}")
        else:
            logger.error("No se pudo obtener el driver de Neo4j para recargar User IDs.")


    if not all_user_ids_from_graph: 
         logger.warning("Lista de User IDs sigue vacía después del intento de recarga.")
         raise HTTPException(status_code=404, detail="No hay IDs de usuario cargados o generados. Revisa los logs de inicio o reentrena.")
    return {"user_ids": all_user_ids_from_graph}

@app.get("/genres", summary="Listar todos los géneros disponibles")
async def list_genres_endpoint():
    db_driver = get_neo4j_driver()
    query = "MATCH (g:Genre) RETURN DISTINCT g.name AS name ORDER BY name"
    genres = []
    try:
        with db_driver.session(database="neo4j") as session:
            results = session.run(query)
            raw_genres = [record["name"] for record in results]
            genres = [sanitize_value_for_json(g) for g in raw_genres]
            genres = [g for g in genres if g is not None] 
    except Exception as e:
        logger.error(f"Error listando géneros: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al listar géneros: {e}")
    
    if not genres:
        logger.info("No se encontraron géneros.")
    return {"genres": genres}
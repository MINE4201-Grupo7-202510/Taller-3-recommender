from py2neo import Graph, Node, Relationship
import pandas as pd
import ast

# === 1. Conexión a Neo4j ===
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    print("✅ Conexión exitosa a Neo4j")
    print(graph.run("RETURN 'conectado' AS estado").data())
except Exception as e:
    print("❌ Error de conexión con Neo4j:", e)
    exit()

# === 2. Cargar archivos CSV ===
try:
    pelis_df = pd.read_csv("peliculas_con_directores_y_actores.csv")
    print("✅ Archivo 'peliculas_con_directores_y_actores.csv' cargado")
except Exception as e:
    print("❌ Error cargando archivo de películas:", e)
    exit()

try:
    base_df = pd.read_csv("final_combined_sample.csv")
    print("✅ Archivo 'final_combined_sample.csv' cargado")
except Exception as e:
    print("❌ Error cargando archivo base:", e)
    exit()

# === 3. Preparar y unir los datos ===
pelis_df['movieId'] = pelis_df['movieId'].astype(str)
base_df['movieId'] = base_df['movieId'].astype(str)

df = pd.merge(base_df, pelis_df[['movieId', 'tconst', 'directors', 'actors']], on='movieId', how='left')

# Eliminar duplicados por combinación usuario/película
df = df.drop_duplicates(subset=['userId', 'movieId'])

# LIMITAR A 1000 FILAS PARA PRUEBA
# df = df.head(1000)

print(f"📊 Filas después del merge y filtro: {len(df)}")
print(df.head(3))

# === 4. Crear nodos y relaciones ===
for i, (_, row) in enumerate(df.iterrows()):
    if i % 100 == 0:
        print(f"➡ Procesando fila {i}/{len(df)}: {row['title']}")

    # Nodo Movie
    movie = Node("Movie",
                 movieId=row['movieId'],
                 title=row['title'],
                 tconst=row['tconst'],
                 genres=row['genres'] if pd.notna(row['genres']) else None)
    graph.merge(movie, "Movie", "movieId")

    # Nodo User
    user = Node("User", userId=str(row['userId']))
    graph.merge(user, "User", "userId")

    # Relación RATED
    if pd.notna(row['rating']):
        rating_rel = Relationship(user, "RATED", movie, score=float(row['rating']))
        graph.merge(rating_rel)

    # Nodo Tag y relación
    if pd.notna(row['tag']):
        tag = Node("Tag", name=row['tag'].strip().lower())
        graph.merge(tag, "Tag", "name")
        graph.merge(Relationship(movie, "TAGGED_AS", tag))

    # Géneros y relación HAS_GENRE
    if pd.notna(row['genres']):
        for genre_name in row['genres'].split('|'):
            genre = Node("Genre", name=genre_name.strip())
            graph.merge(genre, "Genre", "name")
            graph.merge(Relationship(movie, "HAS_GENRE", genre))

    # Actores
    if pd.notna(row['actors']):
        try:
            actors_list = ast.literal_eval(row['actors'])
            for actor_name in actors_list:
                actor = Node("Actor", name=actor_name.strip())
                graph.merge(actor, "Actor", "name")
                graph.merge(Relationship(movie, "ACTED_BY", actor))
        except Exception as e:
            print(f"⚠️ Error al procesar actores de {row['title']}: {e}")

    # Directores
    if pd.notna(row['directors']):
        try:
            directors_list = ast.literal_eval(row['directors'])
            for director_name in directors_list:
                director = Node("Director", name=director_name.strip())
                graph.merge(director, "Director", "name")
                graph.merge(Relationship(movie, "DIRECTED_BY", director))
        except Exception as e:
            print(f"⚠️ Error al procesar directores de {row['title']}: {e}")

print("✅ Carga de grafo completada.")

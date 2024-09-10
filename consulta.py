from app import Embedding, load_embeddings_from_db, get_embedding, SessionLocal
import faiss
import numpy as np

# Función para cargar los embeddings desde la base de datos
def load_embeddings_from_db():
    session = SessionLocal() # abre la sesión
    results = session.query(Embedding.embedding, Embedding.text).all() # obtiene los embeddings y los textos
    session.close()

    embedding_vectors = []
    texts = []
    max_size = 3072  # Tamaño esperado de los embeddings

    for embedding_bytes, text in results:
        # Convierte los bytes a un array NumPy
        buffer = embedding_bytes
        size = len(buffer) # obtiene el tamaño del buffer
        # Verifica que el tamaño del buffer sea múltiplo de 4
        if size % 4 != 0:
            print(f"Buffer size {size} no es múltiplo del tamaño del elemento 4.")
            continue
        # Convierte el buffer a un array NumPy
        embedding_vector = np.frombuffer(buffer, dtype=np.float32)
        # Verifica que el tamaño del embedding sea el esperado
        if embedding_vector.size != max_size:
            print(f"Embedding size {embedding_vector.size} es inconsistente con el tamaño esperado {max_size}.")
            continue
        embedding_vectors.append(embedding_vector) # agrega el embedding al array
        texts.append(text)

    return np.array(embedding_vectors), texts


# Función para crear el índice FAISS y realizar la búsqueda
def search_similar_embeddings(query_text, top_k=5, embedding_model="text-embedding-3-large"):
    # Carga embeddings desde la base de datos
    db_embeddings, texts = load_embeddings_from_db()

    if db_embeddings.size == 0:
        print("No hay embeddings en la base de datos.")
        return

    # Obtener el embedding para el texto de consulta
    query_embedding = get_embedding(query_text, engine=embedding_model)
    if query_embedding is None:
        print("Error al obtener el embedding para la consulta.")
        return

    # Convertir el embedding de la consulta a un array NumPy
    query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    # Crear el índice FAISS
    dimension = query_embedding_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(db_embeddings)

    # Buscar los top_k embeddings más cercanos
    distances, indices = index.search(query_embedding_np, top_k)

    print(f"Resultados para la consulta '{query_text}':")
    for i in range(top_k):
        idx = indices[0][i]
        if idx >= 0:  # Verifica que el índice sea válido
            print(f"Texto: {texts[idx]}")
            print(f"Distancia: {distances[0][i]}")
        else:
            print(f"Índice {idx} es inválido.")


# Ejecutar el script
if __name__ == "__main__":
    query_text = "¿Qué es la regresión lineal?"
    search_similar_embeddings(query_text, top_k=5, embedding_model="text-embedding-3-large")
import os
from dotenv import load_dotenv
import openai
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Text, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
import pdfplumber
import re

load_dotenv()  # Carga el contenido de .env en las variables de entorno

# Configuración del modelo de base de datos
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/embeddings"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Embedding(Base):
    # Define la tabla "embeddings" en la base de datos con los campos correspondientes
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    text = Column(Text)

# Crea las tablas en la base de datos si no existen
Base.metadata.create_all(bind=engine)

# Establece la clave API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Función para generar embeddings
def get_embedding(text, engine="text-embedding-3-large"):
    try:
        response = openai.Embedding.create(model=engine, input=[text]) # obtiene el embedding
        return response["data"][0]["embedding"] # devuelve el embedding
    except openai.error.APIError as e:
        print(f"Error al obtener el embedding: {e}")
        return None

# Función para limpiar el texto
def clean_text(text):
    # Remover caracteres especiales y saltos de línea innecesarios
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_path):
    # Extraer texto de cada página del PDF
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += clean_text(page.extract_text()) or ""
        return text
    except FileNotFoundError:
        print(f"El archivo PDF no se encontró: {pdf_path}")
        return None

# Función para almacenar embeddings en la base de datos
def store_embedding(text, embedding, title="Estadística Descriptiva"):
    if embedding is None:
        return
    # Convertir el embedding a bytes
    db = SessionLocal()
    embedding_np = np.array(embedding, dtype=np.float32) # Convierte a array NumPy
    embedding_bytes = embedding_np.tobytes() # Convierte a bytes
    new_embedding = Embedding(label=title, embedding=embedding_bytes, text=text) # Crea un nuevo objeto Embedding
    db.add(new_embedding) # Agrega el nuevo objeto a la sesión
    db.commit()
    db.refresh(new_embedding) # Actualiza el objeto en la base de datos
    db.close()

# Función principal para procesar un PDF y almacenar los embeddings
def procesar_pdf(pdf_path, embedding_model="text-embedding-3-large"):
    pdf_text = extract_text_from_pdf(pdf_path) # Extrae el texto del PDF
    if pdf_text is None:
        return

    # Divide el texto en párrafos
    paragraphs = pdf_text.split("\n\n")

    # Embedding y almacenamiento en la base de datos
    for paragraph in paragraphs:
        if paragraph.strip():  # Verifica que el párrafo no esté vacío
            embedding = get_embedding(paragraph, engine=embedding_model)
            store_embedding(paragraph, embedding)

    print("Embeddings almacenados correctamente.")

def load_embeddings_from_db(): # carga los embeddings desde la base de datos
    session = SessionLocal() # abre la sesión
    results = session.query(Embedding.embedding, Embedding.text).all() # obtiene los embeddings y los textos
    session.close()

    embedding_vectors = []
    texts = []
    max_size = 1536  # Tamaño esperado de los embeddings

    # Convierte los embeddings a arrays NumPy
    for embedding_bytes, text in results:
        buffer = embedding_bytes
        size = len(buffer)
        # Verifica que el tamaño del buffer sea múltiplo de 4
        if size % 4 != 0:
            print(f"Buffer size {size} no es múltiplo del tamaño del elemento 4.")
            continue
        # Convierte el buffer a un array NumPy
        embedding_vector = np.frombuffer(buffer, dtype=np.float32)
        if embedding_vector.size != max_size:
            print(f"Embedding size {embedding_vector.size} es inconsistente con el tamaño esperado {max_size}.")
            continue
        embedding_vectors.append(embedding_vector) # agrega el embedding al array
        texts.append(text) # agrega el texto al array

    return np.array(embedding_vectors), texts # devuelve los embeddings y los textos


# Ejecutar el script
if __name__ == "__main__":
    # Procesar el archivo PDF
    pdf_path = "./EstadisticaDescriptiva.pdf"
    procesar_pdf(pdf_path, embedding_model="text-embedding-3-large")
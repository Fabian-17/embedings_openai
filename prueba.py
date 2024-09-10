import pandas as pd
import openai
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np

# Configurar la clave de API de OpenAI
openai.api_key = OPENAI_API_KEY='api_key'

# Función para obtener embeddings utilizando la API de OpenAI
def get_embedding(text, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=[text], engine=engine) # metodo para obtener el embedding
    return response['data'][0]['embedding']

# Función para calcular la similitud del coseno
def cosine_similarity(embedding1, embedding2):
    # Convierte los embeddings a arrays NumPy
    embedding1 = np.array(embedding1).reshape(1, -1) # reshape para que tenga la misma forma
    embedding2 = np.array(embedding2).reshape(1, -1)
    return sklearn_cosine_similarity(embedding1, embedding2)[0][0] # calcula la similitud del coseno

# Carga y procesa el PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Divide el contenido en fragmentos más pequeños
    split = CharacterTextSplitter(chunk_size=300, separator='.\n')
    textos = split.split_documents(pages)
    
    # Crea DataFrame con los textos y embeddings
    textos_df = pd.DataFrame([str(i.page_content) for i in textos], columns=["texto"])
    textos_df['Embedding'] = textos_df["texto"].apply(lambda x: get_embedding(x)) # aplica la función get_embedding a cada texto
    return textos_df

# Función para buscar en los textos
def buscar(busqueda, datos, n_resultados=5): # busca en los datos el texto más similar a la busqueda
    busqueda_embed = get_embedding(busqueda) # obtiene el embedding de la busqueda
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed)) # calcula la similitud del coseno
    datos = datos.sort_values("Similitud", ascending=False) # ordena los datos por similitud
    # Devuelve solo el mejor resultado
    mejor_resultado = datos.iloc[:1][["texto", "Similitud"]]
    return mejor_resultado

# Carga y procesa el PDF
pdf_path = './EstadisticaDescriptiva.pdf'
textos_df = process_pdf(pdf_path)

# Interfaz Gradio para la búsqueda
def responder_pregunta(pregunta):
    resultados = buscar(pregunta, textos_df)
    return resultados

with gr.Blocks() as demo:
    busqueda = gr.Textbox(label="Buscar")
    output = gr.DataFrame(headers=["Texto", "Similitud"])
    greet_btn = gr.Button("Preguntar")
    greet_btn.click(fn=responder_pregunta, inputs=busqueda, outputs=output)

demo.launch()
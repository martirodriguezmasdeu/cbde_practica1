#C0
import os
import time
import numpy as np
import chromadb
from chromadb.config import Settings

from datasets import load_dataset

# --- Carrega el mateix dataset i chunks que a P0, per garantir consistència ---

dataset = load_dataset("bctnry/BookCorpus",
    split="train",
    streaming=True
)

dataset = dataset.take(10_000)
sentences = [example["text"] for example in dataset]

chunk_size = 2000

chunks = [
    sentences[i:i + chunk_size]
    for i in range(0, len(sentences), chunk_size)
]

# --- Client persistent de Chroma (guarda les dades a disc, a la carpeta indicada) ---

client = chromadb.PersistentClient(path="./chroma_data")

# Elimina la col·lecció si ja existia, per començar de zero (equivalent al DROP TABLE de P0)
try:
    client.delete_collection("bookCorpus")
except Exception:
    pass

# embedding_function=None: NO volem que Chroma calculi embeddings automàticament aquí.
# Volem mesurar només el cost d'inserir el TEXT, i deixar els embeddings reals per C1.
collection = client.create_collection(
    name="bookCorpus",
    embedding_function=None
)

# Dimensió del model que farem servir a C1 (all-MiniLM-L6-v2 -> 384)
EMBEDDING_DIM = 384

# Guardar el temps de cada inserció
text_times = []

sentence_id = 0

for chunk in chunks:
    ids = [str(sentence_id + i) for i in range(len(chunk))]
    # Embeddings "placeholder" (vector de zeros): només per satisfer l'API de Chroma,
    # que exigeix un embedding per cada document afegit a la col·lecció.
    placeholder_embeddings = [[0.0] * EMBEDDING_DIM for _ in chunk]

    start = time.perf_counter()

    collection.add(
        ids=ids,
        documents=chunk,
        embeddings=placeholder_embeddings
    )

    end = time.perf_counter()

    text_times.append(end - start)
    sentence_id += len(chunk)

# Mostrar estadístiques (mateix format que P0, per comparar directament)
print("--- Temps d'inserció del text (Chroma) ---")
print(f"Mínim: {np.min(text_times):.6f} s")
print(f"Màxim: {np.max(text_times):.6f} s")
print(f"Mitjana: {np.mean(text_times):.6f} s")
print(f"Desviació estàndard: {np.std(text_times):.6f} s")
print(f"Temps total: {np.sum(text_times):.6f} s")

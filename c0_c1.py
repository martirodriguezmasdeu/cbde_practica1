#C0 + C1 (combinats)
#
# Decisió de disseny: a Chroma, l'API collection.add() amb una embedding_function
# activa insereix el text I calcula/emmagatzema l'embedding en una sola operació
# atòmica. No hi ha manera nativa de separar-ho (a diferència de PostgreSQL, on
# text i embeddings viuen en dues insercions independents). Per això mesurem el
# temps d'aquesta operació combinada, i ho justifiquem a la memòria com un
# exemple concret d'impedance mismatch: el model relacional permet aquesta
# separació de manera natural, el model de Chroma no.

import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

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

# --- Client persistent de Chroma ---

client = chromadb.PersistentClient(path="./chroma_data")

# Elimina la col·lecció si ja existia, per començar de zero
try:
    client.delete_collection("bookCorpus")
except Exception:
    pass

# Funció d'embeddings: mateix model que a PostgreSQL (all-MiniLM-L6-v2),
# per poder comparar de manera justa entre els dos sistemes.
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="bookCorpus",
    embedding_function=sentence_transformer_ef
)

# Guardar el temps de cada inserció (text + embedding junts)
insert_times = []

sentence_id = 0

for chunk in chunks:
    ids = [str(sentence_id + i) for i in range(len(chunk))]

    start = time.perf_counter()

    collection.add(
        ids=ids,
        documents=chunk
        # No cal passar 'embeddings': Chroma els calcula automàticament
        # amb la sentence_transformer_ef que hem definit a la col·lecció.
    )

    end = time.perf_counter()

    insert_times.append(end - start)
    sentence_id += len(chunk)

# Mostrar estadístiques
print("--- Temps d'inserció de text + embeddings (Chroma) ---")
print(f"Mínim: {np.min(insert_times):.6f} s")
print(f"Màxim: {np.max(insert_times):.6f} s")
print(f"Mitjana: {np.mean(insert_times):.6f} s")
print(f"Desviació estàndard: {np.std(insert_times):.6f} s")
print(f"Temps total: {np.sum(insert_times):.6f} s")
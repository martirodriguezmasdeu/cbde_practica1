#C2
#
# Reutilitza la col·lecció creada a c0_c1.py (persistent a ./chroma_data).
#
# NOTA IMPORTANT sobre disseny: Chroma només admet com a mètrica NATIVA
# d'indexació (hnsw:space) "l2", "ip" o "cosine". NO admet Manhattan.
# Per poder comparar exactament les mateixes dues mètriques que a P2
# (euclidiana i Manhattan), fem el mateix que a PostgreSQL: portem tots els
# embeddings a memòria i calculem les distàncies manualment amb numpy.
# Això és, de fet, un exemple clar d'impedance mismatch en sentit contrari:
# fins i tot una base de dades vectorial "nativa" com Chroma obliga a un
# fallback manual/brute-force quan necessites una mètrica que el seu índex
# no suporta.

import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="bookCorpus",
    embedding_function=sentence_transformer_ef
)

# --- Recupera totes les dades (mateix esperit que el SELECT ... ORDER BY id de P2) ---

all_data = collection.get(include=["documents", "embeddings"])

# Els ids de Chroma són strings ("0", "1", ...); els ordenem numèricament
# perquè coincideixin amb l'ordre original d'inserció (mateix ordre que a P2).
order = np.argsort([int(i) for i in all_data["ids"]])

ids = [all_data["ids"][i] for i in order]
sentences = [all_data["documents"][i] for i in order]
embeddings = np.array([all_data["embeddings"][i] for i in order])

# Escull les 10 frases inicials (mateixos índexs que a P2)
query_indexes = list(range(10))

# --- Cas distància euclidiana ---
euclidean_times = []

for index in query_indexes:

    start = time.perf_counter()

    query_embedding = embeddings[index]
    distances = np.linalg.norm(
        embeddings - query_embedding,
        axis=1
    )

    # No compara la frase amb si mateixa
    distances[index] = np.inf

    top2_idx = np.argpartition(distances, 2)[:2]
    top2_euclidean = top2_idx[np.argsort(distances[top2_idx])]

    end = time.perf_counter()

    euclidean_times.append(end - start)

    print(f"\nFrase seleccionada: ID {int(ids[index]) + 1}: {sentences[index]}")    # ID {int(ids[index]) + 1} per igualar amb P2 (que té id SERIAL començant a 1)
    print("Top-2 frases amb distància euclidiana:")

    for i in top2_euclidean:
        print(f"  ID {int(ids[i]) + 1} | distancia = {distances[i]:.4f} | {sentences[i]}")

# --- Cas distància de Manhattan ---
manhattan_times = []

for index in query_indexes:

    start = time.perf_counter()

    query_embedding = embeddings[index]

    distances = np.sum(
        np.abs(embeddings - query_embedding),
        axis=1
    )

    # No compara la frase amb si mateixa
    distances[index] = np.inf

    top2_idx = np.argpartition(distances, 2)[:2]
    top2_manhattan = top2_idx[np.argsort(distances[top2_idx])]

    end = time.perf_counter()

    manhattan_times.append(end - start)

    print(f"\nFrase seleccionada: ID {int(ids[index]) + 1}: {sentences[index]}")
    print("Top-2 frases amb distància de Manhattan:")

    for i in top2_manhattan:
        print(f"  ID {int(ids[i]) + 1} | distancia = {distances[i]:.4f} | {sentences[i]}")

# --- Estadístiques de temps ---

print("\n--- Temps distància euclidiana (Chroma) ---")
print(f"Mínim: {np.min(euclidean_times):.6f} s")
print(f"Màxim: {np.max(euclidean_times):.6f} s")
print(f"Mitjana: {np.mean(euclidean_times):.6f} s")
print(f"Desviació estàndard: {np.std(euclidean_times):.6f} s")
print(f"Temps total: {np.sum(euclidean_times):.6f} s")

print("\n--- Temps distància de Manhattan (Chroma) ---")
print(f"Mínim: {np.min(manhattan_times):.6f} s")
print(f"Màxim: {np.max(manhattan_times):.6f} s")
print(f"Mitjana: {np.mean(manhattan_times):.6f} s")
print(f"Desviació estàndard: {np.std(manhattan_times):.6f} s")
print(f"Temps total: {np.sum(manhattan_times):.6f} s")
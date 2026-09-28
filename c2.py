#C2


import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions


#Carrega la base de dades
client = chromadb.PersistentClient(path="./chroma_data")


#Carrega la funció per crear embeddings amb el model corresponent
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Recupera la col·lecció de frases i embeddings
collection = client.get_collection(
    name="bookCorpus",
    embedding_function=sentence_transformer_ef
)

all_data = collection.get(include=["documents", "embeddings"])

#Perquè els ids no estan ordenats, els ordenem per poder comparar amb P2
order = np.argsort([int(i) for i in all_data["ids"]])

ids = [all_data["ids"][i] for i in order]
sentences = [all_data["documents"][i] for i in order]
embeddings = np.array([all_data["embeddings"][i] for i in order])

#Escull les 10 frases inicials
query_indexes = list(range(10))

#Cas distància euclidiana
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

    top2_idx = np.argpartition(distances, 2)[:2]                #Cost O(n), si s'utilitzés np.argsort seria O(n log n)
    top2_euclidean = top2_idx[np.argsort(distances[top2_idx])]

    end = time.perf_counter()

    euclidean_times.append(end - start)

    #{int(ids[index]) + 1} per igualar amb P2 (que té id SERIAL començant a 1)
    print(f"\nFrase seleccionada: ID {int(ids[index]) + 1}: {sentences[index]}")
    print("Top-2 frases amb distància euclidiana:")

    for i in top2_euclidean:
        print(f"  ID {int(ids[i]) + 1} | distancia = {distances[i]:.4f} | {sentences[i]}")


#Cas distància de Manhattan
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

    top2_idx = np.argpartition(distances, 2)[:2]                    #Cost O(n), si s'utilitzés np.argsort seria O(n log n)
    top2_manhattan = top2_idx[np.argsort(distances[top2_idx])]

    end = time.perf_counter()

    manhattan_times.append(end - start)

    #{int(ids[index]) + 1} per igualar amb P2 (que té id SERIAL començant a 1)
    print(f"\nFrase seleccionada: ID {int(ids[index]) + 1}: {sentences[index]}")
    print("Top-2 frases amb distància de Manhattan:")

    for i in top2_manhattan:
        print(f"  ID {int(ids[i]) + 1} | distancia = {distances[i]:.4f} | {sentences[i]}")


#Mostra estadístiques

print("\n--- Temps distància euclidiana (Chroma) ---")
print(f"Mínim: {np.min(euclidean_times):.6f} s")
print(f"Màxim: {np.max(euclidean_times):.6f} s")
print(f"Mitjana: {np.mean(euclidean_times):.6f} s")
print(f"Desviació estàndard: {np.std(euclidean_times):.6f} s")
print(f"Temps total: {np.sum(euclidean_times):.6f} s")             #Afegit per obtenir més informació sobre els temps

print("\n--- Temps distància de Manhattan (Chroma) ---")
print(f"Mínim: {np.min(manhattan_times):.6f} s")
print(f"Màxim: {np.max(manhattan_times):.6f} s")
print(f"Mitjana: {np.mean(manhattan_times):.6f} s")
print(f"Desviació estàndard: {np.std(manhattan_times):.6f} s")
print(f"Temps total: {np.sum(manhattan_times):.6f} s")             #Afegit per obtenir més informació sobre els temps

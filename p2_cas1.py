# P2 Cas 1

import psycopg2
import time
import numpy as np

connection_string = "postgresql://neondb_owner:npg_I0LgpBV8PRDz@ep-patient-salad-b2n9m6fn-pooler.c-6.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

conn = psycopg2.connect(connection_string)
cursor = conn.cursor()

#Agafa els embeddings de la base de dades
cursor.execute("SELECT id, sentence, embedding FROM bookCorpus_embeddings ORDER BY id")

rows = cursor.fetchall()
ids = [row[0] for row in rows]
sentences = [row[1] for row in rows]
embeddings = np.array([
    row[2] for row in rows
])

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

    top2_idx = np.argpartition(distances, 2)[:2]
    top2_euclidean = top2_idx[np.argsort(distances[top2_idx])]

    end = time.perf_counter()

    euclidean_times.append(end - start)

    print(f"\nFrase seleccionada: ID {ids[index]}: {sentences[index]}")

    print("Top-2 frases amb distància euclidiana:")

    for i in top2_euclidean:
        print(f"  ID {ids[i]} | "f"distancia = {distances[i]:.4f} | "f"{sentences[i]}")

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

    top2_idx = np.argpartition(distances, 2)[:2]
    top2_manhattan = top2_idx[np.argsort(distances[top2_idx])]

    end = time.perf_counter()

    manhattan_times.append(end - start)

    print(f"\nFrase seleccionada: ID {ids[index]}: {sentences[index]}")

    print("Top-2 frases amb distància de Manhattan:")

    for i in top2_manhattan:
        print(
            f"  ID {ids[i]} | "
            f"distancia = {distances[i]:.4f} | "
            f"{sentences[i]}"
        )

print("\n--- Temps distància euclidiana ---")

print(f"Mínim: {np.min(euclidean_times):.6f} s")
print(f"Màxim: {np.max(euclidean_times):.6f} s")
print(f"Mitjana: {np.mean(euclidean_times):.6f} s")
print(f"Desviació estàndard: {np.std(euclidean_times):.6f} s")
print(f"Temps total: {np.sum(euclidean_times):.6f} s")

print("\n--- Temps distància de Manhattan ---")

print(f"Mínim: {np.min(manhattan_times):.6f} s")
print(f"Màxim: {np.max(manhattan_times):.6f} s")
print(f"Mitjana: {np.mean(manhattan_times):.6f} s")
print(f"Desviació estàndard: {np.std(manhattan_times):.6f} s")
print(f"Temps total: {np.sum(manhattan_times):.6f} s")
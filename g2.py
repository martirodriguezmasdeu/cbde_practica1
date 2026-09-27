#G2
import os
from dotenv import load_dotenv

load_dotenv()

import psycopg2
import time
import numpy as np
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432')
)

register_vector(conn)
cursor = conn.cursor()

#Agafa els ids i sentences per poder imprimir els resultats igual que a P2
cursor.execute("SELECT id, sentence FROM bookCorpus_pgvector ORDER BY id;")
rows = cursor.fetchall()
ids = [row[0] for row in rows]
sentences = [row[1] for row in rows]

#Escull les 10 frases inicials (mateixos índexs que P2 i C2)
query_indexes = list(range(10))

# --- Cas distància euclidiana (operador natiu <->) ---
euclidean_times = []

for index in query_indexes:
    query_id = ids[index]

    start = time.perf_counter()

    cursor.execute(
        """
        SELECT id, sentence, embedding <-> (SELECT embedding FROM bookCorpus_pgvector WHERE id = %s) AS dist
        FROM bookCorpus_pgvector
        WHERE id != %s
        ORDER BY dist
        LIMIT 2;
        """,
        (query_id, query_id)
    )
    top2_euclidean = cursor.fetchall()

    end = time.perf_counter()

    euclidean_times.append(end - start)

    print(f"\nFrase seleccionada: ID {query_id}: {sentences[index]}")
    print("Top-2 frases amb distància euclidiana (pgvector natiu <->):")

    for rid, rsentence, rdist in top2_euclidean:
        print(f"  ID {rid} | distancia = {rdist:.4f} | {rsentence}")

# --- Cas distància de Manhattan (operador natiu <+>, requereix pgvector >= 0.7.0) ---
manhattan_times = []

for index in query_indexes:
    query_id = ids[index]

    start = time.perf_counter()

    cursor.execute(
        """
        SELECT id, sentence, embedding <+> (SELECT embedding FROM bookCorpus_pgvector WHERE id = %s) AS dist
        FROM bookCorpus_pgvector
        WHERE id != %s
        ORDER BY dist
        LIMIT 2;
        """,
        (query_id, query_id)
    )
    top2_manhattan = cursor.fetchall()

    end = time.perf_counter()

    manhattan_times.append(end - start)

    print(f"\nFrase seleccionada: ID {query_id}: {sentences[index]}")
    print("Top-2 frases amb distància de Manhattan (pgvector natiu <+>):")

    for rid, rsentence, rdist in top2_manhattan:
        print(f"  ID {rid} | distancia = {rdist:.4f} | {rsentence}")

# --- Estadístiques de temps ---

print("\n--- Temps distància euclidiana (pgvector) ---")
print(f"Mínim: {np.min(euclidean_times):.6f} s")
print(f"Màxim: {np.max(euclidean_times):.6f} s")
print(f"Mitjana: {np.mean(euclidean_times):.6f} s")
print(f"Desviació estàndard: {np.std(euclidean_times):.6f} s")
print(f"Temps total: {np.sum(euclidean_times):.6f} s")

print("\n--- Temps distància de Manhattan (pgvector) ---")
print(f"Mínim: {np.min(manhattan_times):.6f} s")
print(f"Màxim: {np.max(manhattan_times):.6f} s")
print(f"Mitjana: {np.mean(manhattan_times):.6f} s")
print(f"Desviació estàndard: {np.std(manhattan_times):.6f} s")
print(f"Temps total: {np.sum(manhattan_times):.6f} s")

cursor.close()
conn.close()
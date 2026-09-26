#P1 Cas 3


import time
import numpy as np


#Carregar el model
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#Carregar la base de dades
import psycopg2
from psycopg2.extras import execute_values


connection_string = "postgresql://neondb_owner:npg_I0LgpBV8PRDz@ep-patient-salad-b2n9m6fn-pooler.c-6.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


conn = psycopg2.connect(connection_string)
cursor = conn.cursor()


#Agafa totes les frases amb els seus ids
cursor.execute("SELECT id, sentence FROM bookCorpus ORDER BY id;")


rows = cursor.fetchall()
ids = [row[0] for row in rows]
sentences = [row[1] for row in rows]


#Crea els embeddings i els afegeix a la base de dades
embeddings = model.encode(sentences)


cursor.execute("ALTER TABLE bookCorpus ADD COLUMN embedding REAL[];")


values = [
    (id_, embedding.tolist())
    for id_, embedding in zip(ids, embeddings)
]


# Insereix per chunks i mesura el temps de cada inserció
chunk_size = 2000
embedding_times = []


for i in range(0, len(values), chunk_size):
    chunk = values[i:i + chunk_size]


    start = time.perf_counter()


    execute_values(
        cursor,
        "UPDATE bookCorpus AS b SET embedding = v.embedding FROM (VALUES %s) AS v(id, embedding) WHERE b.id = v.id",
        chunk
    )
    conn.commit()


    end = time.perf_counter()


    embedding_times.append(end - start)


# Mostrar estadístiques
print("--- Temps d'inserció dels embeddings ---")
print(f"Mínim: {np.min(embedding_times):.6f} s")
print(f"Màxim: {np.max(embedding_times):.6f} s")
print(f"Mitjana: {np.mean(embedding_times):.6f} s")
print(f"Desviació estàndard: {np.std(embedding_times):.6f} s")
print(f"Temps total: {np.sum(embedding_times):.6f} s")
cursor.close()
conn.close()
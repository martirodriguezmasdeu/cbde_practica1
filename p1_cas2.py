#P1 Cas 2


import os
from dotenv import load_dotenv

load_dotenv()

import time
import numpy as np


#Carregar el model
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#Carregar la base de dades
import psycopg2
from psycopg2.extras import execute_values


conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432')
)
cursor = conn.cursor()


#Agafa totes les frases amb els seus ids
cursor.execute("SELECT id, sentence FROM bookCorpus ORDER BY id;")


rows = cursor.fetchall()
ids = [row[0] for row in rows]
sentences = [row[1] for row in rows]


#Crea els embeddings i els afegeix a la base de dades
embeddings = model.encode(sentences)


cursor.execute("DROP TABLE IF EXISTS bookCorpus_embeddings;")
cursor.execute("CREATE TABLE bookCorpus_embeddings (id INTEGER NOT NULL UNIQUE, embedding REAL[] NOT NULL, FOREIGN KEY (id) REFERENCES bookCorpus(id));")


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
        "INSERT INTO bookCorpus_embeddings (id, embedding) VALUES %s",
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
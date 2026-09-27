#G1
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
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432')
)

# Registra el tipus 'vector' perquè psycopg2 sàpiga convertir arrays de
# numpy directament al format natiu de pgvector (a diferència de P1, on
# guardàvem els embeddings com REAL[] genèric)
register_vector(conn)

cursor = conn.cursor()

#Agafa totes les frases amb els seus ids
cursor.execute("SELECT id, sentence FROM bookCorpus_pgvector ORDER BY id;")

rows = cursor.fetchall()
ids = [row[0] for row in rows]
sentences = [row[1] for row in rows]

#Crea els embeddings
embeddings = model.encode(sentences)

# Columna de tipus 'vector' natiu de pgvector, amb la dimensió exacta
# del model (all-MiniLM-L6-v2 -> 384)
# Si ja existia d'una execució anterior, l'esborrem primer per poder
# tornar a executar aquest script sense haver de recrear tot bookCorpus_pgvector
embedding_dim = embeddings.shape[1]
cursor.execute("ALTER TABLE bookCorpus_pgvector DROP COLUMN IF EXISTS embedding;")
cursor.execute(f"ALTER TABLE bookCorpus_pgvector ADD COLUMN embedding vector({embedding_dim});")
conn.commit()

values = [
    (id_, embedding)
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
        "UPDATE bookCorpus_pgvector AS b SET embedding = v.embedding::vector FROM (VALUES %s) AS v(id, embedding) WHERE b.id = v.id",
        chunk
    )
    conn.commit()

    end = time.perf_counter()

    embedding_times.append(end - start)

# Mostrar estadístiques
print("--- Temps d'inserció dels embeddings (pgvector) ---")
print(f"Mínim: {np.min(embedding_times):.6f} s")
print(f"Màxim: {np.max(embedding_times):.6f} s")
print(f"Mitjana: {np.mean(embedding_times):.6f} s")
print(f"Desviació estàndard: {np.std(embedding_times):.6f} s")
print(f"Temps total: {np.sum(embedding_times):.6f} s")

cursor.close()
conn.close()
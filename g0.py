#G0
import os
from dotenv import load_dotenv

load_dotenv()

import time
import numpy as np

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

# Activa l'extensió pgvector (si no ho estava ja) en aquesta base de dades
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

from datasets import load_dataset

# Llegeix el dataset de BookCorpus progressivament (mateix corpus que P0/C0)
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

# Taula NOVA i diferent de la de P0, per no barrejar resultats
cursor.execute("DROP TABLE IF EXISTS bookCorpus_pgvector CASCADE;")
cursor.execute("CREATE TABLE bookCorpus_pgvector (id SERIAL PRIMARY KEY, sentence TEXT NOT NULL);")
conn.commit()

# Guardar el temps de cada inserció
text_times = []

for chunk in chunks:
    values = [(sentence,) for sentence in chunk]

    start = time.perf_counter()

    execute_values(
        cursor,
        "INSERT INTO bookCorpus_pgvector (sentence) VALUES %s",
        values
    )

    conn.commit()

    end = time.perf_counter()

    text_times.append(end - start)

# Mostrar estadístiques
print("--- Temps d'inserció del text (pgvector) ---")
print(f"Mínim: {np.min(text_times):.6f} s")
print(f"Màxim: {np.max(text_times):.6f} s")
print(f"Mitjana: {np.mean(text_times):.6f} s")
print(f"Desviació estàndard: {np.std(text_times):.6f} s")
print(f"Temps total: {np.sum(text_times):.6f} s")

cursor.close()
conn.close()
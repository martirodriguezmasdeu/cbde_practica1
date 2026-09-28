#P0


import os
from dotenv import load_dotenv

load_dotenv()

import time
import numpy as np

#Carrega la base de dades
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


#Carrega el dataset de BookCorpus progressivament (streaming=True)
from datasets import load_dataset

dataset = load_dataset("bctnry/BookCorpus", split="train", streaming=True)


#Només agafa 10.000 frases que correspon als 10.000 registres
dataset = dataset.take(10_000)

sentences = [example["text"] for example in dataset]


#Crea "chunks" de 2.000 frases
chunk_size = 2000

chunks = [
    sentences[i:i + chunk_size]
      for i in range(0, len(sentences), chunk_size)
      ]


#Crea la nova taula de bookCorpus
cursor.execute("DROP TABLE IF EXISTS bookCorpus CASCADE;")
cursor.execute("CREATE TABLE bookCorpus (id SERIAL PRIMARY KEY, sentence TEXT NOT NULL);")
conn.commit()


#Guarda el temps de cada inserció
text_times = []

#Afegeix les frases a la taula
for chunk in chunks:
    values = [(sentence,) for sentence in chunk]

    start = time.perf_counter()

    execute_values(
        cursor,
        "INSERT INTO bookCorpus (sentence) VALUES %s",
        values
    )

    conn.commit()

    end = time.perf_counter()

    text_times.append(end - start)


#Mostra estadístiques
print("--- Temps d'inserció del text ---")
print(f"Mínim: {np.min(text_times):.6f} s")
print(f"Màxim: {np.max(text_times):.6f} s")
print(f"Mitjana: {np.mean(text_times):.6f} s")
print(f"Desviació estàndard: {np.std(text_times):.6f} s")
print(f"Temps total: {np.sum(text_times):.6f} s")           #Afegit per obtenir més informació sobre els temps

cursor.close()
conn.close()

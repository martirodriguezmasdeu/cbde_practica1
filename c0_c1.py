#C0 + C1

import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions


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


#Carrega la base de dades
client = chromadb.PersistentClient(path="./chroma_data")

try:
    client.delete_collection("bookCorpus")
except Exception:
    pass


#Carrega la funció per crear embeddings amb el model corresponent
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Crea la col·leció frase i embedding
collection = client.create_collection(
    name="bookCorpus",
    embedding_function=sentence_transformer_ef
)

#Guarda el temps de cada inserció
insert_times = []

sentence_id = 0

#Insereix les frases i embeddings a la col·lecció per chunks
#i mesura el temps de cada inserció
for chunk in chunks:
    ids = [str(sentence_id + i) for i in range(len(chunk))]

    start = time.perf_counter()

    collection.add(
        ids=ids,
        documents=chunk
    )

    end = time.perf_counter()

    insert_times.append(end - start)
    sentence_id += len(chunk)


#Mostra estadístiques
print("--- Temps d'inserció de text + embeddings (Chroma) ---")
print(f"Mínim: {np.min(insert_times):.6f} s")
print(f"Màxim: {np.max(insert_times):.6f} s")
print(f"Mitjana: {np.mean(insert_times):.6f} s")
print(f"Desviació estàndard: {np.std(insert_times):.6f} s")
print(f"Temps total: {np.sum(insert_times):.6f} s")                 #Afegit per obtenir més informació sobre els temps

from huggingface_hub import notebook_login

notebook_login()

pip install torch

pip install -U transformers datasets evaluate accelerate timm
pip install -U sentence-transformers

pip install -U psycopg2-binary
pip freeze > requirements.txt

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import psycopg2

connection_string = "postgresql://neondb_owner:npg_I0LgpBV8PRDz@ep-patient-salad-b2n9m6fn-pooler.c-6.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

conn = psycopg2.connect(connection_string)

from datasets import load_dataset

# Read data progressively
dataset = load_dataset(
    "bctnry/BookCorpus",
    split="train",
    streaming=True
)

# Keep only 10,000
dataset = dataset.take(10_000)
sentences = [example["text"] for example in dataset]

# Create chunks of 2,000 sentences
chunk_size = 2000

chunks = [
    sentences[i:i + chunk_size]
      for i in range(0, len(sentences), chunk_size)
      ]


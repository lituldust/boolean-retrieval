#%% Import Library
import pandas as pd
import pyserini
from pyserini.analysis import Analyzer, get_lucene_analyzer
import json
# %%
# List dari file corpus-assignment#1.txt
docs = [
  ("d1",  "The cat chased a small mouse into the garden."),
  ("d2",  "A friendly dog played fetch by the river."),
  ("d3",  "BM25 is a ranking function widely used in search engines."),
  ("d4",  "Boolean retrieval uses logical operators like AND and OR."),
  ("d5",  "TF-IDF weights terms by frequency and rarity."),
  ("d6",  "Neural retrieval uses dense embeddings for semantic search."),
  ("d7",  "The dog and the cat slept on the same couch."),
  ("d8",  "The library hosts a workshop on information retrieval."),
  ("d9",  "Students implemented BM25 and compared it with TF-IDF."),
  ("d10", "The chef roasted chicken with rosemary and garlic."),
  ("d11", "A black cat crossed the old stone bridge at night."),
  ("d12", "Dogs are loyal companions during long hikes."),
  ("d13", "The dataset contains fifteen short sentences for testing."),
  ("d14", "Reranking models reorder BM25 candidates using transformers."),
  ("d15", "The dog sniffed a cat but ignored the mouse.")
]
# %% Mengubah List Menjadi Dataframe
df = pd.DataFrame(docs, columns=['doc_id', 'doc_text'])
df
# %% Preprocessing
# Nanti isi proses preprocessing pake Analyzer dari pyserini aja
# %% Indexing - Make .jsonl file
output_file = 'index.jsonl'

with open(output_file, 'w') as f:
    for index, row in df.iterrows():
        json_record={
            "id": row['doc_id'],
            "contents": row['doc_text']
        }
        json_string = json.dumps(json_record)
        f.write(json_string + '\n')
# %% Indexing - 

# %% Boolean Retrieval

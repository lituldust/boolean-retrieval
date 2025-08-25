#%% Import Library
import pandas as pd
import nltk
import json
import os
import string
# %%
nltk.download('stopwords')  # Download stopwords jika belum terinstall
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
df['doc_text'] = df['doc_text'].apply(lambda x: x.lower())  # Lowercase
df['doc_text'] = df['doc_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in nltk.corpus.stopwords.words('english')])) # Remove Stopword
df['doc_text'] = df['doc_text'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation])) # Remove Punctuation

# Cek kembali df setelah preprocessing
df
# %% Buat file .jsonl untuk proses indexing
output_file = 'test/index.jsonl'
output_dir = os.path.dirname(output_file)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

with open(output_file, 'w') as f:
    for index, row in df.iterrows():
        json_record={
            "id": row['doc_id'],
            "contents": row['doc_text']
        }
        json_string = json.dumps(json_record)
        f.write(json_string + '\n')
#%% Indexing + stemming (VsCode)
'''
Default Indexing dari pyserini tapi tambahin stemming, jalanin di terminal

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {path ke file index.jsonl} \
  --index indexes/sample_collection_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --stemmer porter \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
'''
#%% Indexing + stemming (Google Colab)
'''
!python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {path ke file index.jsonl} \
  --index indexes/sample_collection_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --stemmer porter \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
'''
# %% Boolean Retrieval
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/sample_collection_jsonl')

queries = [
  'dog AND cat',
  'dog OR cat',
  'dog AND NOT cat',
  '(bm25 OR tf-idf) AND retrieval',
  'dog OR (cat AND mouse)',
  'retrieval AND (neural OR bm25)'
]

# %% Print hasil pencarian
for query in queries:
  hits = searcher.search(query)
  print(f'\n{query}:')
  print('  ', 'Id', ' ', 'Teks')

  for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f} {df[df['doc_id'] == hits[i].docid]['doc_text'].values[0]}')
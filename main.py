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
from pyserini.search.lucene import LuceneSearcher, querybuilder

searcher = LuceneSearcher('indexes/sample_collection_jsonl')

queries = [
  'dog AND cat',
  'dog OR cat',
  'dog AND NOT cat',
  '(bm25 OR tf-idf) AND retrieval',
  'dog OR (cat AND mouse)',
  'retrieval AND (neural OR bm25)'
]

# %% Definisikan Logical Operator
should = querybuilder.JBooleanClauseOccur['should'].value
must = querybuilder.JBooleanClauseOccur['must'].value
must_not = querybuilder.JBooleanClauseOccur['must_not'].value
# %% Fungsi untuk print hasil pencarian
def display_results(query, hits):
  print(f'\n{query}:')

  if not hits:
      print("Tidak ada dokumen yang sesuai.")
      return
  
  print('  ', 'Id', ' ', 'Score', ' ', 'Teks')
  
  for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f} {df[df['doc_id'] == hits[i].docid]['doc_text'].values[0]}')
  print("-"*100)
# %% Query 1 - dog AND cat
builder_1 = querybuilder.get_boolean_query_builder()
builder_1.add(querybuilder.get_term_query("dog"), must)
builder_1.add(querybuilder.get_term_query("cat"), must)
query_1 = builder_1.build()
hits_1 = searcher.search(query_1)
# %% Query 2 - dog OR cat
builder_2 = querybuilder.get_boolean_query_builder()
builder_2.add(querybuilder.get_term_query("dog"), should)
builder_2.add(querybuilder.get_term_query("cat"), should)
query_2 = builder_2.build()
hits_2 = searcher.search(query_2)
# %% Query 3 - dog AND NOT cat
builder_3 = querybuilder.get_boolean_query_builder()
builder_3.add(querybuilder.get_term_query("dog"), should)
builder_3.add(querybuilder.get_term_query("cat"), must_not)
query_3 = builder_3.build()
hits_3 = searcher.search(query_3)

# %% Query 4 - (bm25 OR tf-idf) AND retrieval
inner_builder_4 = querybuilder.get_boolean_query_builder()
inner_builder_4.add(querybuilder.get_term_query("bm25"), should)
inner_builder_4.add(querybuilder.get_term_query("tf-idf"), should)
inner_query_4 = inner_builder_4.build()

outer_builder_4 = querybuilder.get_boolean_query_builder()
outer_builder_4.add(inner_query_4, must)
outer_builder_4.add(querybuilder.get_term_query("retrieval"), must)
query_4 = outer_builder_4.build()
hits_4 = searcher.search(query_4)

# %% Query 5 - dog OR (cat AND mouse)
inner_builder_5 = querybuilder.get_boolean_query_builder()
inner_builder_5.add(querybuilder.get_term_query("cat"), must)
inner_builder_5.add(querybuilder.get_term_query("mouse"), must)
inner_query_5 = inner_builder_5.build()

outer_builder_5 = querybuilder.get_boolean_query_builder()
outer_builder_5.add(inner_query_5, should)
outer_builder_5.add(querybuilder.get_term_query("dog"), should)
query_5 = outer_builder_5.build()
hits_5 = searcher.search(query_5)

# %% Query 6 - retrieval AND (neural OR bm25)
inner_builder_6 = querybuilder.get_boolean_query_builder()
inner_builder_6.add(querybuilder.get_term_query("neural"), should)
inner_builder_6.add(querybuilder.get_term_query("bm25"), should)
inner_query_6 = inner_builder_6.build()

outer_builder_6 = querybuilder.get_boolean_query_builder()
outer_builder_6.add(inner_query_6, must)
outer_builder_6.add(querybuilder.get_term_query("retrieval"), must)
query_6 = outer_builder_6.build()
hits_6 = searcher.search(query_6)

# %% Tampilkan hasil
hits = [
   hits_1,
   hits_2,
   hits_3,
   hits_4,
   hits_5,
   hits_6
]

for i in range(len(queries)):
   display_results(queries[i], hits[i])

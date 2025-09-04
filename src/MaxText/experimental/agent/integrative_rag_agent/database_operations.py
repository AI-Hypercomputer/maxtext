# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file defines functions for managing a SQLite database to store and retrieve
code-related documents and their embeddings for a Retrieval-Augmented Generation (RAG)
system. It includes functionalities for:
- Initializing the database and creating a `documents` table.
- Saving new documents along with their names, text content, descriptions,
  file paths, and embedding vectors.
- Loading all stored documents and their associated metadata and embeddings.
- Building a FAISS (Facebook AI Similarity Search) index for efficient similarity
  search over the document embeddings.
- Performing similarity searches to find the most relevant documents based on a
  given query embedding.
"""

import sqlite3
import pickle

import numpy as np

import faiss

from maxtext.src.maxtext.experimental.agent.integrative_rag_agent.config import rag_db_file


# -------- 1. Create DB --------
def create_db():
  """Create the SQLite database and `documents` table if they do not exist.

  Uses `rag_db_file` from `config` as the database path and ensures the
  `documents` table is present with columns for metadata and an embedding blob.
  """
  conn = sqlite3.connect(rag_db_file)
  cur = conn.cursor()
  cur.execute(
      """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        text TEXT NOT NULL,
        desc TEXT NOT NULL,
        file TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    """
  )
  conn.commit()
  conn.close()


create_db()


# -------- 2. Save Document + Embedding --------
def save_document(name, text, desc, file, embedding):
  """Insert a document and its embedding into the database.

  Args:
      name (str): Logical name/identifier for the document.
      text (str): Raw text content of the document.
      desc (str): Short description or summary of the document.
      file (str): File path or source identifier for the document.
      embedding (numpy.ndarray): Dense vector representation of the document
          with shape (dim,) and dtype convertible to float32.

  Returns:
      None
  """
  conn = sqlite3.connect(rag_db_file)
  cur = conn.cursor()
  # Convert NumPy embedding to binary
  emb_binary = pickle.dumps(embedding.astype(np.float32))
  cur.execute(
      "INSERT INTO documents (name,text,desc,file, embedding) VALUES (?, ?,?,?,?)", (name, text, desc, file, emb_binary)
  )
  conn.commit()
  conn.close()


# -------- 3. Load All Documents + Embeddings --------
def load_all_documents():
  """Load all documents and embeddings from the database.

  Returns:
      tuple[list[int], list[str], list[str], list[str], numpy.ndarray]:
          - ids: Row IDs for each document.
          - names: Names for each document.
          - texts: Text content for each document.
          - files: Source file paths/identifiers.
          - embeddings: Array of shape (num_docs, dim) with dtype float32.
  """
  conn = sqlite3.connect(rag_db_file)
  cur = conn.cursor()
  cur.execute("SELECT id,name, text,file, embedding FROM documents")
  rows = cur.fetchall()
  conn.close()

  ids, names, texts, files, embeddings = [], [], [], [], []
  for r in rows:
    ids.append(r[0])
    names.append(r[1])
    texts.append(r[2])
    files.append(r[3])
    embeddings.append(pickle.loads(r[4]))
  return ids, names, texts, files, np.array(embeddings, dtype=np.float32)


# -------- 4. Build FAISS Index --------
def build_faiss_index(embeddings):
  """Build a FAISS IndexFlatL2 from document embeddings.

  Args:
      embeddings (numpy.ndarray): Array of shape (num_docs, dim), dtype float32.

  Returns:
      faiss.IndexFlatL2 or None: L2 index with the provided vectors added, or
      None if the embeddings array is empty or not 2-dimensional.
  """
  if embeddings.ndim != 2 or embeddings.shape[0] == 0:
    return None
  index = faiss.IndexFlatL2(embeddings.shape[1])
  index.add(embeddings)
  return index


# -------- 5. Search in FAISS --------
def search_embedding(query_embedding, index, texts, top_k=3):
  """Search the index for nearest neighbors to a query embedding.

  Args:
      query_embedding (array-like): Vector of shape (dim,) convertible to float32.
      index (faiss.Index): A FAISS index built over document embeddings.
      texts (list[str]): Texts aligned with vectors in the index.
      top_k (int): Number of nearest neighbors to retrieve.

  Returns:
      list[tuple[str, float, int]]: For each neighbor, a tuple of (text, distance, index_in_corpus).
        Distances are squared L2 (Euclidean) norms; smaller values indicate greater similarity.
  """
  if index is None:
    return []
  query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
  distances, indices = index.search(query_embedding, top_k)
  results = [(texts[i], distances[0][pos], i) for pos, i in enumerate(indices[0])]
  return results


# -------- 6. Make Embedding Index --------
def make_embedding_index():
  """Load all documents and build a FAISS index over their embeddings.

  Returns:
      tuple[list[int], list[str], list[str], list[str], faiss.Index]:
          (ids, names, texts, files, index)
  """
  ids, names, texts, files, embeddings = load_all_documents()
  index = build_faiss_index(embeddings)
  return ids, names, texts, files, index

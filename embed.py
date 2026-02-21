import os
import chromadb
from sentence_transformers import SentenceTransformer

from ingest import extract_text_by_page, chunk_by_sections

# ── constants ─────────────────────────────────────────────────────────────────

PDF_PATH        = "DRHP_20250430105517.pdf"   # relative — place PDF next to this file
CHROMA_PATH     = "chroma_db"                 # relative — created automatically
COLLECTION_NAME = "drhp_chunks"
MODEL_NAME      = "all-MiniLM-L6-v2"

# ── helpers ───────────────────────────────────────────────────────────────────

def _get_model():
    """Load the sentence-transformer model (downloads ~90MB on first run)."""
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def _get_client():
    """Return a persistent ChromaDB client at CHROMA_PATH."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


# ── public API ────────────────────────────────────────────────────────────────

def build_vectorstore(chunks):
    """
    Embed all chunks and store them in ChromaDB.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have keys: "text", "section", "page_number".
        Output of ingest.chunk_by_sections().

    Returns
    -------
    chromadb.Collection
    """
    model  = _get_model()
    client = _get_client()

    # reset collection if it already exists (avoids DuplicateIDError on re-runs)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' exists — resetting it.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # filter out empty chunks before embedding
    chunks = [c for c in chunks if c["text"].strip()]

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks... (may take ~1 min on CPU)")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"section": c["section"], "page_number": c["page_number"]} for c in chunks]
    documents = texts

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {len(ids)} chunks in ChromaDB at: {os.path.abspath(CHROMA_PATH)}")
    return collection


def query_vectorstore(query_text, n_results=3):
    """
    Embed a query and retrieve the top-n most relevant chunks.

    Parameters
    ----------
    query_text : str
        Natural-language question or search phrase.
    n_results : int
        Number of top matches to return (default 3).

    Returns
    -------
    list[dict]  — keys: id, text, section, page_number, distance
    """
    model      = _get_model()
    client     = _get_client()
    collection = client.get_collection(COLLECTION_NAME)

    print(f"\nEmbedding query: '{query_text}'")
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # flatten ChromaDB's nested list structure
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id":          results["ids"][0][i],
            "text":        results["documents"][0][i],
            "section":     results["metadatas"][0][i]["section"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "distance":    results["distances"][0][i],
        })
    return output


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 1: Extracting text from PDF")
    print("=" * 60)
    pages = extract_text_by_page(PDF_PATH)
    print(f"Extracted {len(pages)} pages.\n")

    print("=" * 60)
    print("STEP 2: Chunking by sections")
    print("=" * 60)
    chunks = chunk_by_sections(pages)
    print(f"Created {len(chunks)} chunks.\n")

    print("=" * 60)
    print("STEP 3: Building vector store")
    print("=" * 60)
    build_vectorstore(chunks)
    print()

    print("=" * 60)
    print("STEP 4: Sample query")
    print("=" * 60)
    query   = "What is the issue size?"
    results = query_vectorstore(query, n_results=3)

    print(f"\nQuery : '{query}'")
    print(f"Top {len(results)} results:\n")
    for rank, r in enumerate(results, start=1):
        print(f"  Rank {rank}")
        print(f"  ID      : {r['id']}")
        print(f"  Section : {r['section']}")
        print(f"  Page    : {r['page_number']}")
        print(f"  Distance: {r['distance']:.4f}")
        print(f"  Snippet : {r['text'][:300]}".encode('utf-8', errors='replace').decode('utf-8'))
        print()

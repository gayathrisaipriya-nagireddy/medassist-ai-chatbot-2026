import json, hashlib
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
INPUT = ROOT / "data" / "medical_knowledge_base.json"
CHUNK_OUT = ROOT / "data" / "medical_chunks.json"
DB_PATH = ROOT / "vectorstore" / "chroma_db"
DB_PATH.mkdir(parents=True, exist_ok=True)

# Token-based chunks: 600 tokens with 100-token overlap.
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=600, chunk_overlap=100
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

items = json.loads(INPUT.read_text(encoding="utf-8"))
chunks, seen = [], set()
for item in items:
    for idx, text in enumerate(splitter.split_text(item["content"]), start=1):
        text = " ".join(text.split())
        if len(text) < 50:
            continue
        digest = hashlib.sha256(text.casefold().encode()).hexdigest()[:10]
        if digest in seen:
            continue
        seen.add(digest)
        chunks.append({
            "id": f'{item["document_id"]}_chunk_{idx}_{digest}',
            "text": text,
            "metadata": {
                "title": item["title"],
                "category": item["category"],
                "publisher": item["publisher"],
                "region_focus": item["region_focus"],
                "source_url": item["source_url"],
                "document_id": item["document_id"],
                "chunk_index": idx,
                "prepared_date": item["prepared_date"]
            }
        })

CHUNK_OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection("medassist_health_education")
for start in range(0, len(chunks), 64):
    batch = chunks[start:start+64]
    vectors = embedder.encode([x["text"] for x in batch], normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[x["id"] for x in batch],
        documents=[x["text"] for x in batch],
        metadatas=[x["metadata"] for x in batch],
        embeddings=vectors
    )
print(f"Indexed {len(items)} topics into {len(chunks)} chunks.")
print(f"Chunk JSON saved to: {CHUNK_OUT}")
print(f"ChromaDB saved to: {DB_PATH}")

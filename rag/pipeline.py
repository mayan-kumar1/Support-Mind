import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# rag/pipeline.py
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _embedding_model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded")
    return _embedding_model


def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "supportmind")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    logger.info(f"Connected to Pinecode index {index_name}")

    return index


# def get_embedding_model():
#     logger.info("Loading embedding model: %s", MODEL_NAME)

#     model = SentenceTransformer(MODEL_NAME)
#     logger.info("Embedding model loaded")

#     return model


def ingest_faq(faq_data: list):
    """Embed and upload FAQ entries to Pinecone. Run once."""
    model = get_embedding_model()
    index = get_pinecone_index()

    vectors = []

    for item in faq_data:
        text = f"{item['question']} {item['answer']}"
        embedding = model.encode(text).tolist()
        vectors.append(
            {
                "id": item["id"],
                "values": embedding,
                "metadata": {
                    "question": item["question"],
                    "answer": item["answer"],
                    "category": item["category"],
                },
            }
        )

    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        logger.info("Upserted batch %d to Pinecone", i // batch_size + 1)

    logger.info("FAQ ingestion complete. Total vectors: %d", len(vectors))


def retrieve(query: str, top_k: int = 3):
    """Retrieve top_k most relevant FAQ entries for a query."""
    model = get_embedding_model()
    index = get_pinecone_index()

    query_embedding = model.encode(query).tolist()

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    matches = []
    for match in results.matches:
        matches.append(
            {
                "score": round(match.score, 4),
                "question": match.metadata["question"],  # type: ignore
                "answer": match.metadata["answer"],  # type: ignore
                "category": match.metadata["category"],  # type: ignore
            }
        )
        logger.debug(
            "Match  score: %.4f  question: %s", match.score, match.metadata["question"]  # type: ignore
        )

    logger.info("Retrieved %d matches for query: %s", len(matches), query)
    return matches

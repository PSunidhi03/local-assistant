from sentence_transformers import SentenceTransformer
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "onboarding_assistant",
    "user": "postgres",
    "password": "postgres",
}

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(query).tolist()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT file_path, start_line, end_line, chunk_text,
               embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s
        """,
        (query_embedding, top_k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


if __name__ == "__main__":
    query = "how is the model trained"
    results = search(query)
    for file_path, start_line, end_line, text, distance in results:
        print(f"\n--- {file_path}:{start_line}-{end_line} (distance={distance:.4f}) ---")
        print(text[:200])  # first 200 chars, just to eyeball it
from pathlib import Path

# Directories we never want to walk into — not source code, or huge/irrelevant
SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", ".idea", ".vscode"}

# File extensions we consider "source code" worth ingesting
INCLUDE_EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg"}


def collect_files(repo_path: Path) -> list[Path]:
    """Walk repo_path recursively, return a list of file paths worth ingesting."""
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(skip_dir in path.parts for skip_dir in SKIP_DIRS):
            continue
        if path.suffix not in INCLUDE_EXTENSIONS:
            continue
        files.append(path)
    return files

CHUNK_SIZE_LINES = 50
CHUNK_OVERLAP_LINES = 8


def chunk_file(file_path: Path) -> list[dict]:
    """Split one file's contents into overlapping line-based chunks.
    Returns a list of dicts: {chunk_no, start_line, end_line, text}
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    chunks = []
    chunk_no = 0
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE_LINES, len(lines))
        chunk_lines = lines[start:end]
        chunks.append({
            "chunk_no": chunk_no,
            "start_line": start + 1,   # +1 because lines are 0-indexed here, but we want human-readable line numbers
            "end_line": end,           # already correct since `end` is exclusive in slicing
            "text": "\n".join(chunk_lines),
        })
        chunk_no += 1
        start += CHUNK_SIZE_LINES - CHUNK_OVERLAP_LINES  # move forward, but overlap with previous chunk
    return chunks

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load once, reuse for every chunk — loading the model is slow, embedding with it is fast
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an 'embedding' key to each chunk dict, in place."""
    texts = [c["text"] for c in chunks]
    vectors = embedding_model.encode(texts, show_progress_bar=False)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()
    return chunks

import subprocess

def get_git_info(repo_path: Path) -> tuple[str, str]:
    """Return (commit_hash, branch_name) for the repo at repo_path."""
    commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, check=True
    ).stdout.strip()

    branch_name = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_path, capture_output=True, text=True, check=True
    ).stdout.strip()

    return commit_hash, branch_name

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "onboarding_assistant",
    "user": "postgres",
    "password": "postgres",
}

def get_existing_commit_hash(repo_name: str, branch: str) -> str | None:
    """Return the stored commit_hash for this repo+branch, or None if never ingested."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT commit_hash FROM repos WHERE repo_name = %s AND branch = %s",
        (repo_name, branch),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None

def delete_existing_repo(repo_name: str, branch: str):
    """Delete the repos row for this repo+branch. ON DELETE CASCADE removes its chunks too."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM repos WHERE repo_name = %s AND branch = %s",
        (repo_name, branch),
    )
    conn.commit()
    cur.close()
    conn.close()

def insert_repo_and_chunks(repo_name: str, branch: str, commit_hash: str, all_chunks: list[dict]):
    """all_chunks is a list of dicts, each with: file_path, chunk_no, start_line, end_line, text, embedding"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO repos (repo_name, branch, commit_hash)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (repo_name, branch, commit_hash),
        )
        repo_id = cur.fetchone()[0]

        for chunk in all_chunks:
            cur.execute(
                """
                INSERT INTO chunks (repo_id, file_path, chunk_no, start_line, end_line, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    repo_id,
                    chunk["file_path"],
                    chunk["chunk_no"],
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk["text"],
                    chunk["embedding"],
                ),
            )

        conn.commit()
        print(f"Inserted repo_id={repo_id} with {len(all_chunks)} chunks.")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    repo_path = Path(r"C:\Users\P.Sunidhi\Desktop\test-repos\nanoGPT")
    repo_name = "nanoGPT"

    commit_hash, branch = get_git_info(repo_path)

    existing_hash = get_existing_commit_hash(repo_name, branch)
    if existing_hash == commit_hash:
        print(f"Repo already ingested at commit {commit_hash[:8]} — nothing to do.")
    else:
        if existing_hash is not None:
            print(f"Repo changed ({existing_hash[:8]} -> {commit_hash[:8]}), re-ingesting...")
            delete_existing_repo(repo_name, branch)
        else:
            print(f"Ingesting {repo_name} @ {branch} ({commit_hash[:8]}) for the first time")

        files = collect_files(repo_path)
        all_chunks = []
        for f in files:
            file_chunks = chunk_file(f)
            file_chunks = embed_chunks(file_chunks)
            for c in file_chunks:
                c["file_path"] = f.relative_to(repo_path).as_posix()
            all_chunks.extend(file_chunks)

        print(f"Total chunks across {len(files)} files: {len(all_chunks)}")
        insert_repo_and_chunks(repo_name, branch, commit_hash, all_chunks)
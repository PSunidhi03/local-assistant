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

if __name__ == "__main__":
    repo_path = Path("C:/Users/P.Sunidhi/Desktop/test-repos")  
    found = collect_files(repo_path)
    print(f"Found {len(found)} files to ingest:")
    for f in found:
        print(f" - {f}")
    sample_chunks = chunk_file(found[0])
    print(f"\n{found[0]} produced {len(sample_chunks)} chunks:")
    for c in sample_chunks:
        print(f"  chunk {c['chunk_no']}: lines {c['start_line']}-{c['end_line']}")
    sample_chunks = embed_chunks(sample_chunks)
    print(f"\nFirst chunk embedding: length={len(sample_chunks[0]['embedding'])}")
    print(f"First 5 values: {sample_chunks[0]['embedding'][:5]}")
    
    
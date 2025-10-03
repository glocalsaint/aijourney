import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from ctransformers import AutoModelForCausalLM


# -----------------------------
# Config
# -----------------------------
@dataclass
class RAGConfig:
    pdf_path: str
    gguf_model_dir: str         # directory containing the .gguf file
    gguf_model_file: str        # filename, e.g. "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    chunk_size: int = 600       # characters per chunk
    chunk_overlap: int = 200    # overlap between chunks
    top_k: int = 2              # retrieve top K chunks
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_new_tokens: int = 256
    temperature: float = 0.7


# -----------------------------
# Utils
# -----------------------------
def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    full = "\n".join(texts)
    # basic cleanup
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full)
    return full.strip()


def make_chunks(text: str, size: int, overlap: int) -> List[str]:
    if size <= overlap:
        raise ValueError("chunk_size must be > chunk_overlap")
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size]
        chunks.append(chunk)
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(emb_model: SentenceTransformer, chunks: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    embs = emb_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized dot-product
    index.add(embs.astype(np.float32))
    return index, embs


def retrieve(emb_model: SentenceTransformer, index: faiss.IndexFlatIP, chunks: List[str], query: str, top_k: int) -> List[Tuple[int, float, str]]:
    q = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        results.append((int(idx), float(score), chunks[int(idx)]))
    return results

def format_prompt(question: str, contexts: List[str], max_context_tokens: int = 400):
    # join and truncate contexts so we don't exceed the model's window
    joined = "\n\n---\n\n".join(contexts)
    if len(joined.split()) > max_context_tokens:
        joined = " ".join(joined.split()[:max_context_tokens])

    prompt = f"""[INST] You are a precise assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say you don't know.

# Context
{joined}

# Question
{question}

# Answer
[/INST]"""
    return prompt


def load_llm(gguf_dir: str, gguf_file: str):
    path = gguf_dir
    model = AutoModelForCausalLM.from_pretrained(
        path,
        model_file=gguf_file,
        model_type="mistral",   # important for correct tokenizer behavior
        gpu_layers=0            # force pure CPU
    )
    return model


def generate(llm, prompt: str, max_new_tokens: int, temperature: float) -> str:
    # ctransformers provides a callable for generation
    out = llm(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        stop=["</s>"]
    )
    return out.strip()


def run_rag(cfg: RAGConfig):
    # 1) Load & chunk
    print("Reading PDF...")
    text = read_pdf_text(cfg.pdf_path)
    if not text:
        print("No extractable text found. If this is a scanned PDF, you need OCR (e.g., pytesseract).")
        sys.exit(1)

    print(f"Document length: {len(text):,} characters")
    chunks = make_chunks(text, cfg.chunk_size, cfg.chunk_overlap)
    print(f"Created {len(chunks)} chunks (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

    # 2) Embeddings + FAISS
    print("Loading embedding model...")
    emb_model = SentenceTransformer(cfg.embedding_model)
    print("Building FAISS index...")
    index, _ = build_faiss_index(emb_model, chunks)

    # 3) LLM
    print("Loading LLM (GGUF via ctransformers, CPU-only)...")
    llm = load_llm(cfg.gguf_model_dir, cfg.gguf_model_file)
    print("Ready. Ask questions; type 'exit' to quit.\n")

    # 4) Loop
    while True:
        try:
            q = input("Q: ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = retrieve(emb_model, index, chunks, q, cfg.top_k)
        contexts = [h[2] for h in hits]

        prompt = format_prompt(q, contexts)
        answer = generate(llm, prompt, cfg.max_new_tokens, cfg.temperature)

        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources (top-k chunks) ---")
        for i, (idx, score, _) in enumerate(hits, 1):
            snippet = chunks[idx][:240].replace("\n", " ")
            print(f"{i}. chunk#{idx} score={score:.3f}: {snippet}...")
        print()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rag_pdf_qa.py <path/to/document.pdf> <path/to/model_dir> [model_file.gguf]")
        print("Example: python rag_pdf_qa.py ./docs/paper.pdf ./models mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    model_dir = sys.argv[2]
    model_file = sys.argv[3] if len(sys.argv) >= 4 else "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    cfg = RAGConfig(
        pdf_path=pdf_path,
        gguf_model_dir=model_dir,
        gguf_model_file=model_file
    )
    run_rag(cfg)

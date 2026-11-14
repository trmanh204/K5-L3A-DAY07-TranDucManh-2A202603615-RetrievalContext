import json
import math
import os
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
import numpy as np


def load_env(env_path=".env"):
    """Đọc file .env thủ công không cần thư viện bên ngoài."""
    p = Path(env_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()


def tokenize(text: str) -> list[str]:
    """Tách từ tiếng Việt chuẩn hóa NFC."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


# ==============================================================================
# 1. LEXICAL RETRIEVER (TF-IDF)
# ==============================================================================
class TfidfRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        query_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, doc_vec in zip(self.chunks, self.vectors):
            if filters:
                meta = chunk.get("metadata", {})
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            score = sum(v * doc_vec.get(term, 0.0) for term, v in query_vec.items())
            if score > 0:
                results.append({**chunk, "score": float(score), "strategy": "tfidf"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


# ==============================================================================
# 2. DENSE RETRIEVER (OPENROUTER TEXT-EMBEDDING-3-SMALL)
# ==============================================================================
class DenseRetriever:
    def __init__(self, chunks: list[dict], cache_file: str = "data/chunks_with_embeddings.json"):
        self.chunks = chunks
        self.cache_file = Path(cache_file)
        self.model_name = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Thiếu OPENROUTER_API_KEY trong file .env!")

        self.embeddings = self._load_or_create_embeddings()

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        batch_size = 50
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = json.dumps({"model": self.model_name, "input": batch}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                sorted_items = sorted(data["data"], key=lambda x: x["index"])
                for item in sorted_items:
                    all_vecs.append(item["embedding"])
        return all_vecs

    def _load_or_create_embeddings(self) -> np.ndarray:
        if self.cache_file.exists():
            print(f"[*] Load vector cache từ: {self.cache_file}")
            cached_data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            vecs = np.array([item["embedding"] for item in cached_data], dtype=np.float32)
        else:
            print(f"[*] Đang embed {len(self.chunks)} chunks bằng {self.model_name} qua OpenRouter...")
            texts = [c["text"] for c in self.chunks]
            raw_vecs = self._call_api(texts)
            vecs = np.array(raw_vecs, dtype=np.float32)
            
            # Lưu cache
            cache_payload = [
                {"id": c["id"], "embedding": raw_vecs[i]}
                for i, c in enumerate(self.chunks)
            ]
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
            print(f"[*] Đã lưu vector cache vào: {self.cache_file}")

        # Chuẩn hóa L2-norm để cosine distance chuyển thành phép nhân vô hướng dot-product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_query(self, query: str) -> np.ndarray:
        raw_vec = self._call_api([query])[0]
        v = np.array(raw_vec, dtype=np.float32)
        norm = np.linalg.norm(v) or 1.0
        return v / norm

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        q_vec = self.embed_query(query)
        # Cosine similarity bằng dot product với ma trận vector đã chuẩn hóa
        scores = np.dot(self.embeddings, q_vec)

        results = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if filters:
                meta = chunk.get("metadata", {})
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            results.append({**chunk, "score": float(score), "strategy": "dense"})

        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


# ==============================================================================
# 3. HYBRID RETRIEVER (RECIPROCAL RANK FUSION - RRF)
# ==============================================================================
class HybridRrfRetriever:
    def __init__(self, tfidf: TfidfRetriever, dense: DenseRetriever, rrf_k: int = 60):
        self.tfidf = tfidf
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, filters: dict | None = None, candidate_k: int = 10) -> list[dict]:
        # Lấy candidate từ cả 2 nguồn
        dense_hits = self.dense.search(query, top_k=candidate_k, filters=filters)
        tfidf_hits = self.tfidf.search(query, top_k=candidate_k, filters=filters)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        # Tính điểm RRF từ Dense
        for rank, hit in enumerate(dense_hits):
            cid = hit["id"]
            chunk_map[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Tính điểm RRF từ TF-IDF
        for rank, hit in enumerate(tfidf_hits):
            cid = hit["id"]
            if cid not in chunk_map:
                chunk_map[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sắp xếp theo tổng điểm RRF
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results = []
        for cid in sorted_ids[:top_k]:
            c = dict(chunk_map[cid])
            c["score"] = float(rrf_scores[cid])
            c["strategy"] = "hybrid_rrf"
            results.append(c)

        return results
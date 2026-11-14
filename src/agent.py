import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path("starter").resolve()))
from agentic import run_agent


def clean_citations(raw_citations: Any) -> list[str]:
    """Trích xuất chính xác các mã chunk_id dạng hust-handbook-2026_pX_para_Y."""
    if isinstance(raw_citations, str):
        raw_citations = [raw_citations]
    elif not isinstance(raw_citations, list):
        raw_citations = []

    cleaned = []
    for c in raw_citations:
        matches = re.findall(r"hust-handbook-2026_p\d+_para_\d+", str(c))
        if matches:
            cleaned.extend(matches)
        elif isinstance(c, str) and c.strip():
            cleaned.append(c.strip())
    return list(dict.fromkeys(cleaned))


def extract_json(content: str) -> dict[str, Any]:
    """Trích xuất an toàn JSON object từ phản hồi của LLM."""
    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        return json.loads(content[start : end + 1])
    return json.loads(content)


class OpenRouterController:
    """Controller điều phối quyết định của Agent bằng OpenRouter GPT-4o-mini."""

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Thiếu OPENROUTER_API_KEY trong file .env!")
        self.model_name = model_name
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"[*] Agent Controller đang sử dụng OPENROUTER: {self.model_name}")

    def _call_openrouter(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }).encode("utf-8")

        req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data["choices"][0]["message"]["content"]
            return extract_json(raw)

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # Lượt 1: Dùng trực tiếp câu hỏi gốc để tìm kiếm chính xác
        if not history:
            return {"action": "search", "query": question}

        system_prompt = (
            "Bạn là trợ lý giải đáp học vụ và quy chế ĐHBK Hà Nội.\n"
            "Dựa vào câu hỏi và ngữ cảnh trích xuất dưới đây, hãy quyết định hành động tiếp theo.\n"
            "Chỉ trả về DUY NHẤT một JSON object theo đúng một trong các định dạng sau:\n\n"
            "1. Nếu ngữ cảnh ĐÃ CÓ BẰNG CHỨNG để trả lời:\n"
            '   {"action": "answer", "answer": "<câu trả lời đầy đủ, ngắn gọn> [chunk_id]", "citations": ["chunk_id"]}\n'
            "   (LƯU Ý: citations là danh sách mã chunk ID, ví dụ [\"hust-handbook-2026_p5_para_0\"])\n\n"
            "2. Nếu ngữ cảnh chưa đủ thông tin và muốn tìm kiếm thêm:\n"
            '   {"action": "search", "query": "<từ khóa tìm kiếm mới ngắn gọn>"}\n\n'
            "3. Nếu tài liệu hoàn toàn KHÔNG CÓ BẰNG CHỨNG để trả lời:\n"
            '   {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
        )

        latest = history[-1]
        queries_str = ", ".join(latest.get("queries", []))
        user_content = (
            f"Câu hỏi của sinh viên: {question}\n\n"
            f"Các truy vấn đã tìm kiếm trước đó: [{queries_str}]\n"
            f"Ngữ cảnh thu thập được từ tài liệu:\n{latest.get('context', '')}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        action = self._call_openrouter(messages)

        # Xử lý chuẩn hóa an toàn
        if action.get("action") == "answer":
            answer = action.get("answer", "")
            citations = clean_citations(action.get("citations", []))
            for cid in citations:
                if f"[{cid}]" not in answer:
                    answer = f"{answer} [{cid}]"
            action["answer"] = answer
            action["citations"] = citations

        return action


class AgentRetrieverAdapter:
    """Bọc bộ tìm kiếm Hybrid RRF để tương thích với giao diện Retriever của agentic.py."""

    def __init__(self, backend_retriever):
        self.backend = backend_retriever

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        raw_results = self.backend.search(query, top_k=top_k, filters=filters)
        results = []
        for r in raw_results:
            results.append({
                "id": r["id"],
                "text": r["text"],
                "source_id": r["source_id"],
                "source": r["source"],
                "page": r["page"],
                "metadata": r.get("metadata", {}),
                "score": float(r.get("score", 0.0)),
            })
        return results
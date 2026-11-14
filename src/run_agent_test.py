import json
import time
from pathlib import Path
import sys

from agent import OpenRouterController, AgentRetrieverAdapter
from retrievers import TfidfRetriever, DenseRetriever, HybridRrfRetriever

sys.path.insert(0, str(Path("starter").resolve()))
from agentic import run_agent


def main():
    print("=" * 70)
    print("   CHẠY AGENTIC RAG TRÊN TẬP TEST (7 CÂU HỎI)")
    print("=" * 70)

    # 1. Nạp dữ liệu
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    cases = json.loads(Path("data/cases.json").read_text(encoding="utf-8"))
    test_cases = [c for c in cases if c.get("split") == "test"]

    print(f"\n[1/3] Đã nạp {len(chunks)} chunks và {len(test_cases)} câu hỏi Test.")

    # 2. Khởi tạo Retriever & Controller (cấu hình đóng băng từ Dev)
    print("\n[2/3] Khởi tạo Hybrid RRF Retriever và LLM Controller...")
    tfidf = TfidfRetriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRrfRetriever(tfidf, dense, rrf_k=60)
    retriever_tool = AgentRetrieverAdapter(hybrid)
    controller = OpenRouterController(model_name="nvidia/nemotron-3.5-lightning:free")

    # 3. Chạy Agent trên tập test
    print("\n[3/3] Bắt đầu chạy vòng lặp Agent...")
    traces = []
    answered_count = 0
    abstain_count = 0

    for idx, case in enumerate(test_cases, start=1):
        cid = case["id"]
        q_text = case["question"]
        is_answerable = case["answerable"]

        print(f"\n--- [{idx}/{len(test_cases)}] Câu hỏi ({cid}): {q_text}")
        print(f"    Loại câu: {'Có bằng chứng' if is_answerable else 'THIẾU BẰNG CHỨNG (Kỳ vọng từ chối)'}")

        start_time = time.perf_counter()
        try:
            res = run_agent(
                question=q_text,
                controller=controller,
                retriever=retriever_tool,
                max_searches=3,
                top_k=3,
                budget=5000,
            )
            elapsed = time.perf_counter() - start_time
            res["query_id"] = cid
            res["elapsed_ms"] = round(elapsed * 1000, 2)
            traces.append(res)

            status = res.get("status")
            if status == "answered":
                answered_count += 1
                cites = res.get("citations", [])
                print(f" -> Trạng thái: ANSWERED | Trích dẫn: {cites} | Thời gian: {res['elapsed_ms']} ms")
                print(f"    Trả lời: {res.get('answer')[:120]}...")
            else:
                abstain_count += 1
                print(f" -> Trạng thái: ABSTAIN (Từ chối) | Thời gian: {res['elapsed_ms']} ms")
                print(f"    Lý do sự kiện: {[e['event'] for e in res.get('events', [])]}")

        except Exception as e:
            print(f" -> LỖI HỆ THỐNG: {e}")
            traces.append({
                "query_id": cid,
                "status": "error",
                "error": str(e),
                "events": [],
                "citations": [],
            })

    # 4. Lưu kết quả traces
    out_dir = Path("results/agent_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_file = out_dir / "traces.json"
    traces_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print("   TỔNG KẾT VÒNG CHẠY AGENT TRÊN TẬP TEST")
    print("=" * 70)
    print(f"Tổng số câu test: {len(test_cases)}")
    print(f"Số câu trả lời được (Answered): {answered_count}")
    print(f"Số câu từ chối (Abstain)     : {abstain_count}")
    print(f"Đã lưu chi tiết trace tại     : {traces_file}\n")


if __name__ == "__main__":
    main()
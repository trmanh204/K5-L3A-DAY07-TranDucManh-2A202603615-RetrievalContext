import argparse
import json
import statistics
import time
from pathlib import Path
from retrievers import TfidfRetriever, DenseRetriever, HybridRrfRetriever


def hit(chunk: dict, gold: list[dict]) -> bool:
    return any(
        chunk["source_id"] == g["source_id"]
        and chunk["page"] == g["page"]
        and g["evidence"] in chunk["text"]
        for g in gold
    )


def pack_context(results: list[dict], budget: int = 2500) -> tuple[str, list[str]]:
    """Đóng gói ngữ cảnh với ngân sách ký tự budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def evaluate_strategy(name: str, retriever, questions: list[dict], budget: int = 2500) -> tuple[dict, list[dict]]:
    traces = []
    times_ms = []

    for q in questions:
        filters = q.get("filter")
        start = time.perf_counter()
        results = retriever.search(q["question"], top_k=3, filters=filters)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

        context, ids = pack_context(results, budget=budget)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": name,
            "top_k": 3,
            "budget_chars": budget,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(c["hit"] for c in marked[:3]),
            "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
        })

    n = len(traces)
    hit1_cnt = sum(t["hit1"] for t in traces)
    hit3_cnt = sum(t["hit3"] for t in traces)
    ctx_cnt = sum(t["context_hit"] for t in traces)

    summary = {
        "strategy": name,
        "n": n,
        "hit1": f"{hit1_cnt}/{n} ({hit1_cnt / n * 100:.1f}%)",
        "hit3": f"{hit3_cnt}/{n} ({hit3_cnt / n * 100:.1f}%)",
        "context_evidence_rate": f"{ctx_cnt}/{n} ({ctx_cnt / n * 100:.1f}%)",
        "median_ms": round(statistics.median(times_ms), 2),
        "mean_ms": round(statistics.mean(times_ms), 2),
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description="So sánh 3 phương pháp Retrieval trên tập Dev/Test")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--budget", type=int, default=2500, help="Ngân sách context budget (ký tự)")
    args = parser.parse_args()

    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
    selected_questions = [q for q in questions if q.get("split") == args.split]

    print("=" * 70)
    print(f"   ĐÁNH GIÁ VÀ SO SÁNH RETRIEVAL TRÊN TẬP [{args.split.upper()}] ({len(selected_questions)} câu)")
    print("=" * 70)

    # 1. Khởi tạo các Retriever
    print("\n[1/3] Đang khởi tạo các Retriever...")
    tfidf_retriever = TfidfRetriever(chunks)
    dense_retriever = DenseRetriever(chunks)
    hybrid_retriever = HybridRrfRetriever(tfidf_retriever, dense_retriever, rrf_k=60)

    # 2. Đánh giá từng phương pháp
    print("\n[2/3] Đang chạy đánh giá trên từng phương pháp...")
    retrievers = [
        ("TF-IDF (Lexical)", tfidf_retriever),
        ("Dense (Vector Store)", dense_retriever),
        ("Hybrid (RRF)", hybrid_retriever),
    ]

    summaries = []
    out_dir = Path("results/retrieval_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, r in retrievers:
        print(f" -> Đang chạy: {name}...")
        summary, traces = evaluate_strategy(name, r, selected_questions, budget=args.budget)
        summaries.append(summary)

        # Lưu trace chi tiết
        slug = name.split()[0].lower()
        (out_dir / f"traces_{slug}_{args.split}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # 3. In bảng so sánh Markdown
    print("\n" + "=" * 70)
    print(f"   BẢNG KẾT QUẢ SO SÁNH TRÊN TẬP {args.split.upper()} (Context Budget: {args.budget} chars)")
    print("=" * 70)
    print(f"| {'Phương pháp':<25} | {'Hit@1':<14} | {'Evidence Hit@3':<16} | {'Context Evidence Rate':<22} | {'Độ trễ median (ms)':<18} |")
    print(f"|{'-'*27}|{'-'*16}|{'-'*18}|{'-'*24}|{'-'*20}|")
    for s in summaries:
        print(f"| {s['strategy']:<25} | {s['hit1']:<14} | {s['hit3']:<16} | {s['context_evidence_rate']:<22} | {str(s['median_ms']) + ' ms':<18} |")
    print("=" * 70)

    # Lưu bảng tổng kết
    (out_dir / f"summary_{args.split}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[3/3] Đã lưu toàn bộ vết và tổng hợp vào: {out_dir}\n")


if __name__ == "__main__":
    main()
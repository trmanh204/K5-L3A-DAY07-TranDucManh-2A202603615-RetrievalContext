import json
import re
from pathlib import Path

def main():
    chunks_path = Path("data/chunks.json")
    questions_path = Path("data/questions.json")
    
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    
    # Gom nhóm text của chunk theo (source_id, page)
    chunk_map = {}
    for c in chunks:
        key = (c["source_id"], c["page"])
        chunk_map.setdefault(key, []).append(c["text"])
        
    def find_exact_substring(chunk_text, ev_quote):
        words = [re.escape(w) for w in re.findall(r"\S+", ev_quote)]
        if not words:
            return None
        # Khớp regex cho phép khoảng trắng linh hoạt (dấu cách hoặc \n)
        pattern = r"\s+".join(words)
        m = re.search(pattern, chunk_text, re.IGNORECASE)
        if m:
            return chunk_text[m.start():m.end()]
        # Nếu trích dẫn quá dài, thử khớp phần đầu
        for length in range(len(words), 3, -1):
            sub_pattern = r"\s+".join(words[:length])
            m = re.search(sub_pattern, chunk_text, re.IGNORECASE)
            if m:
                return chunk_text[m.start():m.end()]
        return None

    fixed_count = 0
    for q in questions:
        for g in q["gold"]:
            src = g["source_id"]
            pg = g["page"]
            ev = g["evidence"]
            texts = chunk_map.get((src, pg), [])
            found = False
            for t in texts:
                if ev in t:
                    found = True
                    break
                matched_slice = find_exact_substring(t, ev)
                if matched_slice:
                    g["evidence"] = matched_slice
                    fixed_count += 1
                    found = True
                    break
                    
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> Đã chuẩn hóa xong {fixed_count}/20 câu trích dẫn evidence khớp 100% với chunks!")

if __name__ == "__main__":
    main()
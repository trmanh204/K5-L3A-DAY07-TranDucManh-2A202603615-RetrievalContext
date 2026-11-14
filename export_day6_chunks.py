import sys
import json
from pathlib import Path

# Thêm đường dẫn mã nguồn Day 6 vào môi trường chạy
day6_src = Path(r"D:\LAB06\K5-L3A-DAY06-TranDucManh-2A202603615-DataIngestion\starter")
sys.path.insert(0, str(day6_src))

from src.ingestion import load_pdf
from src.chunking import by_paragraph

def main():
    pdf_path = Path("data/So_tay_sinh_vien_2026_compressed.pdf")
    print(f"1. Đang đọc PDF: {pdf_path}")
    pages = load_pdf(pdf_path, document_id="hust-handbook-2026", source=str(pdf_path))
    
    print("2. Đang tạo Chunks theo Chiến lược B (Paragraph-aware 800 ký tự)...")
    chunks_B = []
    for p in pages:
        chunks_B.extend(by_paragraph(p, max_size=800))
        
    print(f" -> Đã tạo được {len(chunks_B)} chunks.")
    
    # Chuyển đổi thành dạng dict để lưu JSON
    output_data = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "source": c.source,
            "page": c.page,
            "text": c.text,
            "metadata": c.metadata
        }
        for c in chunks_B
    ]
    
    out_file = Path("data/day6_chunks.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"3. Đã xuất thành công ra file: {out_file}")

if __name__ == "__main__":
    main()
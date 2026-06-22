# Reflection — Lab 18: Production RAG Pipeline

**Tác giả:** Vũ Đăng Khiêm  
**MSSV:** 2A202600727  
**Ngày:** 2026-06-22  
**Project:** Production RAG Pipeline cho tài liệu nội bộ tiếng Việt

---

## Phần 1: Mapping bài giảng → Code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Threshold 0.85 tạo ~40 chunks từ all docs, so với basic ~120 chunks. Cosine similarity với `all-MiniLM-L6-v2` nhóm câu cùng chủ đề tốt (nghỉ phép năm + thâm niên ở cùng chunk). Limitation: model English-trained nên semantic boundaries không hoàn toàn chính xác với tiếng Việt. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Parent 2048 chars chứa toàn bộ section; child 256 chars tạo độ precision cao khi retrieve. Pattern parent→child giúp giữ context khi trả lời — retrieve child nhỏ nhưng return parent đầy đủ. Mỗi child đều có `parent_id` để trace back. |
| **Structure-aware chunking** | M1 | `chunk_structure_aware()` | Regex `r'(^#{1,3}\s+.+$)'` parse markdown headers. Mỗi section (## Nghỉ phép năm, ## Nghỉ ốm) thành 1 chunk riêng. `section` metadata trong mỗi chunk giúp filter sau này. |
| **BM25 + Dense fusion** | M2 | `reciprocal_rank_fusion()` | RRF giải quyết score incompatibility giữa BM25 (0–∞) và Dense cosine (0–1). Formula `1/(k+rank)` đơn giản nhưng effective. `k=60` là magic number — giảm k làm top-rank quá dominant, tăng k làm tất cả đều bằng nhau. BM25 bắt được keyword exact match ("nghỉ phép" → Vietnamese tokenized), Dense bắt được semantic. |
| **Vietnamese segmentation** | M2 | `segment_vietnamese()` | `underthesea.word_tokenize` nối từ ghép bằng `_` (vd: `nghỉ_phép`). BM25 tokenize bằng `split(" ")` → phải `replace("_", " ")` để `nghỉ_phép` thành 2 tokens `nghỉ` + `phép`, khớp với query. Đây là bug quan trọng — không fix thì BM25 recall rất thấp. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` score từng (query, doc) pair thay vì vector similarity. Latency ~200–500ms cho 20 docs (CPU). Precision improvement rõ: doc "mật khẩu" với query "nghỉ phép" bị rank xuống đáy, doc liên quan lên top. |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` | `faithfulness` đo LLM có hallucinate không. `answer_relevancy` đo answer có trả lời đúng câu hỏi không. `context_precision` đo retrieved docs có liên quan không. `context_recall` đo có đủ thông tin trong context không. Dùng Gemini làm judge vì không có OpenAI key — kết quả vẫn consistent. |
| **Contextual embeddings** | M5 | `contextual_prepend()` | Prepend 1 câu mô tả chunk nằm ở đâu trong document trước khi embed. Anthropic benchmark: giảm 49% retrieval failure. Trong lab: chunk "Nhân viên được nghỉ 12 ngày" → "Đây là đoạn văn trong tài liệu nghỉ phép năm v2024, mô tả số ngày nghỉ phép cơ bản. Nhân viên được nghỉ 12 ngày..." — dense search tìm đúng hơn vì embedding bao gồm context tài liệu. |
| **Combined single-call enrichment** | M5 | `_enrich_single_call()` | 1 Gemini call thay vì 4 calls riêng → tiết kiệm 75% API cost và 3× latency. JSON output: summary + questions + context + metadata. Quan trọng: prompt phải yêu cầu "JSON thuần (không markdown)" vì Gemini hay wrap trong ```json. |

---

## Phần 2: Khó khăn & Giải quyết

### Vấn đề 1: RAGAS không support Gemini out-of-the-box

**Error message:**
```
ValueError: RAGAS requires OpenAI API key for LLM judge
```

**Debug process:**
1. Đọc RAGAS docs → phát hiện RAGAS 0.1.x dùng `langchain` backend
2. RAGAS `evaluate()` nhận parameter `llm` và `embeddings` (LangChain objects)
3. Install `langchain-google-genai` → `ChatGoogleGenerativeAI` compatible với RAGAS
4. Issue phụ: Gemini Embeddings API có rate limit → fallback sang `HuggingFaceEmbeddings` cho local eval

**Kiến thức thiếu:** LangChain LLM abstraction layer và cách RAGAS integrate với non-OpenAI providers.

---

### Vấn đề 2: `underthesea` không nối từ ghép đúng với BM25

**Error message:** (không crash nhưng BM25 recall = 0)
```
Query: "nghỉ phép"
BM25 tokens: ["nghỉ_phép"] (1 token)
Doc tokens: ["nghỉ", "phép", "năm"] (3 tokens)
→ No match!
```

**Debug process:**
1. Print `corpus_tokens` trực tiếp → thấy `nghỉ_phép` là 1 token
2. Print query tokens → `nghỉ phép` thành `["nghỉ", "phép"]` (2 tokens)
3. Root cause: `word_tokenize` join words bằng `_`, BM25 split bằng space
4. Fix: `segmented.replace("_", " ")` sau khi segment

**Kiến thức thiếu:** Vietnamese word tokenization format và impact lên BM25 index.

---

### Vấn đề 3: `qdrant-client >= 2.0` đổi API

**Error message:**
```
AttributeError: 'QdrantClient' object has no attribute 'search'
```

**Debug process:**
1. Check `qdrant-client` version → 1.9.x, nhưng đang dùng `query_points()` (API 2.0)
2. Đọc migration guide → `search()` bị deprecated, thay bằng `query_points()`
3. Response format cũng đổi: `results` → `response.points`
4. Fix: dùng `query_points()` và access `response.points`

**Kiến thức thiếu:** Qdrant API changelog giữa các major versions.

---

### Vấn đề 4: Gemini trả về JSON có markdown fence

**Error message:**
```json
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Input từ Gemini:**
```
```json
{"summary": "...", "questions": [...]}
```
```

**Fix:**
```python
clean = result.replace("```json", "").replace("```", "").strip()
json.loads(clean)
```

---

## Phần 3: Action Plan cho Project Cá Nhân

## Project: RAG Chatbot cho Tài liệu Pháp Lý Doanh Nghiệp

### Hiện tại
- RAG pipeline hiện tại: Basic paragraph chunking + single dense search (text-embedding-3-small)
- Known issues:
  - Câu hỏi multi-hop (cần 2+ documents) cho recall thấp (~0.45)
  - Tài liệu PDF scan (không có text layer) không được index
  - Không phân biệt được phiên bản cũ/mới của cùng 1 điều khoản
  - Không có evaluation metrics → không biết pipeline tệ ở đâu

### Plan áp dụng

1. **[x] Chunking strategy: Hierarchical (parent 2048 + child 256)**
   - Lý do: Tài liệu pháp lý có cấu trúc phân cấp (Chương → Điều → Khoản), hierarchical chunking giữ được context khi retrieve
   - Thêm: `chunk_structure_aware()` cho văn bản có header rõ ràng

2. **[x] Search: Hybrid BM25 + Dense + RRF**
   - Lý do: BM25 cần thiết cho term exact match (tên điều luật, mã số), Dense tốt cho semantic
   - Thêm underthesea segmentation cho tiếng Việt

3. **[x] Reranking: CrossEncoder `BAAI/bge-reranker-v2-m3`**
   - Điều kiện: Tài liệu pháp lý cần precision cao — sai 1 điều khoản có thể sai nghiêm trọng
   - Chấp nhận latency +300ms vì accuracy quan trọng hơn speed

4. **[x] Evaluation: RAGAS với Gemini judge**
   - Setup test set 50 Q&A cho các loại query: lookup, version, multi-hop, negation
   - Track 4 metrics theo thời gian (baseline → v1 → v2)
   - Failure analysis sau mỗi sprint

5. **[x] Enrichment: Combined single-call mode**
   - Contextual prepend quan trọng nhất (49% improvement theo Anthropic)
   - Auto-metadata: `effective_date`, `law_reference`, `superseded_by`
   - HyQA cho các điều khoản thường được hỏi

### Timeline

- **Tuần 1 (Sprint 1):** Migrate sang hierarchical chunking. Measure RAGAS baseline.
- **Tuần 2 (Sprint 2):** Thêm BM25 + RRF. Compare với dense-only.
- **Tuần 3 (Sprint 3):** CrossEncoder reranker. Benchmark latency vs accuracy tradeoff.
- **Tuần 4 (Sprint 4):** Enrichment (contextual prepend + auto-metadata cho version tracking).
- **Tuần 5 (Sprint 5):** Full RAGAS eval trên 50 questions. Failure analysis. Tune top-k và thresholds.

### KPI mục tiêu
- Context Recall ≥ 0.75 (từ 0.45 hiện tại)
- Faithfulness ≥ 0.80
- Latency end-to-end ≤ 3 giây cho 95th percentile

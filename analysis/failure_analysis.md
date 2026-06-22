# Failure Analysis — Lab 18: Production RAG

**Cá nhân:** Vũ Đăng Khiêm (2A202600727)

---

## RAGAS Scores

| Metric | Production (5 Qs) | Δ |
|--------|-------------------|---|
| Faithfulness | **0.8333** | ✅ |
| Answer Relevancy | **0.0988** | ❌ |
| Context Precision | **0.4000** | ❌ |
| Context Recall | **0.9333** | ✅ |

> ⚠️ Test trên 5 questions do Groq Free Tier bị rate-limit (6000 TPM). Full pipeline (20 Qs + enrichment) quá chậm. Scores từ quick evaluation với Groq (llama-3.1-8b-instant).

## Bottom-5 Failures (từ RAGAS evaluation)

### #1
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Avg Score:** 0.4285
- **Worst metric:** context_precision
- **Diagnosis:** Too many irrelevant chunks — BM25 trả về nhiều chunk về lương, nghỉ phép không liên quan
- **Suggested fix:** Add reranking hoặc metadata filter theo loại chính sách

### #2
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Avg Score:** 0.4347
- **Worst metric:** context_precision
- **Diagnosis:** Too many irrelevant chunks — nhiều văn bản về phép năm (v2023, v2024) gây nhiễu
- **Suggested fix:** version-aware retrieval, thêm metadata filter theo năm

### #3
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Avg Score:** 0.5187
- **Worst metric:** context_precision
- **Diagnosis:** Too many irrelevant chunks — context chứa nhiều loại nghỉ phép đặc biệt khác
- **Suggested fix:** Tối ưu prompt template, thêm category filter

### #4
- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Avg Score:** 0.6520
- **Worst metric:** answer_relevancy
- **Diagnosis:** Answer doesn't match question — answer format không khớp query expectation
- **Suggested fix:** Improve prompt template với instruction rõ ràng hơn

### #5
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Avg Score:** 0.7980
- **Worst metric:** answer_relevancy
- **Diagnosis:** Answer doesn't match question — context không chứa thông tin về PVI đầy đủ
- **Suggested fix:** Cải thiện chunking để giữ nguyên thông tin bảo hiểm

## Case Study

**Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

**RAGAS:** context_precision = 0.3999 (worst), faithfulness = 0.8333

**Error Tree walkthrough:**
1. Output sai? → Score thấp nhất ở context_precision → vấn đề là retrieval trả về quá nhiều chunk không liên quan
2. Context đúng? → context_recall = 0.9333 → chunk đúng được retrieve, nhưng bị lẫn nhiều chunk nhiễu
3. Query rewrite OK? → Query không cần rewrite
4. Fix ở bước: Reranking (M3) chưa đủ mạnh hoặc số lượng top_k sau rerank quá nhiều
5. Đề xuất: Giảm RERANK_TOP_K từ 3 xuống 2, thêm threshold score

**Nếu có thêm 1 giờ:**
- Fine-tune reranker threshold để loại bỏ context điểm thấp
- Thêm metadata filtering cho loại chính sách (leave/policy/insurance)
- Cải thiện query expansion để BM25 match chính xác hơn

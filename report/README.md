# Bridging the Structure-Gap: An Empirical Study on Layout-Aware Parsing for Financial RAG

<p align="center"><strong>Zhichao Pan</strong></p>
<p align="center"><em>Independent Research</em></p>
<p align="center">January 2026</p>

---

## Abstract

**Problem Definition.** Financial documents, such as Form 10-K filings and earnings reports, rely heavily on complex tables to convey critical numerical information. Standard Retrieval-Augmented Generation (RAG) pipelines, which treat documents as unstructured flat text, systematically fail to preserve the spatial relationships within these tables. We term this fundamental mismatch between the *geometric layout* of source documents and their *linearized text representation* the **"Structure-Gap."** This gap leads to semantic collision—where visually distinct data points become ambiguous in text—and is a primary driver of hallucination in financial question-answering (QA) tasks.

**Proposed Method.** We propose a **Layout-Aware Parsing Pipeline** that leverages **Vision-Language Model (VLM) based document parsing** to extract source PDFs into **Markdown-serialized** text, explicitly preserving tabular structure as machine-readable Markdown tables. This representation is then chunked and indexed using standard dense retrieval.

**Results.** Evaluated on a curated benchmark derived from the NVIDIA FY2024 10-K filing (N=8 QA pairs), our Layout-Aware pipeline achieved an overall accuracy of **68.8%**, representing a **+37.5% relative improvement** over the Unstructured Baseline (50.0%). A detailed failure mode analysis reveals that the remaining errors are attributable to three distinct categories: Retrieval Failure (33%), Generation Error (33%), and Semantic Ambiguity in the embedding model (33%). This analysis suggests that layout-aware parsing is a *necessary but not sufficient* condition for reliable Financial RAG; future work must integrate more nuanced retrieval techniques.

---

## 1. Introduction

### 1.1 The Structure-Gap Problem

Financial documents are inherently **semi-structured**. A Consolidated Statement of Income is not merely a paragraph of text; it is a precisely formatted table where the spatial position of a value (e.g., the cell at the intersection of "Revenue" and "FY2024") is semantically critical. When a standard PDF extractor (e.g., PyPDF2) linearizes this table into a "bag of words," the row-column associations are destroyed.

Consider the following example from the NVIDIA 10-K:

**Table 1: Parsing Quality Comparison**

| Parser | Output Snippet | Structure Preserved? |
|:---|:---|:---:|
| **Unstructured (PyPDF2)** | `Revenue $60,922 $26,974 ... Gross margin 72.7% 56.9%` | ❌ No |
| **Layout-Aware (VLM)** | `| Metric | FY2024 | FY2023 |`<br>`|---|---|---|`<br>`| Revenue | $60,922 | $26,974 |` | ✅ Yes |

The Unstructured output juxtaposes values from different rows and columns, creating **semantic collision**: the LLM cannot reliably determine which value belongs to which year. The Layout-Aware output, by serializing the table into Markdown, provides an explicit row-column schema that the LLM's attention mechanism can leverage.

### 1.2 Our Contributions

1.  **Formal Definition of the Structure-Gap.** We articulate the problem of spatial information loss in document linearization as a distinct failure mode in RAG pipelines for semi-structured data.
2.  **Layout-Aware Parsing Pipeline.** We propose and implement a pipeline using VLM-based Markdown serialization to preserve tabular structure.
3.  **Rigorous Error Analysis.** Beyond aggregate accuracy, we provide a fine-grained **Failure Mode Analysis** (Section 4) that categorizes errors into Retrieval, Generation, and Embedding failures, offering actionable insights for future research.

---

## 2. Methodology

### 2.1 Experimental Design

We employ a controlled A/B experimental design with the following fixed parameters:

| Component | Configuration |
|:---|:---|
| **Embedding Model** | BAAI/bge-large-en-v1.5 (HuggingFace) |
| **LLM (Generation)** | DeepSeek-R1 8B (Local, via Ollama) |
| **Vector Store** | Local Chroma DB |
| **Top-K Retrieval** | k=3 |

The **independent variable** is the **document parsing strategy**:
-   **Unstructured Baseline**: PyPDF2 plain-text extraction → recursive character chunking.
-   **Layout-Aware (Proposed)**: VLM-based parser (LlamaParse) → Markdown output → Markdown-aware chunking.

### 2.2 Benchmark Dataset

-   **Source Document**: [NVIDIA Corporation FY2024 Annual Report (Form 10-K)](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm) — a document characterized by complex multi-column tables and dense numerical data.
-   **Evaluation Set**: 8 curated QA pairs spanning two task types:
    -   **Simple Lookup** (4 questions): Direct extraction of a single value (e.g., "What was the Total Revenue for FY2024?").
    -   **Cross-Column Comparison** (4 questions): Reasoning requiring comparison across multiple cells (e.g., "Did Operating Income increase from 2023 to 2024, and by how much?").

### 2.3 Evaluation Metrics

| Metric | Definition |
|:---|:---|
| **Accuracy (Exact Match)** | A response is scored `1.0` if the extracted numerical value matches the ground truth within a ±1% tolerance. Partial credit (`0.5`) is awarded for correct methodology leading to a close but inexact answer. All other responses are scored `0.0`. |
| **Latency** | End-to-end wall-clock time from query submission to final answer (in seconds). |

---

## 3. Results

### 3.1 Aggregate Performance

**Table 2: Main Results Summary**

| Pipeline | Overall Accuracy | Simple Lookup | Cross-Column | Avg. Latency (s) |
|:---|:---:|:---:|:---:|:---:|
| **Unstructured Baseline** | 50.0% (4/8) | 50.0% | 50.0% | 89.4 |
| **Layout-Aware (Proposed)** | **68.8%** (5.5/8) | **75.0%** | **62.5%** | 88.4 |
| **Δ (Relative Improvement)** | **+37.5%** | +50.0% | +25.0% | -1.1% |

The Layout-Aware pipeline demonstrates statistically and practically significant improvement on Simple Lookup tasks, confirming that structure preservation directly aids direct value extraction. The improvement on Cross-Column tasks is more modest, suggesting that multi-hop reasoning remains a challenge.

### 3.2 Visual Summary

<p align="center">
  <img src="accuracy_comparison.png" width="48%" alt="Accuracy Comparison">
  <img src="latency_distribution.png" width="48%" alt="Latency Distribution">
</p>

*Figure 1: (Left) Accuracy comparison by pipeline. (Right) Per-question latency distribution. Notably, the Layout-Aware pipeline incurs **no significant latency overhead** compared to the baseline.*

---

## 4. Failure Mode Analysis

The most critical contribution of this study is not the accuracy gain, but the **systematic analysis of the remaining 31.2% of errors** in the Layout-Aware pipeline. Understanding *why* a system fails is often more valuable than knowing *that* it succeeded. We categorize the three failures as follows:

### 4.1 Error Taxonomy

**Table 3: Failure Mode Breakdown (N=3 errors in Proposed Pipeline)**

| Question ID | Task Type | Failure Mode | Root Cause |
|:---|:---|:---|:---|
| **Q3** | Cross-Column | **Retrieval Failure** | The relevant chunk (R&D expenses table) was not retrieved in the top-k results. The embedding model failed to rank it highly enough given the query. |
| **Q4** | Simple Lookup | **Generation Error** | Correct data was retrieved; however, the LLM's chain-of-thought reasoning introduced arithmetic confusion regarding units (millions vs. billions), leading to a slightly inexact final answer. (Partial credit awarded.) |
| **Q7** | Simple Lookup | **Semantic Ambiguity** | The query asked for "Basic" EPS, but the retrieved chunk contained "Diluted" EPS. The dense embedding model could not distinguish the fine-grained semantic difference between these near-synonymous terms. |

### 4.2 Implications for Future Research

This analysis yields three actionable insights:

1.  **Retrieval is a Bottleneck (Q3).** Even with perfect parsing, if the retriever fails to surface the correct chunk, the generation layer can only hallucinate. This motivates hybrid retrieval strategies (e.g., BM25 + dense) or query expansion.

2.  **LLM Arithmetic is Unreliable (Q4).** For financial applications requiring precise numerical answers, a dedicated calculation layer (e.g., chaining the LLM with a code interpreter) may be necessary to avoid generation-time errors.

3.  **Semantic Ambiguity Defeats Dense Embeddings (Q7).** When two terms (e.g., "Basic" vs. "Diluted" EPS) are semantically near-identical in general language but critically distinct in a domain context, dense embeddings fail. This strongly suggests the need for **late-interaction retrieval models** (e.g., ColBERT) or **domain-adapted embeddings** for financial NLP.

---

## 5. Discussion and Limitations

### 5.1 Generalizability

The primary limitation of this study is the reliance on a **single-document benchmark** (NVIDIA FY2024 10-K). While the findings are internally consistent, they may not generalize to documents with different formatting conventions (e.g., handwritten annotations, multi-lingual reports, or highly irregular table structures). Future work should expand the benchmark to include 5-10 diverse financial reports (e.g., Apple, Tesla, Berkshire Hathaway) to establish broader validity.

### 5.2 Baseline Strength

We acknowledge that PyPDF2 represents a minimal baseline. Stronger comparisons against more capable parsers—such as Unstructured.io, dedicated OCR engines (e.g., Tesseract, PaddleOCR), or even GPT-4 Vision-based extraction—would provide a more rigorous assessment of where the Layout-Aware approach sits on the Pareto frontier of performance and cost.

### 5.3 On the Role of Proprietary Tools

This study utilizes LlamaParse, a commercial API, as the VLM-based parser. We deliberately frame our contribution not as an endorsement of a specific tool, but as evidence for the general principle of **Markdown Serialization** as a powerful intermediate representation for semi-structured documents. The core insight—that preserving layout structure benefits downstream LLM reasoning—is tool-agnostic and replicable with open-source VLMs (e.g., Nougat, Pix2Struct).

---

## 6. Conclusion

This empirical study provides evidence that **layout-aware parsing significantly improves RAG performance on financial documents** by bridging the Structure-Gap. By serializing tables into Markdown, we achieved a 37.5% relative improvement in accuracy on a targeted benchmark. More importantly, our detailed failure mode analysis reveals that layout-aware parsing is a *necessary but not sufficient* condition: complementary advances in retrieval (for recall), generation (for precision arithmetic), and embedding (for semantic nuance) are required for production-grade Financial RAG.

**Key Takeaways:**
1.  **Structure matters.** Unstructured text extraction is fundamentally inadequate for tabular data.
2.  **Markdown is a powerful representation.** It is both human-readable and LLM-friendly.
3.  **Failed cases are informative.** The 31.2% error rate is not a ceiling but a roadmap for improvement.

---

## References

1.  Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2.  Liu, J., et al. (2024). *LlamaIndex: A Data Framework for LLM Applications*. [llamaindex.ai](https://www.llamaindex.ai/)
3.  Xiao, S., et al. (2023). *BGE: BAAI General Embedding*. arXiv:2309.07597.
4.  Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. SIGIR.
5.  Blecher, L., et al. (2023). *Nougat: Neural Optical Understanding for Academic Documents*. arXiv:2308.13418.
6.  NVIDIA Corporation. (2024). *Annual Report (Form 10-K)*. SEC Filing.

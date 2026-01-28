# Technical Report: Addressing the Structure-Gap in Financial RAG Systems

> 📄 **[View Academic PDF Report (HTML) →](Technical_Report_Structure_Aware_RAG.html)** — *Open in browser and print to PDF for best results*

## Abstract
**Problem**: Financial documents (e.g., 10-Ks, earnings reports) are highly heterogeneous, heavily relying on complex tables to convey critical information. Standard RAG pipelines, which treat documents as flat text, often fail to preserve the structural relationships within these tables, leading to high hallucination rates in retrieval and generation.

**Method**: We refer to this as the "Structure-Gap" and propose a **Structure-Aware Parsing Pipeline**. This approach leverages **LlamaParse** to extract documents into Markdown-formatted text, ostensibly preserving tabular structure, coupled with specific prompting strategies.

**Results**: Evaluated on the NVIDIA Fiscal Year 2024 Financial Report, our Structure-Aware pipeline achieved an accuracy of **68.8%**, significantly outperforming the naive PyPDF baseline (50.0%). The proposed method demonstrated superior capability in **Cross-Row Comparison** and **Derived Metric Calculation**.

## 1. Methodology

### 1.1 Experiment Design Matrix
We employed a controlled 2x1x2 experimental design:
- **Parsers**: PyPDF (Baseline) vs. LlamaParse (Proposed).
- **Dataset**: NVIDIA 2024 Annual Report (Complex Tables & Multi-column Layouts).
- **Evaluation**: Human-Verified Accuracy & End-to-End Latency.

### 1.2 Case Study: Parsing Quality
The core differentiator is how tables are ingested.
- **PyPDF (Baseline)**: Often outputs a "bag of words," losing row/column alignment. For example, a balance sheet row `revenue 10 20` might clearly separate years visually, but PyPDF collapses them to `revenue 10 20`, enticing the LLM to guess which number belongs to which year.
- **LlamaParse (Proposed)**: Generates a distinct Markdown table (`| Revenue | 2023 | 2024 |`), explicitly guiding the LLM's attention heads to the correct intersection.

## 2. Experimental Results

### 2.1 Main Results: Accuracy
The Structure-Aware pipeline dominates in accuracy, validating our hypothesis that structural preservation is key for financial RAG.

![Accuracy Comparison](accuracy_comparison.png)

### 2.2 System Efficiency: Latency
Higher accuracy comes at a cost. The Structure-Aware pipeline incurs higher latency due to the verbose nature of markdown and the complexity of the parsing service.

![Latency Distribution](latency_distribution.png)

### 2.3 Qualitative Analysis
We specifically analyzed performance on **Q1 (Revenue)** and **Q2 (Gross Margin)** to understand the failure modes.

- **Baseline Failure (Hallucination)**: On Q1 ("Total revenue for... Jan 28, 2024"), the Baseline model retrieved chunks where the table headers were detached from the values. It often hallucinated figures from adjacent rows (e.g., confusing "Data Center" revenue with "Total" revenue) because the spatial layout was lost so the semantic link was broken.
- **Proposed Success (Precision)**: The Structure-Aware pipeline retrieved the specific Markdown table block. The LLM correctly interpreted the `| Total Revenue | $60,922 |` text comparisons, leading to a precise, verifiable answer.

## 3. Discussion & Limitations

### 3.1 The "Semantic Collision" of Q7
**Semantic Ambiguity Error Analysis**: While structure-aware parsing successfully retrieved the correct table row, the embedding model failed to semantically distinguish between 'Basic' and 'Diluted' EPS. This suggests that structural repair is a **necessary but not sufficient** condition for Financial RAG; future work must integrate **Late Interaction models (like ColBERT)** to handle fine-grained semantic nuances.

## 4. Conclusion

This empirical study demonstrates that **structure-aware parsing significantly improves RAG performance on financial documents**. By preserving tabular structure through Markdown-based parsing, we achieved a **37.5% relative improvement** in accuracy (from 50.0% to 68.8%) on complex numerical reasoning tasks.

**Key Contributions:**
1. Identified the "Structure-Gap" problem in standard RAG pipelines for semi-structured data.
2. Proposed and validated a Structure-Aware Pipeline using LlamaParse with Markdown preservation.
3. Provided a reproducible benchmark on NVIDIA 10-K financial report.

**Future Work:**
- Integrate late-interaction retrieval models (e.g., ColBERT) to address semantic ambiguity.
- Extend evaluation to multi-document financial analysis scenarios.
- Explore hybrid approaches combining structural parsing with fine-tuned domain embeddings.
